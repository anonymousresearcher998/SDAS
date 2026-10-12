from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.backend import select_backend
from src.env import load_env
from src.judge.factory import build_judge
from src.judge.prompts.prompts import JUDGE_SYSTEM_PROMPTS_LIB
from src.mitigation.prompts import MITIGATION_PROMPTS_LIB
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB

LANGS: Dict[str, str] = {"ru": "Russian", "ua": "Ukrainian", "ps": "Arabic", "il": "Hebrew"}

TREES: List[str] = sorted(MITIGATION_PROMPTS_LIB) + ["sdas_test"]


@dataclass
class ExampleResultGeneric:
    id: str
    en: str
    mt: str
    score: float
    explanation: str
    judge_provider: Optional[str] = None


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs))) if xs else 0.0


def latest_run_dir(prompt_dir: str, run_prefix: str) -> Optional[str]:
    if not os.path.isdir(prompt_dir):
        return None
    cands = [d for d in os.listdir(prompt_dir) if os.path.isdir(os.path.join(prompt_dir, d)) and d.startswith(run_prefix)]
    if not cands:
        return None
    cands.sort()
    return os.path.join(prompt_dir, cands[-1])


def load_precomputed_mt(jsonl_path: str, prompt_name: str, lang: str, mt_key: str) -> Dict[str, str]:
    mt_by_id: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r.get("id")
            if not rid:
                continue
            mt_val = r.get(mt_key)
            if mt_val is None:
                mt_val = r.get(f"{lang}_mt", r.get("mt"))
            if mt_val is None:
                continue
            mt_by_id[rid] = str(mt_val)
            pref = prompt_name + "_"
            if rid.startswith(pref):
                mt_by_id[rid[len(pref):]] = str(mt_val)
    return mt_by_id


def already_judged(dest_prompt_dir: str, lang: str, fingerprint: Dict[str, Any]) -> Optional[str]:
    latest = latest_run_dir(dest_prompt_dir, f"{lang}_run_")
    if latest is None:
        return None
    summary_path = os.path.join(latest, f"{lang}_summary.json")
    if not os.path.isfile(summary_path) or not os.path.isfile(os.path.join(latest, f"{lang}_results.jsonl")):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if all(summary.get(key) == value for key, value in fingerprint.items()):
        return latest
    return None


def judge_prompt(
    *,
    items: List[Dict[str, Any]],
    prompt_name: str,
    mt_by_id: Dict[str, str],
    lang: str,
    out_run_dir: str,
    judge_prompt_name: str,
    local_judge_model: str,
    remote_judge_model: Optional[str],
    judge_dtype: str,
    fingerprint: Dict[str, Any],
) -> None:
    language = LANGS[lang]

    ids: List[str] = []
    en_list: List[str] = []
    gt_list: List[str] = []
    mt_list: List[str] = []
    for idx, ex in enumerate(items):
        if "neutral" not in ex or lang not in ex:
            continue
        ex_id = ex.get("id", f"ex_{idx:05d}")
        mt = mt_by_id.get(ex_id)
        if mt is None:
            mt = mt_by_id.get(f"{prompt_name}_{ex_id}")
        if mt is None:
            continue
        ids.append(ex_id)
        en_list.append(ex["neutral"].strip())
        gt_list.append(ex[lang].strip())
        mt_list.append(mt)

    judge = build_judge(
        local_model_path=local_judge_model,
        remote_model=remote_judge_model,
        system_prompt=JUDGE_SYSTEM_PROMPTS_LIB[judge_prompt_name],
        dtype=judge_dtype,
    )
    judge_provenance = getattr(judge, "provenance", None)

    results: List[ExampleResultGeneric] = []
    scores: List[float] = []
    for ex_id, en, gt, mt in zip(ids, en_list, gt_list, mt_list):
        try:
            print(f"Judging ex_id={ex_id}...")
            print(f"  EN: {en}"
                  f"\n  GT: {gt}"
                  f"\n  MT: {mt}", flush=True)

            jres = judge.judge(language=language, sentence_a=mt, sentence_b=gt)
            score = jres.get("score", None)
            if score is None:
                raise ValueError("No score returned by judge")

            expl = jres.get("explanation", "")
            scores.append(float(score))
            results.append(
                ExampleResultGeneric(
                    id=ex_id,
                    en=en,
                    mt=mt,
                    score=float(score),
                    explanation=str(expl),
                    judge_provider=getattr(judge, "last_provider", None),
                )
            )
        except Exception as e:
            print(f"Pipeline-level fallback due to exception: {type(e).__name__}: {e}")
            continue

    if ids and not results:
        raise SystemExit(
            f"{prompt_name}: all {len(ids)} judge calls failed - nothing written for this prompt. "
            f"Fix the judge (key, credit, GPU) and resubmit; finished prompts are skipped."
        )

    os.makedirs(out_run_dir, exist_ok=True)
    out_jsonl = os.path.join(out_run_dir, f"{lang}_results.jsonl")
    out_summary = os.path.join(out_run_dir, f"{lang}_summary.json")

    score_key = f"score_{lang}_mt_vs_{lang}_human"
    expl_key = f"explanation_{lang}_mt_vs_{lang}_human"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            row = {
                "id": r.id,
                "en": r.en,
                f"{lang}_mt": r.mt,
                score_key: r.score,
                expl_key: r.explanation,
                "judge_provider": r.judge_provider,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "n": len(results),
        f"mean_score_{lang}_MT_vs_{lang}_GT": round(mean(scores), 4),
        "convention": "Lower score = closer to GT (by the judge definition).",
        "mode": "precomputed_mt",
        "n_expected": len(ids),
        "n_judge_failures": len(ids) - len(results),
        **fingerprint,
    }
    judge_block = judge_provenance.summary() if judge_provenance is not None else None
    if judge_block:
        summary["openrouter"] = {"judge": judge_block}
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_summary}")

    del judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    load_env()

    ap = argparse.ArgumentParser(
        description="Re-judge stored translations against the human references of --data."
    )
    ap.add_argument("--lang", required=True, choices=sorted(LANGS), help="Language of the stored MT and of the reference: ru | ua | ps | il")
    ap.add_argument("--data", required=True, help="JSON with {'messages': [{'neutral', '<lang>', ...}, ...]} - the references")
    ap.add_argument("--precomputed_mt_root", required=True,
                    help="Stored translations of ONE model: <root>/<prompt_name>/<lang>_run_*/<lang>_results.jsonl")
    ap.add_argument("--translator_model_name", required=True)
    ap.add_argument("--mitigation", required=True, choices=TREES,
                    help="Top layer of the output tree (no prompt text is used - no translation happens)")
    ap.add_argument("--out_dir", required=True, help="Results root, e.g. results | results_il_ps")
    ap.add_argument("--judge_model", required=True, help="Judge bucket name in the output path (<judge_model>_judge/)")
    ap.add_argument("--judge_prompt_name", required=True, choices=sorted(JUDGE_SYSTEM_PROMPTS_LIB))
    ap.add_argument("--local_judge_model", default="", help="Local weights of the judge (LocalJudge)")
    ap.add_argument("--remote_judge_model", default=None, help="OpenRouter slug of the judge (e.g. openai/gpt-5.1)")
    ap.add_argument("--judge_dtype", type=str, default="auto",
                    help="Local judge (LocalJudge) load dtype: auto | float32 | bfloat16 | float16 [auto]; "
                         "'auto' = the library default the existing runs used (fp32 under transformers 4.x)")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N items of --data (smoke tests)")
    ap.add_argument("--overwrite", action="store_true", help="Re-judge prompts already re-judged from the same source run")
    ap.add_argument("--prompt", default=None, choices=list(SYSTEM_PROMPTS_LIB), help="Only this translator prompt [all 13]")
    ap.add_argument("--source_run", default=None,
                    help="With --prompt: re-judge exactly this run dir")
    ap.add_argument("--precomputed_jsonl_filename", default=None, help="JSONL name inside a run dir [<lang>_results.jsonl]")
    ap.add_argument("--precomputed_mt_key", default=None, help="MT field inside the JSONL rows [<lang>_mt]")
    ap.add_argument("--precomputed_run_prefix", default=None, help="Run dir prefix [<lang>_run_]")
    args = ap.parse_args()

    judge_remote, judge_model_id = select_backend(
        local=args.local_judge_model,
        remote=args.remote_judge_model,
        what="judge",
        local_flag="--local_judge_model",
        remote_flag="--remote_judge_model",
    )
    if not os.path.isdir(args.precomputed_mt_root):
        raise SystemExit(f"--precomputed_mt_root not found: {args.precomputed_mt_root}")

    lang = args.lang
    run_prefix = args.precomputed_run_prefix or f"{lang}_run_"
    jsonl_name = args.precomputed_jsonl_filename or f"{lang}_results.jsonl"
    mt_key = args.precomputed_mt_key or f"{lang}_mt"
    if args.source_run is not None:
        if args.prompt is None:
            raise SystemExit("--source_run needs --prompt (a run dir belongs to one prompt)")
        if not args.source_run.startswith(run_prefix) or "/" in args.source_run:
            raise SystemExit(f"--source_run must be the NAME of a {run_prefix}* dir under {args.precomputed_mt_root}/{args.prompt} (got {args.source_run!r})")
        if not os.path.isdir(os.path.join(args.precomputed_mt_root, args.prompt, args.source_run)):
            raise SystemExit(f"--source_run not found: {os.path.join(args.precomputed_mt_root, args.prompt, args.source_run)}")
    prompt_names = [args.prompt] if args.prompt else list(SYSTEM_PROMPTS_LIB)

    print("Re-judge (no translation) with settings:")
    print(f"  Data: {args.data}")
    print(f"  Lang: {lang} ({LANGS[lang]})")
    if args.source_run:
        print(f"  Source: {args.precomputed_mt_root}/{args.prompt}/{args.source_run}  (exactly this run)")
    else:
        print(f"  Source: {args.precomputed_mt_root}  (latest {run_prefix}* of {args.prompt or 'every prompt'})")
    print(f"  Tree / model: {args.mitigation} / {args.translator_model_name}")
    print(f"  Judge ({'REMOTE' if judge_remote else 'LOCAL'}): {judge_model_id}")
    print(f"  Judge bucket: {args.judge_model}")
    print(f"  Judge prompt: {args.judge_prompt_name}")
    if judge_remote:
        from src.openrouter.client import OpenRouterSettings
        print(f"  OpenRouter: {OpenRouterSettings.from_env().describe()}")
    if args.limit is not None:
        print(f"  Limit: {args.limit}")
    print("Using judge prompt:")
    print(JUDGE_SYSTEM_PROMPTS_LIB[args.judge_prompt_name].format(language=LANGS[lang]))

    with open(args.data, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("messages", [])
    if args.limit is not None:
        items = items[: args.limit]

    dest_model_dir = os.path.join(
        args.out_dir, args.mitigation, f"{args.judge_model}_judge", args.judge_prompt_name, args.translator_model_name
    )
    judged: List[str] = []
    done: List[str] = []
    skipped: List[str] = []
    for prompt_name in prompt_names:
        print(f"Prompt: {prompt_name}")
        if args.source_run:
            source_run = os.path.join(args.precomputed_mt_root, prompt_name, args.source_run)
        else:
            source_run = latest_run_dir(os.path.join(args.precomputed_mt_root, prompt_name), run_prefix)
        if source_run is None:
            print(f"  SKIP {prompt_name}: no {run_prefix}* run under {args.precomputed_mt_root}/{prompt_name}", flush=True)
            skipped.append(prompt_name)
            continue
        source_run = os.path.normpath(source_run)
        source_jsonl = os.path.join(source_run, jsonl_name)
        if not os.path.isfile(source_jsonl):
            print(f"  SKIP {prompt_name}: the {'given' if args.source_run else 'latest'} run {source_run} has no {jsonl_name} (counted as missing, as in the notebooks)", flush=True)
            skipped.append(prompt_name)
            continue
        dest_prompt_dir = os.path.join(dest_model_dir, prompt_name)
        fingerprint = {"precomputed_mt_run": source_run, "limit": args.limit}
        previous = None if args.overwrite else already_judged(dest_prompt_dir, lang, fingerprint)
        if previous:
            print(f"  DONE {prompt_name}: {previous} already re-judged {source_run}; skipping (--overwrite to redo)", flush=True)
            done.append(prompt_name)
            continue

        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_run_dir = os.path.join(dest_prompt_dir, f"{lang}_run_{run_ts}")
        print(f"  Source run: {source_run}")
        print(f"  Output directory: {out_run_dir}")
        judge_prompt(
            items=items,
            prompt_name=prompt_name,
            mt_by_id=load_precomputed_mt(source_jsonl, prompt_name=prompt_name, lang=lang, mt_key=mt_key),
            lang=lang,
            out_run_dir=out_run_dir,
            judge_prompt_name=args.judge_prompt_name,
            local_judge_model=args.local_judge_model,
            remote_judge_model=args.remote_judge_model,
            judge_dtype=args.judge_dtype,
            fingerprint=fingerprint,
        )
        judged.append(prompt_name)

    print(f"Re-judge finished: {len(judged)} judged, {len(done)} already done, {len(skipped)} skipped"
          + (f" ({', '.join(skipped)})" if skipped else ""))
    if not judged and not done:
        raise SystemExit(f"nothing was judged: no prompt of {args.precomputed_mt_root} has a usable {run_prefix}* run")


if __name__ == "__main__":
    main()
