from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.backend import select_backend
from src.env import load_env
from src.judge.factory import build_judge
from src.judge.prompts.prompts import JUDGE_SYSTEM_PROMPTS_LIB
from src.pipeline.sdas._common import (
    add_data_arg,
    add_experiment_dir_arg,
    add_judge_args,
    assert_data,
    banner,
    ids_label,
    resolve_ids,
)
from src.sdas.data import load_items
from src.sdas.export_test import default_export_root, export_run
from src.sdas.layout import (
    ExperimentLayout,
    atomic_write_json,
    atomic_write_text,
    load_experiment,
    now_ts,
    read_json,
    read_jsonl,
    unique_run_dir,
    write_run_record,
)
from src.sdas.prompts import CONVENTION, LANG_CODES, MEAN_KEY, assert_lang_in_pair, experiment_pair, lang_cfg
from src.sdas.stats import mean


_FLAT_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def rescue_score(text: str) -> Optional[Tuple[float, str]]:
    best: Optional[Tuple[float, str]] = None
    for match in _FLAT_JSON_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            try:
                score = float(obj["score"])
            except (TypeError, ValueError):
                continue
            best = (max(0.0, min(1.0, score)), str(obj.get("explanation", "")))
    return best


def safe_judge(judge: Any, language: str, a: str, b: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    try:
        jres = judge.judge(language=language, sentence_a=a, sentence_b=b)
        score = jres.get("score", None)
        if score is None:
            raise ValueError("No score returned by judge")
        return float(score), str(jres.get("explanation", "")), None
    except Exception as exc:
        rescued = rescue_score(str(exc))
        if rescued is not None:
            print(f"  judge output rescued from a parse failure ({type(exc).__name__})", flush=True)
            return rescued[0], rescued[1] + " [rescued from parse failure]", None
        message = f"{type(exc).__name__}: {exc}"
        print(f"  judge failure: {message}", flush=True)
        return None, None, message


def existing_complete_run(prompt_dir, lang: str, expected_ids: List[str]) -> Optional[str]:
    if not prompt_dir.is_dir():
        return None
    wanted = set(expected_ids)
    runs = sorted(p for p in prompt_dir.iterdir() if p.is_dir() and p.name.startswith(f"{lang}_run_"))
    for run in reversed(runs):
        results = run / f"{lang}_results.jsonl"
        if not results.is_file() or not (run / f"{lang}_summary.json").is_file():
            continue
        try:
            have = {r["id"] for r in read_jsonl(results) if "id" in r}
        except (OSError, json.JSONDecodeError):
            continue
        if have >= wanted:
            return run.name
    return None


def remove_aborted_runs(prompt_dir, lang: str) -> List[str]:
    if not prompt_dir.is_dir():
        return []
    removed: List[str] = []
    for run in sorted(p for p in prompt_dir.iterdir() if p.is_dir() and p.name.startswith(f"{lang}_run_")):
        if not (run / f"{lang}_results.jsonl").is_file():
            shutil.rmtree(run, ignore_errors=True)
            removed.append(run.name)
    return removed


def judge_condition(
    *,
    judge: Any,
    layout: ExperimentLayout,
    condition: str,
    lang: str,
    ids: Optional[List[str]],
    ids_filter: str,
    translator: str,
    judge_bucket: str,
    judge_prompt_name: str,
    same_prompt: str,
    overwrite: bool,
) -> None:
    lc = lang_cfg(lang)
    leaves = [(lg, prompt_name, gen_dir) for lg, prompt_name, gen_dir in layout.list_leaves(condition) if lg == lang]
    if not leaves:
        raise SystemExit(f"condition {condition!r} has no {lang} leaves under {layout.generations_dir}; known conditions: {layout.list_conditions()}")

    same_key = f"score_{lang}_mt_vs_{lang}_human"
    same_expl = f"explanation_{lang}_mt_vs_{lang}_human"

    for _, prompt_name, gen_dir in leaves:
        rows = read_jsonl(gen_dir / "gen.jsonl")
        if ids is not None:
            wanted = set(ids)
            rows = [r for r in rows if r["id"] in wanted]
        if not rows:
            print(f"[{condition}] {prompt_name}: no rows for the requested ids; skipping", flush=True)
            continue
        prompt_dir = layout.judged_prompt_dir(condition, judge_bucket, judge_prompt_name, translator, prompt_name)
        done = None if overwrite else existing_complete_run(prompt_dir, lang, [r["id"] for r in rows])
        if done:
            print(f"[{condition}] {prompt_name}: already judged in {done} ({len(rows)} rows); skipping", flush=True)
            continue
        aborted = remove_aborted_runs(prompt_dir, lang)
        if aborted:
            print(f"[{condition}] {prompt_name}: removed {len(aborted)} aborted run dir(s): {', '.join(aborted)}", flush=True)

        run_dir = unique_run_dir(prompt_dir, lang)
        out_rows: List[Dict[str, Any]] = []
        same_scores: List[float] = []
        n_fail = 0
        t_leaf = time.time()
        for k, r in enumerate(rows):
            mt = r["mt"]
            ref_same = r[f"ref_{lang}"]
            errors: List[str] = []

            s_same, e_same, err = safe_judge(judge, lc.language, mt, ref_same)
            if err:
                errors.append(f"same: {err}")
            row: Dict[str, Any] = {
                "id": r["id"],
                "en": r["en"],
                f"{lang}_mt": mt,
                same_key: s_same,
                same_expl: e_same,
                "translator_provider": None,
                "judge_provider": getattr(judge, "last_provider", None),
                "condition": condition,
            }
            if errors:
                row["judge_error"] = " | ".join(errors)
            if s_same is None:
                n_fail += 1
            else:
                same_scores.append(s_same)
            out_rows.append(row)
            if (k + 1) % 10 == 0 or k + 1 == len(rows):
                print(f"[{condition}] {prompt_name} {k + 1}/{len(rows)} judged ({time.time() - t_leaf:.0f}s)", flush=True)

        atomic_write_text(run_dir / f"{lang}_results.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows))
        summary: Dict[str, Any] = {
            "n": len(same_scores),
            MEAN_KEY[lang]: round(mean(same_scores), 4) if same_scores else 0.0,
            "convention": CONVENTION[lang],
            "n_rows": len(out_rows),
            "n_judge_failures": n_fail,
            "share_score_ge_0.5": round(sum(1 for s in same_scores if s >= 0.5) / len(same_scores), 4) if same_scores else None,
            "condition": condition,
            "lang": lang,
            "prompt_name": prompt_name,
            "judge_prompt": judge_prompt_name,
            "ids_filter": ids_filter,
            "created": now_ts(),
        }
        provenance = getattr(judge, "provenance", None)
        block = provenance.summary() if provenance is not None else None
        if block:
            summary["openrouter"] = {"judge": block}
        atomic_write_json(run_dir / f"{lang}_summary.json", summary)
        print(f"[{condition}] {prompt_name}: n={summary['n']} mean_same={summary[MEAN_KEY[lang]]} "
              f"failures={n_fail} -> {run_dir}", flush=True)
        if ids_filter == "test" and condition.strip("/").startswith("sdas/"):
            dest = export_run(run_dir, condition=condition, translator=translator, lang=lang, judge_bucket=judge_bucket, judge_prompt=judge_prompt_name, dest_root=default_export_root(layout))
            if dest is not None:
                print(f"[{condition}] {prompt_name}: exported test run -> {dest}", flush=True)


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Stage D of the SDAS flow: same-side judging.")
    add_experiment_dir_arg(ap)
    ap.add_argument("--condition", action="append", required=True, help="Condition slug (repeatable): regular | prompt/<name> | sdas/<dir>/L16_a2")
    ap.add_argument("--lang", required=True, choices=list(LANG_CODES), help="ru | ua (RU/UA experiments) or ps | il (IL/PS)")
    ap.add_argument("--ids", default="val", help="val | test | train | all | comma-separated ids | file [val]")
    add_judge_args(ap)
    add_data_arg(ap)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N items of each set, as Stage A --limit (smoke tests)")
    ap.add_argument("--overwrite", action="store_true", help="Judge again even if a complete run exists (adds a new timestamped run)")
    ap.add_argument("--allow_references", action="store_true", help="Also judge `regular` / `prompt/<intervention>`")
    args = ap.parse_args()
    references = [c for c in args.condition if c.strip("/") == "regular" or c.strip("/").startswith("prompt/")]
    if references and not args.allow_references:
        args.condition = [c for c in args.condition if c not in references]
        if not args.condition:
            print("[judge] nothing left to judge; exiting.", flush=True)
            return

    layout = ExperimentLayout.from_dir(args.experiment_dir)
    experiment = load_experiment(layout)
    lc = lang_cfg(args.lang)
    pair = experiment_pair(experiment)
    assert_lang_in_pair(args.lang, pair)
    judge_remote, judge_model_id = select_backend(
        local=args.local_judge_model, remote=args.remote_judge_model, what="judge",
        local_flag="--local_judge_model", remote_flag="--remote_judge_model",
    )
    if args.judge_prompt_name not in JUDGE_SYSTEM_PROMPTS_LIB:
        raise SystemExit(f"unknown --judge_prompt_name {args.judge_prompt_name}; choose from {sorted(JUDGE_SYSTEM_PROMPTS_LIB)}")
    assert_data(experiment, args.data)
    items = load_items(args.data, args.limit, langs=pair.langs)
    ids = resolve_ids(args.ids, [it.id for it in items])
    ids_filter = None if args.ids.strip().lower() == "all" else ids

    same_prompt = JUDGE_SYSTEM_PROMPTS_LIB[args.judge_prompt_name]

    banner(
        "Stage D - judge",
        [
            ("experiment", str(layout.root)),
            ("conditions", ", ".join(args.condition)),
            ("lang", f"{lc.code} ({lc.language})"),
            ("ids", f"{len(ids)} ({args.ids})"),
            ("judge", f"{'REMOTE' if judge_remote else 'LOCAL'} {judge_model_id} (bucket {args.judge_model})"),
            ("judge prompt", args.judge_prompt_name),
        ],
    )
    if judge_remote:
        from src.openrouter.client import OpenRouterSettings

        print(f"  OpenRouter: {OpenRouterSettings.from_env().describe()}")
    write_run_record(layout, "judge", {"args": vars(args), "n_ids": len(ids)})

    judge = build_judge(
        local_model_path=args.local_judge_model,
        remote_model=args.remote_judge_model,
        system_prompt=same_prompt,
    )
    translator = experiment["translator_model_name"]
    for condition in args.condition:
        judge_condition(
            judge=judge,
            layout=layout,
            condition=condition,
            lang=lc.code,
            ids=ids_filter,
            ids_filter=ids_label(args.ids, ids),
            translator=translator,
            judge_bucket=args.judge_model,
            judge_prompt_name=args.judge_prompt_name,
            same_prompt=same_prompt,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
