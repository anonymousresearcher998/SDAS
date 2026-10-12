import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import gc
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.backend import select_backend
from src.env import load_env
from src.judge.factory import build_judge
from src.judge.prompts.prompts import JUDGE_SYSTEM_PROMPTS_LIB
from src.mitigation.prompts import MITIGATION_PROMPTS_LIB
from src.translator.factory import build_translator
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB
from src.seeding import item_seed

@dataclass
class ExampleResult:
    id: str
    en: str
    il_mt: str
    score_il_mt_vs_il_human: Optional[float]
    explanation_il_mt_vs_il_human: str
    translator_provider: Optional[str] = None
    judge_provider: Optional[str] = None
    seed: Optional[int] = None


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs))) if xs else 0.0

def run_pipeline_for_prompt(
    data_path: str,
    translator_model_path: str,
    judge_model_path: str,
    out_dir: str,
    limit: int | None = None,
    system_prompt: str | None = None,
    judge_model: str | None = None,
    judge_prompt_name: str | None = None,
    translator_remote: bool = False,
    judge_remote_model: str | None = None,
    mitigation_prompt: str = "",
    translator_dtype: str = "auto",
    judge_dtype: str = "auto",
    skip_judge: bool = False,
    remote_reasoning: str | None = None,
    remote_temperature: float | None = 0.4,
    prompt_name: str = "",
    base_seed: int = 1234,
):
    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("messages", [])
    if limit is not None:
        items = items[:limit]

    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "il_results.jsonl")
    out_summary = os.path.join(out_dir, "il_summary.json")

    translator = build_translator(
        model_id=translator_model_path,
        system_prompt=system_prompt or "",
        remote=translator_remote,
        language="Hebrew",
        country="Israel",
        mitigation_prompt=mitigation_prompt,
        dtype=translator_dtype,
        remote_reasoning=remote_reasoning,
        remote_temperature=remote_temperature,
    )
    translator_provenance = getattr(translator, "provenance", None)

    il_mt_list: List[str] = []
    ids: List[str] = []
    en_list: List[str] = []
    il_human_list: List[str] = []
    tr_provider_list: List[Optional[str]] = []
    seed_by_id: Dict[str, int] = {}

    for idx, ex in enumerate(items):
        if "neutral" not in ex or "il" not in ex:
            continue
        ex_id = ex.get("id", f"ex_{idx:05d}")
        en = ex["neutral"].strip()
        il_human = ex["il"].strip()

        seed = item_seed(base_seed, prompt_name, idx)
        seed_by_id[ex_id] = seed
        il_mt = translator.translate_en(en, seed=seed)

        ids.append(ex_id)
        en_list.append(en)
        il_human_list.append(il_human)
        il_mt_list.append(il_mt)
        tr_provider_list.append(getattr(translator, "last_provider", None))

    del translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if skip_judge:
        print("Judge skipped (--skip_judge): writing translations with score=null", flush=True)
        judge = None
        judge_provenance = None
    else:
        judge = build_judge(
            local_model_path=judge_model_path,
            remote_model=judge_remote_model,
            system_prompt=JUDGE_SYSTEM_PROMPTS_LIB[judge_prompt_name],
            dtype=judge_dtype,
        )
        judge_provenance = getattr(judge, "provenance", None)

    results: List[ExampleResult] = []
    scores: List[float] = []

    for ex_id, en, il_human, il_mt, tr_provider in zip(
        ids, en_list, il_human_list, il_mt_list, tr_provider_list
    ):
        if judge is None:
            results.append(
                ExampleResult(
                    id=ex_id,
                    en=en,
                    il_mt=il_mt,
                    score_il_mt_vs_il_human=None,
                    explanation_il_mt_vs_il_human="",
                    translator_provider=tr_provider,
                    judge_provider=None,
                )
            )
            continue
        try:
            jres = judge.judge(language="Hebrew", sentence_a=il_mt, sentence_b=il_human)
            score = jres.get("score", None)
            if score is None:
                raise ValueError("No score returned by judge")

            expl = jres.get("explanation", "")

            scores.append(score)
            results.append(
                ExampleResult(
                    id=ex_id,
                    en=en,
                    il_mt=il_mt,
                    score_il_mt_vs_il_human=score,
                    explanation_il_mt_vs_il_human=expl,
                    translator_provider=tr_provider,
                    judge_provider=getattr(judge, "last_provider", None),
                )
            )
        except Exception as e:
            print(f"Pipeline-level fallback due to exception: {type(e).__name__}: {e}")
            continue

    for r in results:
        r.seed = seed_by_id.get(r.id)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "n": len(results),
        "mean_score_IL_MT_vs_IL_POV": None if skip_judge else round(mean(scores), 4),
        "convention": "Lower score = closer to IL POV = more biased toward IL.",
    }
    if skip_judge:
        summary["judge_skipped"] = True
    summary["base_seed"] = base_seed
    openrouter: Dict[str, Any] = {}
    for key, acc in (("translator", translator_provenance), ("judge", judge_provenance)):
        block = acc.summary() if acc is not None else None
        if block:
            openrouter[key] = block
    if openrouter:
        summary["openrouter"] = openrouter

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_summary}")

def run_pipeline(
    data_path: str,
    translator_model_path: str,
    translator_model_name: str,
    judge_model_path: str,
    out_dir: str,
    limit: int | None = None,
    judge_model: str | None = None,
    judge_prompt_name: str | None = None,
    translator_remote: bool = False,
    judge_remote_model: str | None = None,
    mitigation: str = "baseline",
    translator_dtype: str = "auto",
    judge_dtype: str = "auto",
    skip_judge: bool = False,
    remote_reasoning: str | None = None,
    remote_temperature: float | None = 0.4,
    base_seed: int = 1234,
):
    for prompt_name, prompt_template in SYSTEM_PROMPTS_LIB.items():
        print(f"Prompt: {prompt_name}")
        print(f"Template: {prompt_template}")

        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        out_dir_ts = os.path.join(out_dir, mitigation, f"{judge_model}_judge", judge_prompt_name, translator_model_name, prompt_name, f"il_run_{run_ts}")

        print("Output directory:", out_dir_ts)

        run_pipeline_for_prompt(
            data_path=data_path,
            translator_model_path=translator_model_path,
            judge_model_path=judge_model_path,
            out_dir=out_dir_ts,
            limit=limit,
            system_prompt=prompt_template,
            judge_model=judge_model,
            judge_prompt_name=judge_prompt_name,
            translator_remote=translator_remote,
            judge_remote_model=judge_remote_model,
            mitigation_prompt=MITIGATION_PROMPTS_LIB[mitigation],
            translator_dtype=translator_dtype,
            judge_dtype=judge_dtype,
            skip_judge=skip_judge,
            remote_reasoning=remote_reasoning,
            remote_temperature=remote_temperature,
            prompt_name=prompt_name,
            base_seed=base_seed,
        )


def main():
    load_env()

    ap = argparse.ArgumentParser(description="EN->IL bias pipeline (MT vs human IL POV, judged locally or via OpenRouter).")
    ap.add_argument("--data", help="Path to JSON with {'messages': [{'neutral','il',...}, ...]}",
                    default="")
    ap.add_argument("--translator_model_name", help="Model name for translation")
    ap.add_argument("--local_translator_model", help="Local HF path for the translator model",
                    default="")
    ap.add_argument("--remote_translator_model", help="OpenRouter slug for the translator",
                    default=None)
    ap.add_argument("--local_judge_model", help="Local path used by LocalJudge",
                    default="")
    ap.add_argument("--remote_judge_model", help="OpenRouter slug for the judge",
                    default=None)
    ap.add_argument("--out_dir", help="Directory to write outputs (JSONL + summary JSON)",
                    default="")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    ap.add_argument("--judge_model", type=str, default=None, help="Judge name used as the results-bucket directory")
    ap.add_argument("--judge_prompt_name", type=str, default=None, help="Judge prompt name from JUDGE_SYSTEM_PROMPTS_LIB")
    ap.add_argument("--mitigation", type=str, default="baseline", choices=sorted(MITIGATION_PROMPTS_LIB),
                    help="Prompt-level intervention prepended to the translator system prompt; "
                         "'baseline' prepends nothing.")
    ap.add_argument("--translator_dtype", type=str, default="auto",
                    help="Local translator load dtype: auto | float32 | bfloat16 | float16 [auto]")
    ap.add_argument("--judge_dtype", type=str, default="auto",
                    help="Local judge (LocalJudge) load dtype: auto | float32 | bfloat16 | float16 [auto]")
    ap.add_argument("--remote_reasoning", type=str, default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                    help="Remote translator only: OpenRouter reasoning effort; 'none' disables reasoning; omitted = model default")
    ap.add_argument("--remote_temperature", type=str, default="0.4")
    ap.add_argument("--skip_judge", action="store_true",
                    help="Translate only: write rows with score=null and no judge model loaded; "
                         "re-judge later with en_to_il_same.py --precomputed_mt_root")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    translator_remote, translator_model_id = select_backend(
        local=args.local_translator_model,
        remote=args.remote_translator_model,
        what="translator",
        local_flag="--local_translator_model",
        remote_flag="--remote_translator_model",
    )
    judge_remote, _ = select_backend(
        local=args.local_judge_model,
        remote=args.remote_judge_model,
        what="judge",
        local_flag="--local_judge_model",
        remote_flag="--remote_judge_model",
    )

    print("Running EN->il bias pipeline with the following settings:")
    print(f"  Data: {args.data}")
    print(f"  Mitigation: {args.mitigation}")
    print(f"  Base seed: {args.seed}")
    print(f"  Translator ({'REMOTE' if translator_remote else 'LOCAL'}): {translator_model_id}")
    print(f"  Judge ({'REMOTE' if judge_remote else 'LOCAL'}): "
          f"{args.remote_judge_model if judge_remote else args.local_judge_model}")
    print(f"  Judge bucket: {args.judge_model}")
    print(f"  Judge prompt: {args.judge_prompt_name}")
    if translator_remote or judge_remote:
        from src.openrouter.client import OpenRouterSettings
        print(f"  OpenRouter: {OpenRouterSettings.from_env().describe()}")
    if translator_remote:
        print(f"  Remote reasoning: {args.remote_reasoning or 'model default'}  Remote temperature: {args.remote_temperature}")
    if args.limit is not None:
        print(f"  Limit: {args.limit}")
    if args.translator_dtype != "auto" or args.judge_dtype != "auto" or args.skip_judge:
        print(f"  Translator dtype: {args.translator_dtype}  Judge dtype: {args.judge_dtype}  Skip judge: {args.skip_judge}")

    run_pipeline(
        data_path=args.data,
        translator_model_path=translator_model_id,
        translator_model_name=args.translator_model_name,
        judge_model_path=args.local_judge_model,
        out_dir=args.out_dir,
        limit=args.limit,
        judge_model=args.judge_model,
        judge_prompt_name=args.judge_prompt_name,
        translator_remote=translator_remote,
        judge_remote_model=args.remote_judge_model,
        mitigation=args.mitigation,
        translator_dtype=args.translator_dtype,
        judge_dtype=args.judge_dtype,
        skip_judge=args.skip_judge,
        remote_reasoning=args.remote_reasoning,
        remote_temperature=None if str(args.remote_temperature).lower() == "none" else float(args.remote_temperature),
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
