from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.annotation.prompts import (
    ANNOTATOR_REASONING_EFFORT,
    MECHANISM_LABELER_PROMPT,
    MECHANISM_LABELS,
    SEMANTIC_PRESERVATION_PROMPT,
)
from src.env import load_env
from src.openrouter.client import OpenRouterSettings, build_client, chat_completion
from src.sdas.layout import read_jsonl
from src.translator.openrouter_translator import reasoning_block


def last_run(prompt_dir: Path, lang: str) -> Optional[Path]:
    runs = sorted(p for p in prompt_dir.glob(f"{lang}_run_*") if (p / f"{lang}_results.jsonl").is_file())
    return runs[-1] if runs else None


def messages(task: str, en: str, mt: str) -> List[Dict[str, str]]:
    if task == "preservation":
        text = SEMANTIC_PRESERVATION_PROMPT.replace("{neutral_english_statement}", en.strip()).replace("{generated_output}", mt.strip())
        return [{"role": "user", "content": text}]
    user = f"Neutral English event statement:\n{en.strip()}\n\nGenerated translation:\n{mt.strip()}"
    return [{"role": "system", "content": MECHANISM_LABELER_PROMPT}, {"role": "user", "content": user}]


def parse(task: str, text: str) -> Dict[str, Any]:
    if task == "preservation":
        t = text.strip().upper()
        return {"label": "NOT PRESERVED" if "NOT PRESERVED" in t else "PRESERVED" if "PRESERVED" in t else None}
    obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return {
        "labels": [l for l in obj.get("labels", []) if l in MECHANISM_LABELS],
        "justification": obj.get("justification", ""),
        "evidence": obj.get("evidence", []),
    }


def summarize(rows: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("error") is None]
    out: Dict[str, Any] = {"n": len(ok), "errors": len(rows) - len(ok)}
    if task == "preservation":
        out["preserved_rate"] = round(sum(r["label"] == "PRESERVED" for r in ok) / max(1, len(ok)), 4)
    else:
        counts = Counter(l for r in ok for l in r["labels"])
        out["any_label_rate"] = round(sum(bool(r["labels"]) for r in ok) / max(1, len(ok)), 4)
        out.update({l: round(counts[l] / max(1, len(ok)), 4) for l in MECHANISM_LABELS})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="GPT-5.6-Luna mechanism labels (App. G) or semantic preservation (App. E) of stored outputs")
    ap.add_argument("--task", required=True, choices=["mechanisms", "preservation"])
    ap.add_argument("--root", required=True, help="results | results_il_ps")
    ap.add_argument("--langs", required=True, help="Comma-separated, e.g. ru,ua")
    ap.add_argument("--annotator", required=True, help="OpenRouter slug of the annotator")
    ap.add_argument("--reasoning", default=ANNOTATOR_REASONING_EFFORT)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--trees", default="baseline,control,self_distance,sdas_test")
    ap.add_argument("--models", default="", help="Comma-separated model folders [all]")
    ap.add_argument("--judge_name", default="llama3-8b")
    ap.add_argument("--judge_prompt", default="BASE_JUDGE")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_env()
    settings = OpenRouterSettings.from_env()
    client = build_client(settings)
    root = Path(args.root)
    models = {m for m in args.models.split(",") if m}
    summary: List[Dict[str, Any]] = []

    for tree in [t for t in args.trees.split(",") if t]:
        base = root / tree / f"{args.judge_name}_judge" / args.judge_prompt
        if not base.is_dir():
            print(f"skip {tree}: no runs under {base}")
            continue
        for model_dir in sorted(p for p in base.iterdir() if p.is_dir() and (not models or p.name in models)):
            for lang in args.langs.split(","):
                rows_all: List[Dict[str, Any]] = []
                for prompt_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                    run = last_run(prompt_dir, lang)
                    if run is None:
                        continue
                    out = root / "annotations" / args.task / tree / model_dir.name / prompt_dir.name / f"{lang}.jsonl"
                    done = {r["id"]: r for r in read_jsonl(out) if r.get("error") is None}
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with open(out, "w", encoding="utf-8") as f:
                        for src in read_jsonl(run / f"{lang}_results.jsonl")[: args.limit]:
                            row = done.get(src["id"])
                            if row is None:
                                row = {"id": src["id"], "source_run": run.name, "error": None}
                                try:
                                    text, prov = chat_completion(
                                        client, settings, model=args.annotator,
                                        messages=messages(args.task, src["en"], src[f"{lang}_mt"]),
                                        reasoning=reasoning_block(args.reasoning), max_tokens=args.max_tokens,
                                    )
                                    row.update(parse(args.task, text), raw=text, provider=prov.get("provider"))
                                except Exception as exc:
                                    row["error"] = f"{type(exc).__name__}: {exc}"
                                    print(f"[{tree}/{model_dir.name}/{prompt_dir.name}/{lang}] {src['id']}: {row['error']}")
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows_all.append(row)
                if rows_all:
                    summary.append({"tree": tree, "model": model_dir.name, "lang": lang, **summarize(rows_all, args.task)})
                    print(summary[-1])

    if summary:
        path = root / "annotations" / args.task / f"summary_{'_'.join(args.langs.split(','))}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print(f"summary: {path}")


if __name__ == "__main__":
    main()
