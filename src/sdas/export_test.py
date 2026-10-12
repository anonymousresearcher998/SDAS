from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.sdas.data import split_of
from src.sdas.layout import ExperimentLayout, atomic_write_json, atomic_write_text, format_alpha, parse_condition, read_jsonl
from src.sdas.prompts import CONVENTION, MEAN_KEY

DEFAULT_EXPORT_MITIGATION = "sdas_test"


def pseudo_model(translator: str, condition: str) -> str:
    parsed = parse_condition(condition)
    if parsed["kind"] != "sdas":
        raise ValueError(f"only steer cells are exported (got {condition!r})")
    cell = f"L{parsed['layer']:02d}_a{format_alpha(parsed['alpha'])}"
    direction = parsed["direction"]
    return f"{translator}_{cell}" if direction == "self_distance_v1" else f"{translator}_{direction}_{cell}"


def default_export_root(layout: ExperimentLayout) -> Path:
    return layout.root.parent.parent / DEFAULT_EXPORT_MITIGATION


def old_id(item_id: str) -> str:
    if split_of(item_id) != "test":
        raise ValueError(f"only test ids are exported (got {item_id!r})")
    return "ex_" + item_id.split("_", 1)[1]


def export_run(run_dir: Path, *, condition: str, translator: str, lang: str, judge_bucket: str, judge_prompt: str, dest_root: Path) -> Optional[Path]:
    rows = read_jsonl(run_dir / f"{lang}_results.jsonl")
    if not rows or any(split_of(r["id"]) != "test" for r in rows):
        return None
    score_key = f"score_{lang}_mt_vs_{lang}_human"
    expl_key = f"explanation_{lang}_mt_vs_{lang}_human"
    out_rows: List[Dict] = []
    for r in rows:
        if r.get(score_key) is None:
            continue
        out_rows.append({"id": old_id(r["id"]), "en": r["en"], f"{lang}_mt": r[f"{lang}_mt"], score_key: r[score_key], expl_key: r.get(expl_key, "")})
    out_rows.sort(key=lambda r: r["id"])
    prompt_name = run_dir.parent.name
    dest = dest_root / f"{judge_bucket}_judge" / judge_prompt / pseudo_model(translator, condition) / prompt_name / run_dir.name
    atomic_write_text(dest / f"{lang}_results.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
    scores = [r[score_key] for r in out_rows]
    atomic_write_json(
        dest / f"{lang}_summary.json",
        {"n": len(scores), MEAN_KEY[lang]: round(sum(scores) / len(scores), 4) if scores else 0.0, "convention": CONVENTION[lang]},
    )
    src_path = dest.parent.parent / "_source.json"
    src = json.load(open(src_path)) if src_path.is_file() else {"experiment": str(run_dir).split("/judged/")[0], "translator": translator, "conditions": {}, "note": "exported by src/sdas/export_test.py from the SDAS flow's judged test runs; ids ex_NNNNN = test_NNNNN; replicate/seed cells are further runs of the same cell"}
    src.setdefault("conditions", {})[run_dir.name] = condition
    atomic_write_json(src_path, src)
    return dest


def export_experiment(layout: ExperimentLayout, dest_root: Path, *, translator: str, verbose: bool = True) -> List[Path]:
    written: List[Path] = []
    steer_root = layout.judged_dir / "sdas"
    if not steer_root.is_dir():
        return written
    for results in sorted(steer_root.rglob("*_results.jsonl")):
        run_dir = results.parent
        lang = results.name.split("_", 1)[0]
        parts = run_dir.relative_to(layout.judged_dir).parts
        if len(parts) != 8 or not parts[3].endswith("_judge"):
            continue
        condition = "/".join(parts[:3])
        dest = export_run(run_dir, condition=condition, translator=parts[5], lang=lang, judge_bucket=parts[3][: -len("_judge")], judge_prompt=parts[4], dest_root=dest_root)
        if dest is not None:
            written.append(dest)
            if verbose:
                print(f"[export] {condition} {lang}/{parts[6]} {run_dir.name} -> {dest}")
    return written
