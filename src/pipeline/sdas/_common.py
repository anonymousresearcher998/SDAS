from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from src.sdas.data import DEFAULT_DATA_PATTERN, SPLITS, manifest_diffs, split_of

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DEFAULT_OUT_ROOT = "results/sdas"


def parse_csv(spec: Optional[str]) -> List[str]:
    if spec is None:
        return []
    return [tok.strip() for tok in spec.split(",") if tok.strip()]


def parse_ints(spec: Optional[str]) -> List[int]:
    return [int(tok) for tok in parse_csv(spec)]


def parse_floats(spec: Optional[str]) -> List[float]:
    return [float(tok) for tok in parse_csv(spec)]


def add_experiment_dir_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--experiment_dir",
        required=True,
        help="Path of one experiment, e.g. results/sdas/qwen25-7b",
    )


def add_data_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--data",
        default=DEFAULT_DATA_PATTERN,
        help=f"Data pattern with a {{split}} placeholder for the train/val/test files [{DEFAULT_DATA_PATTERN}]",
    )


def add_judge_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--local_judge_model", default="", help="Local path used by LocalJudge")
    ap.add_argument("--remote_judge_model", default=None, help="OpenRouter slug for the judge")
    ap.add_argument("--judge_model", type=str, required=True, help="Judge name used as the results-bucket directory")
    ap.add_argument(
        "--judge_prompt_name", type=str, default="BASE_JUDGE",
        help="Judge prompt name from JUDGE_SYSTEM_PROMPTS_LIB for the comparison",
    )


def assert_data(experiment: Dict, pattern: str) -> None:
    if "data" not in experiment and "data_sha256" in experiment:
        raise SystemExit(
            f"experiment {experiment.get('translator_model_name')!r} was created by the single-file flow "
            f"(data_basename={experiment.get('data_basename')!r});"
        )
    diffs = manifest_diffs(experiment.get("data"), pattern)
    if diffs:
        raise SystemExit(
            f"--data {pattern} does not match the data files Stage A used:\n  " + "\n  ".join(diffs)
        )


def banner(title: str, rows: Sequence[Tuple[str, object]]) -> None:
    print(title)
    width = max((len(k) for k, _ in rows), default=0)
    for key, value in rows:
        print(f"  {key.ljust(width)}  {value}")
    sys.stdout.flush()


def resolve_ids(spec: Optional[str], all_ids: List[str]) -> List[str]:
    if spec is None or not spec.strip() or spec.strip().lower() == "all":
        return list(all_ids)
    spec = spec.strip()
    if spec in SPLITS:
        return [i for i in all_ids if split_of(i) == spec]
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
    else:
        wanted = set(parse_csv(spec))
    missing = wanted - set(all_ids)
    if missing:
        raise ValueError(f"{len(missing)} requested ids are not in the data (e.g. {sorted(missing)[:3]})")
    return [i for i in all_ids if i in wanted]


def ids_label(spec: Optional[str], ids: List[str]) -> str:
    s = (spec or "all").strip()
    return s if s.lower() == "all" or s in SPLITS else f"{len(ids)} ids"
