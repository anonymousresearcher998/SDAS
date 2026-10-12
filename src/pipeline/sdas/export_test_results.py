from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.pipeline.sdas._common import DEFAULT_OUT_ROOT
from src.sdas.export_test import default_export_root, export_experiment
from src.sdas.layout import ExperimentLayout, load_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Export judged test runs of steer cells into the results layout.")
    ap.add_argument("--experiment_dir", default=None, help="One experiment")
    ap.add_argument("--all", action="store_true", help="Every experiment under --out_root")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--dest_root", default=None, help="Destination mitigation dir")
    args = ap.parse_args()
    if bool(args.experiment_dir) == args.all:
        ap.error("pass exactly one of --experiment_dir or --all")
    exps = [Path(args.experiment_dir)] if args.experiment_dir else sorted(p for p in Path(args.out_root).iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "experiment.json").is_file())
    total = 0
    for exp in exps:
        layout = ExperimentLayout.from_dir(str(exp))
        translator = load_experiment(layout)["translator_model_name"]
        dest_root = Path(args.dest_root) if args.dest_root else default_export_root(layout)
        written = export_experiment(layout, dest_root, translator=translator)
        print(f"{exp.name}: {len(written)} test run(s) exported -> {dest_root}")
        total += len(written)
    print(f"done: {total} run(s)")


if __name__ == "__main__":
    main()
