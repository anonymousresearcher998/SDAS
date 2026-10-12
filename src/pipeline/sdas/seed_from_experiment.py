from __future__ import annotations

import argparse
import errno
import filecmp
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.pipeline.sdas._common import assert_data, banner
from src.sdas.data import load_items
from src.sdas.layout import ExperimentLayout, leaf_complete, load_experiment, read_json, write_run_record
from src.sdas.prompts import experiment_pair
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB

REGULAR = "regular"


def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp{os.getpid()}")


def copy_into_place(src: Path, dst: Path) -> None:
    tmp = _tmp(dst)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def link_into_place(src: Path, dst: Path) -> str:
    tmp = _tmp(dst)
    try:
        os.link(src, tmp)
        how = "linked"
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.copy2(src, tmp)
        how = "copied (other filesystem)"
    os.replace(tmp, dst)
    return how


def source_leaf_problem(leaf: Path, expected_ids: List[str], n_layers: int) -> Optional[str]:
    if not leaf_complete(leaf, expected_ids):
        return "incomplete (no meta.json, or not every train/val/test id)"
    for name in ("acts.pt", "gen.jsonl"):
        if not (leaf / name).is_file():
            return f"no {name}"
    layers = read_json(leaf / "meta.json").get("layers")
    if layers is None or [int(x) for x in layers] != list(range(n_layers)):
        return f"layers {layers} are not all {n_layers} layers"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed a new steering experiment with the regular condition of an existing one")
    ap.add_argument("--src_experiment_dir", required=True, help="The main experiment")
    ap.add_argument("--dst_experiment_dir", required=True, help="The new experiment")
    ap.add_argument("--data", default=None, help="Data pattern with {split} [the pair's: data/ru_ua/neutral_data_{split}.json]")
    ap.add_argument("--dry_run", action="store_true", help="Check everything and print what would be written")
    args = ap.parse_args()

    src, dst = ExperimentLayout.from_dir(args.src_experiment_dir), ExperimentLayout.from_dir(args.dst_experiment_dir)
    if src.root.resolve() == dst.root.resolve():
        raise SystemExit("--src_experiment_dir and --dst_experiment_dir are the same experiment")
    if src.experiment != dst.experiment:
        raise SystemExit(f"experiment names differ ({src.experiment!r} vs {dst.experiment!r})")
    exp = load_experiment(src)
    pair = experiment_pair(exp)
    data = args.data or pair.data_pattern
    assert_data(exp, data)
    expected_ids = [it.id for it in load_items(data, langs=pair.langs)]
    n_layers = int(exp["n_layers"])

    banner("Seed the regular condition of an existing steering experiment", [
        ("source", str(src.root)), ("destination", str(dst.root)), ("pair", f"{pair.name} ({', '.join(pair.langs)})"),
        ("data", data), ("items", f"{len(expected_ids)} (train/val/test)"), ("layers", n_layers),
        ("mode", "DRY RUN - nothing is written" if args.dry_run else "write"),
    ])

    problems: List[str] = []
    if dst.experiment_json.is_file() and dst.experiment_json.read_bytes() != src.experiment_json.read_bytes():
        problems.append(f"{dst.experiment_json} exists and differs from {src.experiment_json}")
    leaves: List[Tuple[str, str, Path, Path]] = []
    n_done = 0
    for lang in pair.langs:
        for prompt in SYSTEM_PROMPTS_LIB:
            s_leaf, d_leaf = src.generation_dir(REGULAR, lang, prompt), dst.generation_dir(REGULAR, lang, prompt)
            why = source_leaf_problem(s_leaf, expected_ids, n_layers)
            if why:
                problems.append(f"source {s_leaf}: {why}")
                continue
            d_meta = d_leaf / "meta.json"
            if d_meta.is_file():
                if filecmp.cmp(s_leaf / "meta.json", d_meta, shallow=False) and all((d_leaf / n).is_file() for n in ("acts.pt", "gen.jsonl")):
                    n_done += 1
                    continue
                problems.append(f"destination {d_leaf} already holds a different regular leaf - refusing to overwrite it")
                continue
            leaves.append((lang, prompt, s_leaf, d_leaf))
    if problems:
        raise SystemExit("nothing was written:\n  " + "\n  ".join(problems))

    s_judged, d_judged = src.judged_dir / REGULAR, dst.judged_dir / REGULAR
    judged_files: List[Tuple[Path, Path]] = []
    if s_judged.is_dir():
        for f in sorted(p for p in s_judged.rglob("*") if p.is_file()):
            target = d_judged / f.relative_to(s_judged)
            if not target.is_file():
                judged_files.append((f, target))

    print(f"regular leaves: {len(leaves)} to seed, {n_done} already seeded "
          f"({len(pair.langs)} langs × {len(SYSTEM_PROMPTS_LIB)} prompts)")
    print(f"judged/regular: {len(judged_files)} file(s) to copy" + ("" if s_judged.is_dir() else " (the source has none)"))
    if args.dry_run:
        for lang, prompt, s_leaf, d_leaf in leaves:
            print(f"  would seed {d_leaf}  <-  {s_leaf}")
        print("DRY RUN - nothing written.")
        return

    dst.root.mkdir(parents=True, exist_ok=True)
    try:
        with open(dst.experiment_json, "xb") as f:
            f.write(src.experiment_json.read_bytes())
        print(f"created {dst.experiment_json}")
    except FileExistsError:
        if dst.experiment_json.read_bytes() != src.experiment_json.read_bytes():
            raise SystemExit(f"{dst.experiment_json} appeared meanwhile and differs from the source")
    hows: Dict[str, int] = {}
    for lang, prompt, s_leaf, d_leaf in leaves:
        d_leaf.mkdir(parents=True, exist_ok=True)
        how = link_into_place(s_leaf / "acts.pt", d_leaf / "acts.pt")
        copy_into_place(s_leaf / "gen.jsonl", d_leaf / "gen.jsonl")
        copy_into_place(s_leaf / "meta.json", d_leaf / "meta.json")
        hows[how] = hows.get(how, 0) + 1
        print(f"  seeded {lang}/{prompt}: acts.pt {how}, gen.jsonl + meta.json copied", flush=True)
    for f, target in judged_files:
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_into_place(f, target)
    record = write_run_record(dst, "seed", {
        "src_experiment_dir": str(src.root), "data": data,
        "leaves_seeded": len(leaves), "leaves_already_seeded": n_done, "acts": hows,
        "judged_regular_files_copied": len(judged_files),
    })
    print(f"done: {len(leaves)} leaves seeded ({hows or 'none'}), {n_done} already there, "
          f"{len(judged_files)} judged/regular file(s) copied; record {record}")


if __name__ == "__main__":
    main()
