from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.env import load_env
from src.pipeline.sdas._common import (
    add_data_arg,
    add_experiment_dir_arg,
    assert_data,
    banner,
    parse_csv,
    parse_ints,
)
from src.sdas.data import SPLITS, load_items, split_ids
from src.sdas.directions import (
    POOLINGS,
    compute_direction,
    diagnostics,
    load_direction,
    load_pairs,
    orthogonalize,
    random_like,
    save_direction,
)
from src.sdas.layout import (
    ExperimentLayout,
    atomic_write_json,
    load_experiment,
    now_ts,
    write_run_record,
)
from src.sdas.prompts import INTERVENTION_PROMPTS, experiment_pair, resolve_langs, resolve_prompts


def print_table(diag: Dict, compare_label: Optional[str]) -> None:
    per_layer = diag["per_layer"]
    langs_key = next((k for k in next(iter(per_layer.values())) if k.startswith("cos_") and k.count("_") == 2 and not k.startswith("cos_compare")), None)
    head = f"{'layer':>5} {'n':>5} {'‖d‖':>8} {'‖Δ‖':>8} {'ratio':>6} {'LOO+':>5} {'LOOcos':>7} {'split½':>7} {('cos(' + langs_key[len('cos_'):].replace('_', ',') + ')') if langs_key else 'cos(a,b)':>10} {'proj reg->int':>16}"
    if compare_label:
        head += f" {'cos(' + compare_label + ')':>16}"
    print(head)
    for layer in diag["layers"]:
        r = per_layer[str(layer)]
        line = (
            f"{layer:>5} {r['n']:>5} {r['norm_d']:>8.3f} {r['mean_norm_delta']:>8.3f} {r['norm_ratio']:>6.3f} "
            f"{r.get('loo_frac_pos_cos', float('nan')):>5.2f} {r.get('loo_mean_cos', float('nan')):>7.3f} "
            f"{r.get('split_half_cos_mean', float('nan')):>7.3f} {r.get(langs_key, float('nan')) if langs_key else float('nan'):>10.3f} "
            f"{r['proj_regular_mean']:>7.3f}->{r['proj_int_mean']:<8.3f}"
        )
        if compare_label:
            line += f" {r.get('cos_compare_to', float('nan')):>16.3f}"
        print(line)
    sys.stdout.flush()


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Stage B of the SDAS flow: mean-difference direction + diagnostics.")
    add_experiment_dir_arg(ap)
    ap.add_argument("--name", required=True, help="Direction name")
    ap.add_argument("--intervention", default="self_distance_v1", choices=sorted(INTERVENTION_PROMPTS), help="Which prompt/<name> condition to pair with regular")
    ap.add_argument("--pooling", default="resp_mean", choices=list(POOLINGS))
    ap.add_argument("--langs", default=None, help="Comma-separated languages to train on")
    ap.add_argument("--prompts", default="all", help="Comma-separated prompt names and/or groups to train on [all]")
    ap.add_argument("--layers", default="all", help="Comma-separated layers to keep [all captured]")
    ap.add_argument("--train", default="train", choices=[*SPLITS, "all"], help="Which split's items to train on [train]")
    add_data_arg(ap)
    ap.add_argument("--balance_by", default="none", choices=["none", "prompt", "lang"])
    ap.add_argument("--random_like", default=None, help="Make a random direction with this direction's per-layer norms instead of learning one")
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--orthogonalize_to", default=None, help="Remove the component along this direction")
    ap.add_argument("--compare_to", default=None, help="Report cos(d, this direction) per layer")
    ap.add_argument("--split_half_k", type=int, default=10)
    ap.add_argument("--diag_seed", type=int, default=0)
    ap.add_argument("--min_group", type=int, default=5, help="Minimum pairs for a sub-group cosine")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    layout = ExperimentLayout.from_dir(args.experiment_dir)
    experiment = load_experiment(layout)
    assert_data(experiment, args.data)
    pair = experiment_pair(experiment)
    args.langs = ",".join(resolve_langs(args.langs, pair))
    items = load_items(args.data, langs=pair.langs)
    ids_by_split = split_ids(items)

    ddir = layout.direction_dir(args.name)
    if ddir.exists():
        if not args.overwrite:
            raise SystemExit(f"{ddir} exists - pass --overwrite to replace it")
        shutil.rmtree(ddir)
    ddir.mkdir(parents=True, exist_ok=True)
    write_run_record(layout, "direction", {"args": vars(args)})

    banner(
        "Stage B - learn direction",
        [
            ("experiment", str(layout.root)),
            ("direction", args.name),
            ("intervention", args.intervention),
            ("pooling", args.pooling),
            ("train split", f"{args.train} ({len(items) if args.train == 'all' else len(ids_by_split[args.train])} items)"),
            ("langs", args.langs),
            ("prompts", args.prompts),
            ("balance_by", args.balance_by),
        ],
    )

    common_meta = {
        "name": args.name,
        "experiment": layout.experiment,
        "train_ids": ids_by_split["train"],
        "val_ids": ids_by_split["val"],
        "test_ids": ids_by_split["test"],
        "split": {"unit": "data_file", "pattern": experiment["data"]["pattern"]},
        "created": now_ts(),
    }

    if args.random_like:
        d_src, meta_src = load_direction(layout.direction_dir(args.random_like))
        d = random_like(d_src, args.random_seed)
        meta = {
            **common_meta,
            "kind": "random_like",
            "source": args.random_like,
            "random_seed": args.random_seed,
            "layers": meta_src["layers"],
            "pooling": meta_src.get("pooling"),
            "intervention": meta_src.get("intervention"),
            "n_pairs": 0,
            "norms": {str(l): float(d[j].norm()) for j, l in enumerate(meta_src["layers"])},
        }
        save_direction(ddir, d, meta)
        print(f"random direction with the per-layer norms of {args.random_like} -> {ddir}")
        return

    train_ids = None if args.train == "all" else ids_by_split[args.train]
    langs = parse_csv(args.langs)
    prompts = resolve_prompts(args.prompts)
    layers = None if args.layers.strip().lower() == "all" else parse_ints(args.layers)

    pairs = load_pairs(
        layout,
        args.intervention,
        langs=langs,
        prompts=prompts,
        pooling=args.pooling,
        ids=train_ids,
        layers=layers,
    )
    d = compute_direction(pairs, args.balance_by)

    if args.orthogonalize_to:
        other, other_meta = load_direction(layout.direction_dir(args.orthogonalize_to))
        if [int(l) for l in other_meta["layers"]] != pairs.layers:
            raise SystemExit(f"--orthogonalize_to {args.orthogonalize_to}: layers {other_meta['layers']} != {pairs.layers}")
        d = orthogonalize(d, other)

    compare = None
    if args.compare_to:
        compare, cmp_meta = load_direction(layout.direction_dir(args.compare_to))
        if [int(l) for l in cmp_meta["layers"]] != pairs.layers:
            raise SystemExit(f"--compare_to {args.compare_to}: layers {cmp_meta['layers']} != {pairs.layers}")

    diag = diagnostics(pairs, d, split_half_k=args.split_half_k, seed=args.diag_seed, compare_to=compare, min_group=args.min_group)
    meta = {
        **common_meta,
        "kind": "mean_difference",
        "intervention": args.intervention,
        "pooling": args.pooling,
        "layers": pairs.layers,
        "train": args.train,
        "langs": langs,
        "prompts": prompts,
        "balance_by": args.balance_by,
        "orthogonalized_to": args.orthogonalize_to,
        "compared_to": args.compare_to,
        "n_pairs": pairs.n,
        "n_items": len(set(pairs.ids)),
        "n_pairs_by_lang": {l: pairs.langs.count(l) for l in sorted(set(pairs.langs))},
    }
    save_direction(ddir, d, meta)
    atomic_write_json(ddir / "diagnostics.json", diag)
    print(f"\n{pairs.n} pairs ({len(set(pairs.ids))} items) -> {ddir}\n")
    print_table(diag, args.compare_to)


if __name__ == "__main__":
    main()
