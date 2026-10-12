from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch

from src.env import load_env
from src.pipeline.sdas._common import add_experiment_dir_arg, banner, parse_csv, parse_floats, parse_ints
from src.sdas.data import in_split
from src.sdas.directions import direction_vector, load_direction
from src.sdas.hooked_model import HookedCausalLM, SteerSpec
from src.sdas.layout import (
    ExperimentLayout,
    assert_generation_env,
    atomic_write_json,
    load_experiment,
    now_ts,
    prompt_condition,
    read_jsonl,
    write_run_record,
)
from src.sdas.prompts import compose_system_prompt, experiment_pair, lang_cfg, resolve_langs, resolve_prompts
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB


def collect_pairs(layout: ExperimentLayout, intervention: str, langs: List[str], prompts: List[str], split: Optional[str]) -> List[Dict]:
    pairs: List[Dict] = []
    for lang in langs:
        for prompt_name in prompts:
            reg = {r["id"]: r for r in read_jsonl(layout.generation_dir("regular", lang, prompt_name) / "gen.jsonl")}
            inter = {r["id"]: r for r in read_jsonl(layout.generation_dir(prompt_condition(intervention), lang, prompt_name) / "gen.jsonl")}
            for item_id in reg:
                if item_id not in inter or not in_split(item_id, split):
                    continue
                r, s = reg[item_id], inter[item_id]
                if not r.get("token_ids") or not s.get("token_ids"):
                    continue
                pairs.append({"lang": lang, "prompt_name": prompt_name, "id": item_id, "reg": r, "int": s})
    return pairs


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Stage B2 of the SDAS flow: teacher-forced layer pre-screen.")
    add_experiment_dir_arg(ap)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--alphas", default="1,2", help="Comma-separated alpha values (use --alphas=-1,1 for negatives)")
    ap.add_argument("--langs", default=None, help="Comma-separated languages")
    ap.add_argument("--prompts", default="all")
    ap.add_argument("--ids", default="val", choices=["val", "test", "train", "all"], help="Split to sample pairs from [val]")
    ap.add_argument("--layers", default="all", help="Layers to screen [all layers of the direction]")
    ap.add_argument("--positions", default="response", choices=["response", "all"])
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--local_translator_model", default=None, help="Defaults to the experiment's model_path")
    args = ap.parse_args()

    layout = ExperimentLayout.from_dir(args.experiment_dir)
    experiment = load_experiment(layout)
    assert_generation_env(experiment)
    args.langs = ",".join(resolve_langs(args.langs, experiment_pair(experiment)))
    d, meta = load_direction(layout.direction_dir(args.direction))
    intervention = meta.get("intervention")
    if not intervention:
        raise SystemExit(f"direction {args.direction} has no intervention")
    dir_layers = [int(l) for l in meta["layers"]]
    layers = dir_layers if args.layers.strip().lower() == "all" else parse_ints(args.layers)
    for l in layers:
        if l not in dir_layers:
            raise SystemExit(f"layer {l} is not in the direction's layers {dir_layers}")
    alphas = parse_floats(args.alphas)
    langs = parse_csv(args.langs)
    prompts = resolve_prompts(args.prompts)
    candidates = collect_pairs(layout, intervention, langs, prompts, None if args.ids == "all" else args.ids)
    if not candidates:
        raise SystemExit("no usable pairs (both rows present, both with generated tokens)")
    rng = random.Random(args.sample_seed)
    sample = candidates if len(candidates) <= args.n_pairs else rng.sample(candidates, args.n_pairs)
    sample.sort(key=lambda p: (p["lang"], p["prompt_name"], p["id"]))

    model_path = args.local_translator_model or experiment["model_path"]
    banner(
        "Stage B2 - layer screen",
        [
            ("experiment", str(layout.root)),
            ("direction", f"{args.direction} (pooling={meta.get('pooling')}, intervention={intervention})"),
            ("layers", f"{len(layers)}: {layers[0]}..{layers[-1]}"),
            ("alphas", alphas),
            ("pairs", f"{len(sample)} of {len(candidates)} candidates ({args.ids} split)"),
            ("positions", args.positions),
            ("model", model_path),
            ("dtype", experiment["dtype"]),
        ],
    )
    run_record = write_run_record(layout, "screen", {"args": vars(args), "n_candidates": len(candidates), "n_sample": len(sample)})

    model = HookedCausalLM(model_path, dtype=experiment["dtype"])
    if model.n_layers != experiment["n_layers"]:
        raise SystemExit(f"model has {model.n_layers} layers, experiment.json says {experiment['n_layers']}")
    vectors = {l: direction_vector(d, dir_layers, l).to(model.device) for l in layers}

    sums: Dict[Tuple[int, float], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    t0 = time.time()
    for k, pair in enumerate(sample):
        lc = lang_cfg(pair["lang"])
        system_prompt = compose_system_prompt(SYSTEM_PROMPTS_LIB[pair["prompt_name"]], "", language=lc.language, country=lc.country)
        prompt = model.build_prompt(system_prompt, pair["reg"]["en"], lc.language)
        y_reg, y_int = pair["reg"]["token_ids"], pair["int"]["token_ids"]
        base_reg, n_reg = model.teacher_forced_logprob(prompt, y_reg)
        base_int, n_int = model.teacher_forced_logprob(prompt, y_int)
        for l in layers:
            for a in alphas:
                spec = SteerSpec(layer=l, vector=vectors[l], alpha=a, positions=args.positions)
                s_reg, _ = model.teacher_forced_logprob(prompt, y_reg, steer=spec)
                s_int, _ = model.teacher_forced_logprob(prompt, y_int, steer=spec)
                d_reg, d_int = s_reg - base_reg, s_int - base_int
                cell = sums[(l, a)]
                cell["delta_int"] += d_int
                cell["delta_reg"] += d_reg
                cell["pref_total"] += d_int - d_reg
                cell["pref_per_token"] += d_int / n_int - d_reg / n_reg
                cell["damage_per_token"] += -d_reg / n_reg
                cell["base_logp_reg_per_token"] += base_reg / n_reg
                cell["base_logp_int_per_token"] += base_int / n_int
        print(f"pair {k + 1}/{len(sample)} {pair['lang']}/{pair['prompt_name']}/{pair['id']} n_reg={n_reg} n_int={n_int} {time.time() - t0:.0f}s", flush=True)

    n = len(sample)
    cells = {}
    for (l, a), acc in sums.items():
        cells[f"L{l:02d}_a{a:g}"] = {"layer": l, "alpha": a, **{key: value / n for key, value in acc.items()}, "n": n}
    ts = now_ts()
    out_dir = layout.screens_dir(args.direction, ts)
    atomic_write_json(
        out_dir / "screen.json",
        {
            "direction": args.direction,
            "intervention": intervention,
            "layers": layers,
            "alphas": alphas,
            "positions": args.positions,
            "n_pairs": n,
            "pairs": [{"lang": p["lang"], "prompt_name": p["prompt_name"], "id": p["id"]} for p in sample],
            "cells": cells,
            "run_record": run_record.name,
        },
    )

    print(f"\nscreen over {n} pairs -> {out_dir / 'screen.json'}\n")
    head = f"{'layer':>5}" + "".join(f" | alpha={a:g}: pref/tok {'damage/tok':>10}" for a in alphas)
    print(head)
    for l in layers:
        line = f"{l:>5}"
        for a in alphas:
            c = cells[f"L{l:02d}_a{a:g}"]
            line += f" | {c['pref_per_token']:>+13.4f} {c['damage_per_token']:>10.4f}"
        print(line)
    for a in alphas:
        ranked = sorted(layers, key=lambda l: cells[f"L{l:02d}_a{a:g}"]["pref_per_token"], reverse=True)[:5]
        print(f"top-5 layers by preference shift at alpha={a:g}: {ranked}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
