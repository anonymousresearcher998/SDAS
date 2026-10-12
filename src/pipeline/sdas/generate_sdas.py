from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch

from src.env import load_env
from src.pipeline.sdas._common import (
    add_data_arg,
    add_experiment_dir_arg,
    assert_data,
    banner,
    parse_floats,
    parse_ints,
    resolve_ids,
)
from src.sdas.data import id_sort_key, load_items
from src.sdas.directions import direction_vector, load_direction, load_released_vector
from src.sdas.hooked_model import DecodingConfig, HookedCausalLM, SteerSpec, build_user_prompt
from src.seeding import item_seed
from src.sdas.layout import (
    ExperimentLayout,
    append_jsonl,
    assert_generation_env,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    leaf_complete,
    load_experiment,
    now_ts,
    read_json,
    read_jsonl,
    replicate_base_seed,
    sdas_condition,
    write_run_record,
)
from src.sdas.prompts import LANG_CODES, assert_lang_in_pair, compose_system_prompt, experiment_pair, lang_cfg, resolve_prompts
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB

POOLINGS = ("resp_mean", "resp_mean_k16", "last_prompt")


def first_divergence(a: List[int], b: List[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def projection(vec: torch.Tensor, d_layer: torch.Tensor) -> Optional[float]:
    if torch.isnan(vec).any():
        return None
    return float(vec.float() @ (d_layer.float() / d_layer.float().norm().clamp_min(1e-12)))


def foreign_direction(direction_dir: Path, experiment: Dict) -> Tuple[torch.Tensor, Dict, Dict[str, object]]:
    direction_dir = Path(direction_dir)
    d, meta = load_direction(direction_dir)
    layers = [int(l) for l in meta["layers"]]
    problems: List[str] = []
    expected = (len(layers), int(experiment["hidden_size"]))
    if tuple(d.shape) != expected:
        problems.append(f"direction tensor {tuple(d.shape)} != {expected} (stored layers x this experiment's hidden_size)")
    src_json = direction_dir.parent.parent / "experiment.json"
    src_experiment = read_json(src_json) if src_json.is_file() else None
    if src_experiment is not None:
        for key in ("model_path", "model_type", "n_layers", "hidden_size"):
            if src_experiment.get(key) != experiment.get(key):
                problems.append(f"{key}: source experiment {src_experiment.get(key)!r} != this experiment {experiment.get(key)!r}")
    if problems:
        raise SystemExit(f"--direction_dir {direction_dir}: not a direction of this experiment's model - " + "; ".join(problems))
    provenance: Dict[str, object] = {
        "dir": str(direction_dir.resolve()),
        "experiment": str(src_json.parent.resolve()) if src_experiment is not None else None,
        "pair": experiment_pair(src_experiment).name if src_experiment is not None else None,
        "name": meta.get("name"),
        "intervention": meta.get("intervention"),
        "pooling": meta.get("pooling"),
        "n_pairs": meta.get("n_pairs"),
        "created": meta.get("created"),
        "sha256_direction_pt": hashlib.sha256((direction_dir / "direction.pt").read_bytes()).hexdigest(),
    }
    return d, meta, provenance


def released_direction(path: Path, experiment: Dict) -> Tuple[torch.Tensor, Dict, Dict[str, object]]:
    """A released SDAS vector (SDAS_models/<model>/layer_LL.pt) as a one-layer direction of this experiment's model."""
    layer, v = load_released_vector(path)
    if v.numel() != int(experiment["hidden_size"]):
        raise SystemExit(f"--direction_file {path}: length {v.numel()} != this experiment's hidden_size {experiment['hidden_size']} - another model's vector?")
    if not 0 <= layer < int(experiment["n_layers"]):
        raise SystemExit(f"--direction_file {path}: layer {layer} outside the model's {experiment['n_layers']} layers")
    provenance: Dict[str, object] = {
        "file": str(Path(path).resolve()),
        "layer": layer,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return v.unsqueeze(0), {"layers": [layer], "pooling": "resp_mean"}, provenance


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Stage C of the SDAS flow: steered generation under the regular prompt.")
    add_experiment_dir_arg(ap)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--direction_dir", default=None,
                    help="Cross-conflict transfer: load the direction from this directory (another experiment of the same model, "
                         "instead of this experiment's directions/<--direction>; --direction is then only the slug name.")
    ap.add_argument("--direction_file", default=None,
                    help="A released SDAS vector, SDAS_models/<model>/layer_LL.pt: steer layer LL with it (--layers must be LL); "
                         "--direction is then only the slug name.")
    ap.add_argument("--layers", required=True, help="Comma-separated layers to steer, e.g. 12,16,20")
    ap.add_argument("--alphas", required=True, help="Comma-separated alpha values; negatives need the = form: --alphas=1,2,4")
    ap.add_argument("--positions", default="response", choices=["response", "all"])
    ap.add_argument("--lang", required=True, choices=list(LANG_CODES), help="ru | ua (RU/UA experiments) or ps | il (IL/PS)")
    ap.add_argument("--prompts", default="all")
    ap.add_argument("--ids", default="val", help="val | test | train | all | comma-separated ids | file [val]")
    ap.add_argument("--seed", type=int, default=None, help="One fixed seed for every item")
    ap.add_argument("--rep", type=int, default=None, help="Replicate number")
    add_data_arg(ap)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N items of each set, as Stage A --limit (smoke tests)")
    ap.add_argument("--local_translator_model", default=None, help="Defaults to the experiment's model_path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Print the composed prompt and the cell/leaf plan without loading the model")
    args = ap.parse_args()
    if args.seed is not None and args.rep is not None:
        ap.error("--seed and --rep are mutually exclusive")
    if args.direction_dir and args.direction_file:
        ap.error("--direction_dir and --direction_file are mutually exclusive")
    if args.rep is not None and args.rep < 1:
        ap.error("--rep must be more than 1")

    layout = ExperimentLayout.from_dir(args.experiment_dir)
    experiment = load_experiment(layout)
    if not args.dry_run:
        assert_generation_env(experiment)
    rep_base = replicate_base_seed(int(experiment["base_seed"]), args.rep) if args.rep is not None else None

    def cell_slug(l: int, a: float) -> str:
        return sdas_condition(args.direction, l, a, args.seed, args.rep)

    def cell_seed(prompt_name: str, idx: int, regular_row: Dict) -> int:
        if args.seed is not None:
            return args.seed
        if rep_base is not None:
            return item_seed(rep_base, prompt_name, idx)
        return int(regular_row["seed"])
    direction_source: Optional[Dict[str, object]] = None
    if args.direction_dir:
        d, meta, direction_source = foreign_direction(Path(args.direction_dir), experiment)
    elif args.direction_file:
        d, meta, direction_source = released_direction(Path(args.direction_file), experiment)
    else:
        d, meta = load_direction(layout.direction_dir(args.direction))
    dir_layers = [int(l) for l in meta["layers"]]
    layers = parse_ints(args.layers)
    alphas = parse_floats(args.alphas)
    for l in layers:
        if l not in dir_layers:
            raise SystemExit(f"layer {l} is not in the direction's layers {dir_layers}")
    lc = lang_cfg(args.lang)
    pair = experiment_pair(experiment)
    assert_lang_in_pair(args.lang, pair)
    prompts = resolve_prompts(args.prompts)
    assert_data(experiment, args.data)
    items = load_items(args.data, args.limit, langs=pair.langs)
    ids = resolve_ids(args.ids, [it.id for it in items])
    items_by_id = {it.id: it for it in items}
    decoding = DecodingConfig(**experiment["decoding"])
    n_layers = int(experiment["n_layers"])
    model_path = args.local_translator_model or experiment["model_path"]
    cells = [(l, a) for l in layers for a in alphas]

    banner(
        "Stage C - steered generation",
        [
            ("experiment", str(layout.root)),
            ("direction", f"{args.direction} (pooling={meta.get('pooling')}, layers={dir_layers[0]}..{dir_layers[-1]})"
                          + (f"  <- {direction_source['dir']} ({direction_source['pair'] or 'unknown'} experiment, "
                             f"sha256 {str(direction_source['sha256_direction_pt'])[:12]})" if direction_source and "dir" in direction_source else "")
                          + (f"  <- released vector {direction_source['file']} (sha256 {str(direction_source['sha256'])[:12]})"
                             if direction_source and "file" in direction_source else "")),
            ("cells", ", ".join(cell_slug(l, a) for l, a in cells)),
            ("positions", args.positions),
            ("lang", f"{lc.code} ({lc.language}/{lc.country})"),
            ("prompts", f"{len(prompts)}: {', '.join(prompts)}"),
            ("ids", f"{len(ids)} ({args.ids})"),
            (
                "seed",
                f"fixed override {args.seed}" if args.seed is not None
                else f"replicate {args.rep}: per-item seeds from base {rep_base} (experiment base {experiment['base_seed']})" if args.rep is not None
                else "per-item seeds of the regular rows",
            ),
            ("decoding", decoding.as_dict()),
            ("model", model_path),
            ("dtype", experiment["dtype"]),
        ],
    )

    if args.dry_run:
        first = items_by_id[ids[0]]
        template = SYSTEM_PROMPTS_LIB[prompts[0]]
        print(f"\n=== regular system prompt / {prompts[0]} ===\n{compose_system_prompt(template, '', language=lc.language, country=lc.country)}")
        print(f"=== user message / {first.id} ===\n{build_user_prompt(lc.language, first.en)}")
        if rep_base is not None:
            reg_first = read_jsonl(layout.generation_dir("regular", lc.code, prompts[0]) / "gen.jsonl")
            reg_seed = next((int(r["seed"]) for r in reg_first if r.get("id") == first.id), None)
            print(f"=== seeds / {prompts[0]} / {first.id} ===\nregular row: {reg_seed}   replicate {args.rep}: {cell_seed(prompts[0], first.idx, {})}")
        print("=== leaf plan ===")
        for l, a in cells:
            condition = cell_slug(l, a)
            for prompt_name in prompts:
                gen_dir = layout.generation_dir(condition, lc.code, prompt_name)
                status = "complete" if leaf_complete(gen_dir, ids) else ("partial" if gen_dir.exists() else "missing")
                print(f"  {status:9s} {gen_dir}")
        return

    write_run_record(layout, "sdas", {"args": vars(args), "cells": [cell_slug(l, a) for l, a in cells], "rep_base_seed": rep_base, "n_ids": len(ids),
                                       **({"direction_source": direction_source} if direction_source else {})})
    model = HookedCausalLM(model_path, dtype=experiment["dtype"])
    if model.n_layers != n_layers:
        raise SystemExit(f"model has {model.n_layers} layers, experiment.json says {n_layers}")

    t_all = time.time()
    for l, a in cells:
        condition = cell_slug(l, a)
        vector = direction_vector(d, dir_layers, l).to(model.device)
        post_layers = sorted({min(l + 4, n_layers - 1), n_layers - 1} - {l})
        capture_all = sorted({l, *post_layers})
        proj_dirs = {pl: direction_vector(d, dir_layers, pl) for pl in capture_all if pl in dir_layers}

        for prompt_name in prompts:
            gen_dir = layout.generation_dir(condition, lc.code, prompt_name)
            regular_rows = {r["id"]: r for r in read_jsonl(layout.generation_dir("regular", lc.code, prompt_name) / "gen.jsonl")}
            leaf_ids = [i for i in ids if i in regular_rows]
            missing = [i for i in ids if i not in regular_rows]
            if missing:
                if args.ids.strip().lower() == "all":
                    print(
                        f"[{condition}] {prompt_name}: regular leaf holds {len(leaf_ids)} of the data's "
                        f"{len(ids)} ids (Stage A --limit?); steering those {len(leaf_ids)}",
                        flush=True,
                    )
                else:
                    raise SystemExit(
                        f"regular leaf {lc.code}/{prompt_name} lacks {len(missing)} of the requested ids "
                        f"(e.g. {missing[:3]}) - run Stage A for them first"
                    )
            if not leaf_ids:
                print(f"[{condition}] {prompt_name}: no requested ids in the regular leaf; skipping", flush=True)
                continue
            if args.overwrite and gen_dir.exists():
                shutil.rmtree(gen_dir)
            if leaf_complete(gen_dir, leaf_ids):
                print(f"[{condition}] {prompt_name}: complete, skipping", flush=True)
                continue
            have_rows: Dict[str, Dict] = {}
            have_acts: Dict[str, Dict[str, torch.Tensor]] = {}
            if (gen_dir / "meta.json").is_file() and (gen_dir / "acts.pt").is_file():
                old = torch.load(gen_dir / "acts.pt")
                old_rows = {r["id"]: r for r in read_jsonl(gen_dir / "gen.jsonl")}
                if [int(x) for x in old["layers"]] != capture_all:
                    print(f"[{condition}] {prompt_name}: stored layers {list(old['layers'])} != {capture_all}; regenerating leaf", flush=True)
                    shutil.rmtree(gen_dir)
                else:
                    for i, item_id in enumerate(old["ids"]):
                        if item_id in old_rows:
                            have_rows[item_id] = old_rows[item_id]
                            have_acts[item_id] = {p: old[p][i] for p in POOLINGS}
                            have_acts[item_id]["resp_n_tokens"] = int(old["resp_n_tokens"][i])
                    n_missing = sum(1 for i in leaf_ids if i not in have_rows)
                    print(f"[{condition}] {prompt_name}: extending leaf, {len(have_rows)} ids present, {n_missing} of the requested {len(leaf_ids)} missing", flush=True)
            elif gen_dir.exists():
                print(f"[{condition}] {prompt_name}: no meta.json (interrupted run) - regenerating the whole leaf", flush=True)
                shutil.rmtree(gen_dir)
            gen_dir.mkdir(parents=True, exist_ok=True)
            if have_rows:
                kept = [have_rows[i] for i in sorted(have_rows, key=id_sort_key)]
                atomic_write_text(gen_dir / "gen.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))

            template = SYSTEM_PROMPTS_LIB[prompt_name]
            system_prompt = compose_system_prompt(template, "", language=lc.language, country=lc.country)
            todo = [i for i in leaf_ids if i not in have_rows]
            t_leaf = time.time()
            for k, item_id in enumerate(todo):
                reg = regular_rows[item_id]
                it = items_by_id[item_id]
                seed = cell_seed(prompt_name, it.idx, reg)
                prompt = model.build_prompt(system_prompt, it.en, lc.language)
                t0 = time.time()
                res = model.generate(
                    prompt,
                    decoding,
                    seed,
                    capture_pre_layers=[l],
                    steer=SteerSpec(layer=l, vector=vector, alpha=a, positions=args.positions),
                    capture_layers=post_layers,
                )
                proj: Dict[str, Optional[float]] = {}
                if l in proj_dirs:
                    proj[f"pre_L{l:02d}"] = projection(res.acts_pre[l].resp_mean, proj_dirs[l])
                for pl in post_layers:
                    if pl in proj_dirs:
                        proj[f"post_L{pl:02d}"] = projection(res.acts[pl].resp_mean, proj_dirs[pl])
                exact = res.token_ids == list(reg.get("token_ids", []))
                div = first_divergence(res.token_ids, list(reg.get("token_ids", [])))
                row = {
                    "id": it.id,
                    "idx": it.idx,
                    "en": it.en,
                    **it.ref_fields(),
                    "lang": lc.code,
                    "prompt_name": prompt_name,
                    "condition": condition,
                    "direction": args.direction,
                    "layer": l,
                    "alpha": a,
                    "positions": args.positions,
                    "seed": seed,
                    "rep": args.rep,
                    "mt": res.text,
                    "token_ids": res.token_ids,
                    "n_tokens": res.n_tokens,
                    "finish": res.finish,
                    "proj": proj,
                    "exact_match_regular": exact,
                    "first_divergent_token": div,
                }
                append_jsonl(gen_dir / "gen.jsonl", row)
                have_rows[item_id] = row
                per_layer = {l: res.acts_pre[l], **{pl: res.acts[pl] for pl in post_layers}}
                have_acts[item_id] = {p: torch.stack([getattr(per_layer[cl], p) for cl in capture_all]) for p in POOLINGS}
                have_acts[item_id]["resp_n_tokens"] = per_layer[l].resp_n_tokens
                print(
                    f"[{condition}] {prompt_name} {k + 1}/{len(todo)} id={item_id} n_tokens={res.n_tokens} finish={res.finish} "
                    f"exact={exact} div@{div} proj={ {k2: (round(v, 3) if v is not None else None) for k2, v in proj.items()} } "
                    f"{time.time() - t0:.1f}s | {res.text[:70]!r}",
                    flush=True,
                )

            all_ids = sorted(have_rows, key=id_sort_key)
            atomic_write_text(gen_dir / "gen.jsonl", "".join(json.dumps(have_rows[i], ensure_ascii=False) + "\n" for i in all_ids))
            atomic_torch_save(
                gen_dir / "acts.pt",
                {
                    "ids": all_ids,
                    "layers": capture_all,
                    "capture": {f"L{cl:02d}": ("pre_hook" if cl == l else "post_hook") for cl in capture_all},
                    "resp_n_tokens": torch.tensor([have_acts[i]["resp_n_tokens"] for i in all_ids], dtype=torch.int32),
                    **{p: torch.stack([have_acts[i][p] for i in all_ids]).float() for p in POOLINGS},
                },
            )
            exact_flags = [bool(have_rows[i].get("exact_match_regular")) for i in all_ids]
            divergences = [have_rows[i]["first_divergent_token"] for i in all_ids if not have_rows[i].get("exact_match_regular") and have_rows[i].get("first_divergent_token") is not None]
            exact_rate = sum(exact_flags) / max(1, len(exact_flags))
            atomic_write_json(
                gen_dir / "meta.json",
                {
                    "ids": all_ids,
                    "n": len(all_ids),
                    "generated_now": len(todo),
                    "condition": condition,
                    "direction": args.direction,
                    "layer": l,
                    "alpha": a,
                    "positions": args.positions,
                    "seed_override": args.seed,
                    "rep": args.rep,
                    "rep_base_seed": rep_base,
                    **({"direction_source": direction_source} if direction_source else {}),
                    "lang": lc.code,
                    "prompt_name": prompt_name,
                    "system_prompt": system_prompt,
                    "decoding": decoding.as_dict(),
                    "exact_match_rate_vs_regular": exact_rate,
                    "mean_first_divergent_token": (sum(divergences) / len(divergences)) if divergences else None,
                    "captured_layers": capture_all,
                    "created": now_ts(),
                    "seconds": round(time.time() - t_leaf, 1),
                },
            )
            print(f"[{condition}] {prompt_name}: {len(all_ids)} rows ({len(todo)} new), exact-match vs regular {exact_rate:.0%}, {time.time() - t_leaf:.0f}s", flush=True)
    print(f"Stage C done in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
