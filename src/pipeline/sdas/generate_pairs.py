from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch

from src.backend import require_local_dir
from src.backends.text_backend import BACKEND_CAUSAL_LM, transformers_major
from src.env import load_env
from src.pipeline.sdas._common import DEFAULT_OUT_ROOT, add_data_arg, assert_data, banner
from src.sdas.data import Item, data_manifest, id_sort_key, load_items, parse_sets
from src.sdas.hooked_model import (
    DTYPES,
    USER_PROMPT_TEMPLATE,
    DecodingConfig,
    HookedCausalLM,
    LayerActs,
    build_user_prompt,
    parse_layers,
)
from src.seeding import item_seed
from src.sdas.layout import (
    ExperimentLayout,
    append_jsonl,
    assert_generation_env,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    create_or_assert_experiment,
    leaf_complete,
    now_ts,
    prompt_condition,
    read_json,
    read_jsonl,
    write_run_record,
)
from src.sdas.prompts import (
    DEFAULT_PAIR,
    INTERVENTION_PROMPTS,
    LANG_CODES,
    assert_lang_in_pair,
    compose_system_prompt,
    experiment_pair,
    intervention_text,
    lang_cfg,
    pair_of_lang,
    resolve_prompts,
)
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB

POOLINGS = ("resp_mean", "resp_mean_k16", "last_prompt")


def experiment_name(translator_model_name: str, tag: str) -> str:
    return f"{translator_model_name}_{tag}" if tag else translator_model_name

def _load_partial(gen_dir):
    acts = torch.load(gen_dir / "acts.pt")
    rows = {r["id"]: r for r in read_jsonl(gen_dir / "gen.jsonl")}
    have: Dict[str, Dict[str, torch.Tensor]] = {}
    for i, item_id in enumerate(acts["ids"]):
        if item_id in rows:
            have[item_id] = {p: acts[p][i] for p in POOLINGS}
            have[item_id]["resp_n_tokens"] = int(acts["resp_n_tokens"][i])
    return rows, have, list(acts["layers"])


def run_leaf(
    model: HookedCausalLM,
    layout: ExperimentLayout,
    *,
    condition: str,
    intervention: Optional[str],
    text: str,
    prompt_name: str,
    lang: str,
    items: Sequence[Item],
    layers: List[int],
    decoding: DecodingConfig,
    base_seed: int,
    overwrite: bool,
) -> None:
    lc = lang_cfg(lang)
    gen_dir = layout.generation_dir(condition, lang, prompt_name)
    expected_ids = [it.id for it in items]

    if overwrite and gen_dir.exists():
        shutil.rmtree(gen_dir)
    if (gen_dir / "meta.json").is_file():
        stored_layers = read_json(gen_dir / "meta.json").get("layers")
        if stored_layers is not None and [int(x) for x in stored_layers] != [int(x) for x in layers]:
            print(
                f"[{condition}] {prompt_name}: stored layers {stored_layers} != requested {list(layers)}; regenerating leaf",
                flush=True,
            )
            shutil.rmtree(gen_dir)
    if leaf_complete(gen_dir, expected_ids):
        print(f"[{condition}] {prompt_name}: complete, skipping", flush=True)
        return

    rows: Dict[str, Dict] = {}
    have: Dict[str, Dict[str, torch.Tensor]] = {}
    if (gen_dir / "meta.json").is_file() and (gen_dir / "acts.pt").is_file():
        rows, have, old_layers = _load_partial(gen_dir)
        if old_layers != layers:
            print(f"[{condition}] {prompt_name}: existing acts.pt has layers {old_layers} != requested {layers}; regenerating leaf", flush=True)
            shutil.rmtree(gen_dir)
            rows, have = {}, {}
        else:
            n_missing = sum(1 for i in expected_ids if i not in have)
            print(
                f"[{condition}] {prompt_name}: extending leaf, {len(have)} ids present, "
                f"{n_missing} of the requested {len(expected_ids)} missing",
                flush=True,
            )
    elif gen_dir.exists():
        print(f"[{condition}] {prompt_name}: no meta.json (interrupted run)", flush=True)
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    template = SYSTEM_PROMPTS_LIB[prompt_name]
    system_prompt = compose_system_prompt(template, text, language=lc.language, country=lc.country)

    if have:
        kept = [rows[i] for i in sorted(have, key=id_sort_key)]
        atomic_write_text(gen_dir / "gen.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        rows = {r["id"]: r for r in kept}

    todo = [it for it in items if it.id not in have]
    t_leaf = time.time()
    for k, it in enumerate(todo):
        prompt = model.build_prompt(system_prompt, it.en, lc.language)
        seed = item_seed(base_seed, prompt_name, it.idx)
        t0 = time.time()
        res = model.generate(prompt, decoding, seed, capture_layers=layers)
        row = {
            "id": it.id,
            "idx": it.idx,
            "en": it.en,
            **it.ref_fields(),
            "lang": lang,
            "prompt_name": prompt_name,
            "condition": condition,
            "seed": seed,
            "mt": res.text,
            "token_ids": res.token_ids,
            "n_tokens": res.n_tokens,
            "finish": res.finish,
        }
        append_jsonl(gen_dir / "gen.jsonl", row)
        rows[it.id] = row
        have[it.id] = {p: torch.stack([getattr(res.acts[l], p) for l in layers]) for p in POOLINGS}
        have[it.id]["resp_n_tokens"] = res.acts[layers[0]].resp_n_tokens if layers else max(0, res.n_tokens - 1)
        print(
            f"[{condition}] {prompt_name} {k + 1}/{len(todo)} id={it.id} n_tokens={res.n_tokens} "
            f"finish={res.finish} {time.time() - t0:.1f}s | {res.text[:80]!r}",
            flush=True,
        )

    ids = sorted(have, key=id_sort_key)
    atomic_write_text(gen_dir / "gen.jsonl", "".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in ids))
    payload = {
        "ids": ids,
        "layers": list(layers),
        "resp_n_tokens": torch.tensor([have[i]["resp_n_tokens"] for i in ids], dtype=torch.int32),
    }
    for p in POOLINGS:
        payload[p] = torch.stack([have[i][p] for i in ids]).float() if ids else torch.zeros(0, len(layers), model.hidden_size)
    atomic_torch_save(gen_dir / "acts.pt", payload)

    n_nan = int(torch.isnan(payload["resp_mean"][:, 0, 0]).sum()) if ids and layers else 0
    atomic_write_json(
        gen_dir / "meta.json",
        {
            "ids": ids,
            "n": len(ids),
            "layers": list(layers),
            "condition": condition,
            "intervention": intervention,
            "lang": lang,
            "prompt_name": prompt_name,
            "system_prompt": system_prompt,
            "decoding": decoding.as_dict(),
            "base_seed": base_seed,
            "n_nan_resp_mean": n_nan,
            "created": now_ts(),
            "seconds": round(time.time() - t_leaf, 1),
        },
    )
    print(f"[{condition}] {prompt_name}: wrote {len(ids)} rows ({n_nan} with NaN resp_mean) in {time.time() - t_leaf:.0f}s", flush=True)

def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(
        description="Stage A of the SDAS flow: regular + prompt/<intervention> generations with activation capture."
    )
    add_data_arg(ap)
    ap.add_argument("--local_translator_model", required=True, help="Local HF weights")
    ap.add_argument("--translator_model_name", required=True)
    ap.add_argument("--tag", default="", help="Optional experiment suffix: <translator_model_name>_<tag>")
    ap.add_argument("--dtype", default="pipeline", choices=sorted(DTYPES))
    ap.add_argument("--lang", required=True, choices=list(LANG_CODES), help="the language picks the pair")
    ap.add_argument("--prompts", default="all", help="Comma-separated prompt names and/or groups [all]")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N items of each set")
    ap.add_argument("--sets", default="all", help="Comma-separated sets to generate: train,val,test [all]")
    ap.add_argument("--layers", default="all", help="Comma-separated 0-indexed decoder layers to capture [all]")
    ap.add_argument("--intervention", default="self_distance_v1", choices=sorted(INTERVENTION_PROMPTS))
    ap.add_argument("--regular_only", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--greedy", action="store_true", help="Greedy decoding instead of the pipeline's sampling config (use a --tag)")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--overwrite", action="store_true", help="Regenerate every requested leaf")
    ap.add_argument("--dry_run", action="store_true", help="Print the composed prompts and the leaf plan without loading the model")
    args = ap.parse_args()

    lc = lang_cfg(args.lang)
    pair = pair_of_lang(args.lang)
    prompts = resolve_prompts(args.prompts)
    sets = parse_sets(args.sets)
    items = load_items(args.data, args.limit, sets=sets, langs=pair.langs)
    if not items:
        raise SystemExit(f"no usable items in {args.data} (sets {sets}; items need 'neutral' + {' + '.join(repr(l) for l in pair.langs)})")
    n_by_set = {s: sum(1 for it in items if it.split == s) for s in sets}
    decoding = DecodingConfig.from_translator_defaults(greedy=args.greedy)
    text = intervention_text(args.intervention)
    layout = ExperimentLayout(args.out_root, experiment_name(args.translator_model_name, args.tag))
    conditions = [("regular", None, "")]
    if not args.regular_only:
        conditions.append((prompt_condition(args.intervention), args.intervention, text))

    banner(
        "Stage A - generate pairs",
        [
            ("experiment", str(layout.root)),
            ("model", args.local_translator_model),
            ("dtype", args.dtype),
            ("lang", f"{lc.code} ({lc.language}/{lc.country})"),
            ("prompts", f"{len(prompts)}: {', '.join(prompts)}"),
            ("items", f"{len(items)} = " + " + ".join(f"{s}:{n}" for s, n in n_by_set.items()) + f" (limit={args.limit} per set)"),
            ("intervention", "none (--regular_only)" if args.regular_only else args.intervention),
            ("decoding", decoding.as_dict()),
            ("base seed", args.seed),
            ("layers", args.layers),
        ],
    )

    if args.dry_run:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.local_translator_model, use_fast=True, trust_remote_code=True)
        first = items[0]
        for prompt_name in prompts[:1]:
            template = SYSTEM_PROMPTS_LIB[prompt_name]
            for condition, _, cond_text in conditions:
                system_prompt = compose_system_prompt(template, cond_text, language=lc.language, country=lc.country)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(lc.language, first.en)},
                ]
                rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                n_ids = len(tok(rendered)["input_ids"])
                print(f"\n=== {condition} / {prompt_name} / {first.id} - full templated prompt ({n_ids} tokens incl. specials) ===")
                print(rendered)
        print("\n=== leaf plan ===")
        expected_ids = [it.id for it in items]
        for prompt_name in prompts:
            for condition, _, _ in conditions:
                gen_dir = layout.generation_dir(condition, lc.code, prompt_name)
                status = "complete" if leaf_complete(gen_dir, expected_ids) else ("partial" if gen_dir.exists() else "missing")
                print(f"  {status:9s} {gen_dir}")
        return

    require_local_dir(args.local_translator_model, what="translator", flag="--local_translator_model")
    if layout.experiment_json.is_file():
        existing = read_json(layout.experiment_json)
        assert_generation_env(existing)
        assert_data(existing, args.data)
        assert_lang_in_pair(args.lang, experiment_pair(existing))
    print(f"TRANSLATOR BACKEND: LOCAL (HookedCausalLM) path={args.local_translator_model}", flush=True)
    model = HookedCausalLM(args.local_translator_model, dtype=args.dtype)
    layers = parse_layers(args.layers, model.n_layers)
    print(
        f"model_type={model.model_type} backend={model.backend} transformers={model.transformers_version} "
        f"n_layers={model.n_layers} hidden={model.hidden_size} runtime_dtype={model.runtime_dtype} "
        f"eos_token_id={model.eos_token_id} device={model.device}",
        flush=True,
    )

    invariants = {
        "translator_model_name": args.translator_model_name,
        "model_path": os.path.abspath(args.local_translator_model),
        "model_type": model.model_type,
        "n_layers": model.n_layers,
        "hidden_size": model.hidden_size,
        "dtype": args.dtype,
        "runtime_dtype": model.runtime_dtype,
        "decoding": decoding.as_dict(),
        "base_seed": args.seed,
        "data": data_manifest(args.data),
        "tokenization": "pipeline_string_default_special_tokens",
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "chat_template_sha256": model.chat_template_sha256,
    }
    if model.backend != BACKEND_CAUSAL_LM or transformers_major() >= 5:
        invariants.update(
            {
                "backend": model.backend,
                "transformers_major": transformers_major(),
                "eos_token_ids": model.eos_token_id,
            }
        )
    if pair.name != DEFAULT_PAIR:
        invariants["pair"] = pair.name
    create_or_assert_experiment(layout, invariants)
    write_run_record(
        layout,
        "pairs",
        {"args": vars(args), "layers": layers, "sets": sets, "n_items": len(items), "prompts": prompts, "data_pattern": os.path.abspath(args.data)},
    )

    t_all = time.time()
    for prompt_name in prompts:
        for condition, intervention, cond_text in conditions:
            run_leaf(
                model,
                layout,
                condition=condition,
                intervention=intervention,
                text=cond_text,
                prompt_name=prompt_name,
                lang=lc.code,
                items=items,
                layers=layers,
                decoding=decoding,
                base_seed=args.seed,
                overwrite=args.overwrite,
            )
    print(f"Stage A done in {(time.time() - t_all) / 60:.1f} min -> {layout.generations_dir}", flush=True)


if __name__ == "__main__":
    main()
