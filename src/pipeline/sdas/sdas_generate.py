from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.sdas.directions import load_released_vector
from src.sdas.hooked_model import DTYPES, DecodingConfig, HookedCausalLM, SteerSpec
from src.sdas.prompts import LANG_CODES, compose_system_prompt, lang_cfg
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB

RELEASED_ROOT = Path(__file__).resolve().parents[3] / "SDAS_models"

RELEASED_ALPHA = {
    "qwen25-7b": 1.0,
    "aya-expanse-8b": 2.0,
    "falcon3-7b-instruct": 2.0,
    "gemma-4-e4b-it": 2.0,
    "mistral-7b-instruct-v0.3": 2.0,
}


def released_file(model_name: str) -> Path:
    files = sorted((RELEASED_ROOT / model_name).glob("layer_*.pt"))
    if len(files) != 1:
        raise SystemExit(f"expected one {RELEASED_ROOT / model_name}/layer_*.pt (found {len(files)}) - pass --vector")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate with a released SDAS steering vector.")
    ap.add_argument("--model_path", required=True, help="Local Hugging Face weights of the model")
    ap.add_argument("--model_name", default=None, choices=sorted(RELEASED_ALPHA))
    ap.add_argument("--vector", default=None, help="A layer_LL.pt file instead of --model_name's")
    ap.add_argument("--alpha", type=float, default=None, help="Steering strength [the released alpha of --model_name]")
    task = ap.add_argument_group("the paper's task (with --text)")
    task.add_argument("--text", default=None, help="English event statement to render")
    task.add_argument("--lang", default=None, choices=list(LANG_CODES), help="Target language: ru | ua | ps | il")
    task.add_argument("--context", default="neutral_1", choices=list(SYSTEM_PROMPTS_LIB), help="Communication context [neutral_1]")
    free = ap.add_argument_group("any prompt (with --prompt)")
    free.add_argument("--prompt", default=None, help="User message, sent through the chat template")
    free.add_argument("--system", default=None, help="Optional system message for --prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--dtype", default="pipeline", choices=sorted(DTYPES))
    ap.add_argument("--compare", action="store_true", help="Also print the unsteered output (same seed)")
    args = ap.parse_args()

    if (args.text is None) == (args.prompt is None):
        ap.error("pass exactly one of --text (the paper's task) or --prompt (any prompt)")
    if args.text is not None and args.lang is None:
        ap.error("--text needs --lang")
    if args.model_name is None and args.vector is None:
        ap.error("pass --model_name (a released model) or --vector")
    vector_path = Path(args.vector) if args.vector else released_file(args.model_name)
    alpha = args.alpha if args.alpha is not None else RELEASED_ALPHA.get(args.model_name or "")
    if alpha is None:
        ap.error("--alpha is required with a --vector of an unlisted model")

    layer, vector = load_released_vector(vector_path)
    model = HookedCausalLM(args.model_path, dtype=args.dtype)
    if vector.numel() != model.hidden_size:
        raise SystemExit(f"{vector_path}: length {vector.numel()} != the model's hidden size {model.hidden_size} - another model's vector?")
    if not 0 <= layer < model.n_layers:
        raise SystemExit(f"{vector_path}: layer {layer} outside the model's {model.n_layers} layers")

    if args.text is not None:
        lc = lang_cfg(args.lang)
        system = compose_system_prompt(SYSTEM_PROMPTS_LIB[args.context], "", language=lc.language, country=lc.country)
        prompt = model.build_prompt(system, args.text, lc.language)
    else:
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": args.prompt}]
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    decoding = DecodingConfig.from_translator_defaults(greedy=args.greedy)
    if args.max_new_tokens is not None:
        decoding.max_new_tokens = args.max_new_tokens
    steer = SteerSpec(layer=layer, vector=vector.to(model.device), alpha=alpha, positions="response")

    print(f"vector {vector_path} (layer {layer}), alpha {alpha:g}, seed {args.seed}, decoding {decoding.as_dict()}\n")
    if args.compare:
        base = model.generate(prompt, decoding, args.seed)
        print(f"--- unsteered ---\n{base.text}\n")
    out = model.generate(prompt, decoding, args.seed, steer=steer)
    print(f"--- SDAS (layer {layer}, alpha {alpha:g}) ---\n{out.text}")


if __name__ == "__main__":
    main()
