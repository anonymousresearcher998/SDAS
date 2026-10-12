from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

EosIds = Union[int, List[int]]

BACKEND_CAUSAL_LM = "causal_lm"
BACKEND_GEMMA4 = "gemma4"

_BACKEND_BY_MODEL_TYPE: Dict[str, str] = {"gemma4": BACKEND_GEMMA4}
_MIN_TRANSFORMERS_MAJOR: Dict[str, int] = {BACKEND_GEMMA4: 5}
_GEMMA4_LAYER_PATHS = (
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "model.layers",
)


def transformers_version() -> str:
    return transformers.__version__


def transformers_major() -> int:
    return int(transformers.__version__.split(".")[0])


def read_model_type(model_path: str) -> Optional[str]:
    path = os.path.join(model_path, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("model_type")


def detect_backend(model_path: str) -> str:
    return _BACKEND_BY_MODEL_TYPE.get(read_model_type(model_path) or "", BACKEND_CAUSAL_LM)


def check_backend_supported(backend: str, model_path: str) -> None:
    """Fail before any weight is touched when the installed transformers cannot load this architecture."""
    need = _MIN_TRANSFORMERS_MAJOR.get(backend)
    if need is not None and transformers_major() < need:
        raise RuntimeError(
            f"{model_path}: model_type {read_model_type(model_path)!r} needs transformers >= {need}.x"
        )


def load_dtype_kwargs(dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    if dtype is None and transformers_major() >= 5:
        dtype = torch.float32
    key = "torch_dtype" if transformers_major() < 5 else "dtype"
    return {key: dtype}


def pipeline_text_kwargs() -> Dict[str, Any]:
    return {"skip_special_tokens": True} if transformers_major() >= 5 else {}


def prepare_pipeline_tokenizer(tokenizer: Any) -> bool:
    if transformers_major() >= 5 and getattr(tokenizer, "response_template", None):
        tokenizer.response_template = None
        return True
    return False


def strip_special_token_strings(text: str, tokenizer: Any) -> str:
    if transformers_major() < 5:
        return text
    for tok in getattr(tokenizer, "all_special_tokens", None) or []:
        if tok and tok in text:
            text = text.replace(tok, "")
    return text


def load_tokenizer(model_path: str, *, use_fast: bool = True, trust_remote_code: bool = True):
    return AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=trust_remote_code)


def load_model(
    model_path: str,
    *,
    dtype: Optional[torch.dtype],
    device_map: str,
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = True,
):
    check_backend_supported(detect_backend(model_path), model_path)
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_dtype_kwargs(dtype),
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
    ).eval()


def _resolve_attr_path(obj: Any, path: str) -> Any:
    for name in path.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def decoder_layers(model: Any, backend: str) -> nn.ModuleList:
    if backend == BACKEND_CAUSAL_LM:
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise ValueError(
                f"{type(model).__name__} exposes no model.layers"
            )
        return layers
    for path in _GEMMA4_LAYER_PATHS:
        layers = _resolve_attr_path(model, path)
        if isinstance(layers, (nn.ModuleList, nn.Sequential)):
            return layers
    raise ValueError(
        f"{type(model).__name__}: no decoder layers found under any of {_GEMMA4_LAYER_PATHS}"
    )


def text_config(config: Any, backend: str) -> Any:
    if backend == BACKEND_CAUSAL_LM:
        return config
    getter = getattr(config, "get_text_config", None)
    return getter() if callable(getter) else getattr(config, "text_config", config)


def eos_token_ids(model: Any, tokenizer: Any, backend: str) -> EosIds:
    if backend == BACKEND_CAUSAL_LM:
        return tokenizer.eos_token_id
    gen_cfg = getattr(model, "generation_config", None)
    ids = getattr(gen_cfg, "eos_token_id", None)
    if isinstance(ids, (list, tuple)) and len(ids) > 0:
        return [int(i) for i in ids]
    if isinstance(ids, int):
        return int(ids)
    return tokenizer.eos_token_id


def eos_set(ids: EosIds) -> Set[int]:
    if ids is None:
        return set()
    return {int(ids)} if isinstance(ids, int) else {int(i) for i in ids}


@dataclass
class TextBackend:
    model: Any
    tokenizer: Any
    layers: nn.ModuleList
    hidden_size: int
    n_layers: int
    eos_token_ids: EosIds
    model_type: str
    backend: str
    transformers_version: str

    @property
    def eos_set(self) -> Set[int]:
        return eos_set(self.eos_token_ids)


def describe_backend(model: Any, tokenizer: Any, backend: str) -> TextBackend:
    layers = decoder_layers(model, backend)
    cfg = text_config(model.config, backend)
    return TextBackend(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        hidden_size=int(cfg.hidden_size),
        n_layers=len(layers),
        eos_token_ids=eos_token_ids(model, tokenizer, backend),
        model_type=str(getattr(model.config, "model_type", type(model).__name__)),
        backend=backend,
        transformers_version=transformers_version(),
    )


def load_text_backend(
    model_path: str,
    *,
    dtype: Optional[torch.dtype],
    device_map: str,
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = True,
) -> TextBackend:
    backend = detect_backend(model_path)
    check_backend_supported(backend, model_path)
    tokenizer = load_tokenizer(model_path, trust_remote_code=trust_remote_code)
    model = load_model(
        model_path, dtype=dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage, trust_remote_code=trust_remote_code
    )
    return describe_backend(model, tokenizer, backend)
