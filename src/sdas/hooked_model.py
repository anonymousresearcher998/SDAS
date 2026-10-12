from __future__ import annotations

import dataclasses
import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.backends.text_backend import (
    check_backend_supported,
    describe_backend,
    detect_backend,
    eos_set,
    load_model,
    load_tokenizer,
)
from src.translator.translator import TranslatorConfig

MIN_RESP_TOKENS = 3
K_PREFIX = 16

USER_PROMPT_TEMPLATE = (
    "Translate the following text into {language}.\n\n"
    "Text: {text}\n"
    "Output only the translation.\n"
)

DTYPES: Dict[str, Optional[torch.dtype]] = {
    "pipeline": None,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def build_user_prompt(language: str, en_sentence: str) -> str:
    return USER_PROMPT_TEMPLATE.format(language=language, text=en_sentence.strip())


def postprocess_text(text: str) -> str:
    return text.strip().strip('“”"').strip()


def resolve_dtype(name: str) -> Optional[torch.dtype]:
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(f"Unknown dtype {name!r}; choose from {', '.join(DTYPES)}") from None


def parse_layers(spec: Optional[str], n_layers: int) -> List[int]:
    if spec is None or not spec.strip() or spec.strip().lower() == "all":
        return list(range(n_layers))
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        layer = int(tok)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} out of range 0..{n_layers - 1}")
        if layer not in out:
            out.append(layer)
    return sorted(out)

@dataclass
class DecodingConfig:
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 50

    @classmethod
    def from_translator_defaults(cls, greedy: bool = False) -> "DecodingConfig":
        defaults = {f.name: f.default for f in dataclasses.fields(TranslatorConfig)}
        cfg = cls(
            max_new_tokens=int(defaults["max_new_tokens"]),
            do_sample=bool(defaults["do_sample"]),
            temperature=float(defaults["temperature"]),
            top_p=float(defaults["top_p"]),
            top_k=int(defaults["top_k"]),
        )
        if greedy:
            cfg.do_sample = False
            cfg.temperature = None
            cfg.top_p = None
            cfg.top_k = None
        return cfg

    def generate_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        for key in ("temperature", "top_p", "top_k"):
            value = getattr(self, key)
            if value is not None:
                kwargs[key] = value
        return kwargs

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _layer_output_tensor(output: Any) -> torch.Tensor:
    """Decoder layers return the hidden-state tensor in 4.55; older versions returned a tuple."""
    return output if torch.is_tensor(output) else output[0]


def _positions(kwargs: Dict[str, Any], hidden: torch.Tensor) -> torch.Tensor:
    pos = kwargs.get("cache_position")
    if pos is None:
        pos_ids = kwargs.get("position_ids")
        if pos_ids is not None and pos_ids.dim() == 2:
            pos = pos_ids[0]
    if pos is None:
        raise RuntimeError(
            "decoder layer received neither cache_position nor position_ids"
        )
    if pos.numel() != hidden.shape[1]:
        raise RuntimeError(f"cache_position has {pos.numel()} entries but the layer output has T={hidden.shape[1]}")
    return pos.to(hidden.device)


@dataclass
class LayerActs:
    last_prompt: torch.Tensor
    resp_mean: torch.Tensor
    resp_mean_k16: torch.Tensor
    resp_n_tokens: int


class CaptureHook:
    def __init__(self, hidden_size: int, device: torch.device):
        self.hidden_size = hidden_size
        self.device = device
        self.reset(0)

    def reset(self, prompt_len: int) -> None:
        self.prompt_len = prompt_len
        self.last_prompt: Optional[torch.Tensor] = None
        self.resp_sum = torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device)
        self.resp_n = 0
        self.k_sum = torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device)
        self.k_n = 0

    def __call__(self, module, args, kwargs, output):
        hidden = _layer_output_tensor(output)
        if hidden.shape[0] != 1:
            raise RuntimeError("CaptureHook supports batch size 1 only")
        pos = _positions(kwargs, hidden)
        h = hidden[0].float()  # [T, d]

        at_last = pos == self.prompt_len - 1
        if bool(at_last.any()):
            self.last_prompt = h[at_last][-1].clone()

        resp = pos >= self.prompt_len
        if bool(resp.any()):
            hr = h[resp]
            self.resp_sum += hr.sum(dim=0)
            self.resp_n += int(hr.shape[0])
            take = min(int(hr.shape[0]), max(0, K_PREFIX - self.k_n))
            if take > 0:
                self.k_sum += hr[:take].sum(dim=0)
                self.k_n += take
        return None

    def result(self) -> LayerActs:
        nan = torch.full((self.hidden_size,), float("nan"), dtype=torch.float32)
        last = self.last_prompt.detach().cpu() if self.last_prompt is not None else nan.clone()
        resp_mean = (self.resp_sum / self.resp_n).detach().cpu() if self.resp_n >= MIN_RESP_TOKENS else nan.clone()
        k16 = (self.k_sum / self.k_n).detach().cpu() if self.k_n >= MIN_RESP_TOKENS else nan.clone()
        return LayerActs(last_prompt=last, resp_mean=resp_mean, resp_mean_k16=k16, resp_n_tokens=self.resp_n)


@dataclass
class SteerSpec:
    layer: int
    vector: torch.Tensor
    alpha: float
    positions: str = "response"

    def __post_init__(self) -> None:
        if self.positions not in ("response", "all"):
            raise ValueError(f"positions must be 'response' or 'all', got {self.positions!r}")
        if self.vector.dim() != 1:
            raise ValueError(f"steering vector must be 1-D, got shape {tuple(self.vector.shape)}")


class SteerHook:
    def __init__(self, spec: SteerSpec):
        self.spec = spec
        self.prompt_len = 0
        self._delta: Optional[torch.Tensor] = None

    def reset(self, prompt_len: int) -> None:
        self.prompt_len = prompt_len

    def _delta_like(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._delta is None or self._delta.device != hidden.device or self._delta.dtype != hidden.dtype:
            self._delta = (float(self.spec.alpha) * self.spec.vector.detach().float()).to(hidden.device, hidden.dtype)
        return self._delta

    def __call__(self, module, args, kwargs, output):
        if self.spec.alpha == 0:
            return None
        hidden = _layer_output_tensor(output)
        pos = _positions(kwargs, hidden)
        if self.spec.positions == "all":
            mask = torch.ones_like(pos, dtype=torch.bool)
        else:
            mask = pos >= self.prompt_len - 1
        if not bool(mask.any()):
            return None
        new = hidden.clone()
        new[:, mask, :] += self._delta_like(hidden)
        if torch.is_tensor(output):
            return new
        return (new,) + tuple(output[1:])


@dataclass
class GenerationResult:
    text: str
    token_ids: List[int]
    n_tokens: int
    finish: str
    prompt_len: int
    seed: int
    acts: Dict[int, LayerActs] = field(default_factory=dict)
    acts_pre: Dict[int, LayerActs] = field(default_factory=dict)


class HookedCausalLM:
    def __init__(self, model_path: str, dtype: str = "pipeline", device: Optional[str] = None):
        self.model_path = model_path
        self.dtype_name = dtype
        self.backend = detect_backend(model_path)
        check_backend_supported(self.backend, model_path)

        self.tokenizer = load_tokenizer(model_path)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise NotImplementedError("Non-chat templates are not implemented.")

        use_cuda = torch.cuda.is_available() and device in (None, "cuda")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = load_model(
            model_path,
            dtype=resolve_dtype(dtype),
            device_map="cuda" if use_cuda else "cpu",
        )

        info = describe_backend(self.model, self.tokenizer, self.backend)
        self.layers = info.layers
        self.n_layers = info.n_layers
        self.hidden_size = info.hidden_size
        self.model_type = info.model_type
        self.transformers_version = info.transformers_version
        self.runtime_dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        chat_template = self.tokenizer.chat_template
        if isinstance(chat_template, dict):
            chat_template = chat_template.get("default") or json.dumps(chat_template, sort_keys=True)
        self.chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        self.eos_token_id = info.eos_token_ids
        self._eos_set = eos_set(self.eos_token_id)
        self.pad_token_id = self.tokenizer.pad_token_id

    def build_prompt(self, system_prompt: str, en_sentence: str, language: str) -> str:
        """Exactly `MinimalTranslator._build_prompt` for a chat model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(language, en_sentence)},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def tokenize(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(prompt, return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def decode_like_pipeline(self, sequence: torch.Tensor, prompt_len: int) -> str:
        seq = sequence.tolist()
        full = self.tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        plen = len(self.tokenizer.decode(seq[:prompt_len], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        return full[plen:]

    def _register(self, stack: ExitStack, layer: int, hook) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} out of range 0..{self.n_layers - 1}")
        handle = self.layers[layer].register_forward_hook(hook, with_kwargs=True)
        stack.callback(handle.remove)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        decoding: DecodingConfig,
        seed: int,
        *,
        capture_layers: Sequence[int] = (),
        steer: Optional[SteerSpec] = None,
        capture_pre_layers: Sequence[int] = (),
    ) -> GenerationResult:
        input_ids, attention_mask = self.tokenize(prompt)
        prompt_len = int(input_ids.shape[1])

        pre = {int(l): CaptureHook(self.hidden_size, self.device) for l in capture_pre_layers}
        post = {int(l): CaptureHook(self.hidden_size, self.device) for l in capture_layers}
        steer_hook = SteerHook(steer) if steer is not None else None
        for hook in list(pre.values()) + list(post.values()) + ([steer_hook] if steer_hook else []):
            hook.reset(prompt_len)

        with ExitStack() as stack:
            for layer, hook in pre.items():
                self._register(stack, layer, hook)
            if steer_hook is not None:
                self._register(stack, steer.layer, steer_hook)
            for layer, hook in post.items():
                self._register(stack, layer, hook)

            torch.manual_seed(int(seed))
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **decoding.generate_kwargs(),
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                use_cache=True,
            )

        seq = out[0]
        new_ids = [int(t) for t in seq[prompt_len:].tolist()]
        n_tokens = len(new_ids)
        if n_tokens and new_ids[-1] in self._eos_set:
            finish = "eos"
        elif n_tokens >= decoding.max_new_tokens:
            finish = "length"
        else:
            finish = "other"
        text = postprocess_text(self.decode_like_pipeline(seq, prompt_len))

        return GenerationResult(
            text=text,
            token_ids=new_ids,
            n_tokens=n_tokens,
            finish=finish,
            prompt_len=prompt_len,
            seed=int(seed),
            acts={layer: hook.result() for layer, hook in post.items()},
            acts_pre={layer: hook.result() for layer, hook in pre.items()},
        )

    @torch.no_grad()
    def teacher_forced_logprob(
        self, prompt: str, continuation_ids: Sequence[int], steer: Optional[SteerSpec] = None
    ) -> Tuple[float, int]:
        if len(continuation_ids) == 0:
            return 0.0, 0
        input_ids, _ = self.tokenize(prompt)
        prompt_len = int(input_ids.shape[1])
        cont = torch.tensor([list(continuation_ids)], dtype=input_ids.dtype, device=self.device)
        ids = torch.cat([input_ids, cont], dim=1)
        attention_mask = torch.ones_like(ids)

        steer_hook = SteerHook(steer) if steer is not None else None
        if steer_hook is not None:
            steer_hook.reset(prompt_len)
        with ExitStack() as stack:
            if steer_hook is not None:
                self._register(stack, steer.layer, steer_hook)
            out = self.model(input_ids=ids, attention_mask=attention_mask, use_cache=False)

        logits = out.logits[0, prompt_len - 1 : ids.shape[1] - 1].float()
        logp = torch.log_softmax(logits, dim=-1).gather(1, cont[0].unsqueeze(1)).squeeze(1)
        return float(logp.sum().item()), int(cont.shape[1])
