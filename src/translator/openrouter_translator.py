from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.openrouter.client import (
    OpenRouterSettings,
    ProvenanceAccumulator,
    build_client,
    chat_completion,
)


@dataclass
class OpenRouterTranslatorConfig:
    language: str
    model: str
    temperature: Optional[float] = 0.4
    max_tokens: int = 1024
    system_prompt: Optional[str] = None
    reasoning_effort: Optional[str] = None
    debug_print_full_response: bool = False


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def reasoning_block(effort: Optional[str]) -> Optional[Dict[str, Any]]:
    if effort is None or effort == "":
        return None
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS} (got {effort!r})")
    if effort == "none":
        return {"enabled": False}
    return {"effort": effort}


class OpenRouterTranslator:
    def __init__(
        self,
        cfg: OpenRouterTranslatorConfig,
        settings: Optional[OpenRouterSettings] = None,
    ):
        self.cfg = cfg
        self.language = cfg.language

        if self.cfg.system_prompt is None:
            raise ValueError("system_prompt must be provided in OpenRouterTranslatorConfig")

        self.settings = settings or OpenRouterSettings.from_env()
        self.client = build_client(self.settings)

        self.provenance = ProvenanceAccumulator(slug=cfg.model)
        self.last_provenance: Optional[Dict[str, Any]] = None

    def _build_messages(self, en_sentence: str) -> List[Dict[str, str]]:
        user_prompt = (
            f"Translate the following text into {self.cfg.language}.\n"
            f"Text: {en_sentence.strip()}\n"
            "Output only the translation.\n"
        )
        return [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def translate_en(self, en_sentence: str, seed: Optional[int] = None) -> str:
        params: Dict[str, Any] = {"max_tokens": self.cfg.max_tokens}
        if self.cfg.temperature is not None:
            params["temperature"] = self.cfg.temperature
        if seed is not None:
            params["seed"] = int(seed) & 0x7FFFFFFF

        text, provenance = chat_completion(
            self.client,
            self.settings,
            model=self.cfg.model,
            messages=self._build_messages(en_sentence),
            debug_print_full_response=self.cfg.debug_print_full_response,
            reasoning=reasoning_block(self.cfg.reasoning_effort),
            **params,
        )

        self.last_provenance = provenance
        self.provenance.add(provenance)

        return text.strip().strip('"').strip()

    @property
    def last_provider(self) -> Optional[str]:
        return (self.last_provenance or {}).get("provider")
