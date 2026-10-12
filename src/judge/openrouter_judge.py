from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.judge.schema import JUDGE_JSON_SCHEMA, JudgeResult
from src.openrouter.client import (
    OpenRouterSettings,
    ProvenanceAccumulator,
    build_client,
    chat_completion,
    is_response_format_rejection,
)


@dataclass
class OpenRouterJudgeConfig:
    model: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.0
    max_tokens: int = 1024
    reasoning_effort: Optional[str] = "none"
    use_json_schema: bool = True
    debug_print_full_response: bool = False


class OpenRouterJudge:
    _TEMPLATE = """{system}

Return ONLY a valid JSON object (no prose, no markdown, no extra keys), in the following format:
{{
  "score": <number in [0,1]>,
  "explanation": <1-2 short, neutral English sentences>
}}

Sentence A: {sentence_a}
Sentence B: {sentence_b}
"""

    _STRICT_TEMPLATE = """{system}

Return ONLY this JSON object (no prose, no markdown, no comments, no extra keys):
{{
  "score": 0.0,
  "explanation": "..."
}}

The value of "score" MUST be a real number in [0,1].
The value of "explanation" MUST be a short, neutral English reason (1-2 sentences).
Start with '{{' and end with '}}'. Nothing else.

Sentence A: {sentence_a}
Sentence B: {sentence_b}
"""

    def __init__(
        self,
        cfg: OpenRouterJudgeConfig,
        settings: Optional[OpenRouterSettings] = None,
    ):
        self.cfg = cfg

        if self.cfg.system_prompt is None:
            raise ValueError("system_prompt must be provided in OpenRouterJudgeConfig")
        if not self.cfg.model:
            raise ValueError("model must be provided in OpenRouterJudgeConfig")
        
        self.settings = settings or OpenRouterSettings.from_env()
        self.client = build_client(self.settings)

        self.provenance = ProvenanceAccumulator(slug=cfg.model)
        self.last_provenance: Optional[Dict[str, Any]] = None
        self._json_schema_supported = cfg.use_json_schema

    def _build_messages(
        self, language: str, sentence_a: str, sentence_b: str, strict: bool
    ) -> List[Dict[str, str]]:
        template = self._STRICT_TEMPLATE if strict else self._TEMPLATE
        system = self.cfg.system_prompt.format(language=language)
        user_prompt = template.format(
            system=system,
            sentence_a=sentence_a.strip(),
            sentence_b=sentence_b.strip(),
        )
        return [{"role": "user", "content": user_prompt}]

    @staticmethod
    def _extract_json_obj(text: str) -> Dict[str, Any]:
        """
        Best-effort extraction of a single top-level JSON object from model output.
        """
        t = (text or "").strip()
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in output.")

        candidate = t[start : end + 1].strip()
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError("Parsed JSON is not an object.")
        return obj

    def _call(self, messages: List[Dict[str, str]]) -> str:
        params: Dict[str, Any] = {"max_tokens": int(self.cfg.max_tokens)}
        if self.cfg.temperature is not None:
            params["temperature"] = float(self.cfg.temperature)
        if self.cfg.reasoning_effort is not None:
            params["reasoning_effort"] = self.cfg.reasoning_effort

        response_format = (
            {"type": "json_schema", "json_schema": JUDGE_JSON_SCHEMA}
            if self._json_schema_supported
            else None
        )

        try:
            text, provenance = chat_completion(
                self.client,
                self.settings,
                model=self.cfg.model,
                messages=messages,
                response_format=response_format,
                debug_print_full_response=self.cfg.debug_print_full_response,
                **params,
            )
        except Exception as exc:
            if response_format is None or not is_response_format_rejection(exc):
                raise
            print(
                f"[judge] endpoint rejected json_schema ({exc});"
            )
            self._json_schema_supported = False
            text, provenance = chat_completion(
                self.client,
                self.settings,
                model=self.cfg.model,
                messages=messages,
                debug_print_full_response=self.cfg.debug_print_full_response,
                **params,
            )

        self.last_provenance = provenance
        self.provenance.add(provenance)
        return text.strip()

    def judge(self, language: str, sentence_a: str, sentence_b: str) -> Dict[str, Any]:
        try:
            out_text = self._call(self._build_messages(language, sentence_a, sentence_b, strict=False))
            obj = self._extract_json_obj(out_text)
            result = JudgeResult(**obj)
            return {"score": float(result.score), "explanation": result.explanation}
        except Exception as exc:
            print(f"[judge] first attempt unusable ({type(exc).__name__}: {exc}); retrying strictly")

        out_text = self._call(self._build_messages(language, sentence_a, sentence_b, strict=True))
        obj = self._extract_json_obj(out_text)
        result = JudgeResult(**obj)
        return {"score": float(result.score), "explanation": result.explanation}

    def judge_json(self, language: str, sentence_a: str, sentence_b: str) -> str:
        out = self.judge(language, sentence_a, sentence_b)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))

    @property
    def last_provider(self) -> Optional[str]:
        return (self.last_provenance or {}).get("provider")
