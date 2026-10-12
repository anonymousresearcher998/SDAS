from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_APP_TITLE = "sdas"

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}

class OpenRouterCallError(RuntimeError):
    def __init__(self, message: str, *, status: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable

def _scrub(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def _env_str(name: str, default: str = "") -> str:
    return _scrub(os.getenv(name, "") or "") or default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        print(f"[openrouter] ignoring non-numeric {name}={raw!r}, using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass
class OpenRouterSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    http_referer: Optional[str] = None
    app_title: str = DEFAULT_APP_TITLE
    timeout: float = 120.0
    max_retries: int = 5
    sleep_seconds: float = 0.0
    provider_order: Tuple[str, ...] = ()
    allow_fallbacks: bool = True
    backoff_base: float = 2.0
    backoff_max: float = 60.0

    @classmethod
    def from_env(cls) -> "OpenRouterSettings":
        api_key = _env_str("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set."
            )

        order_raw = _env_str("OPENROUTER_PROVIDER_ORDER")
        provider_order = tuple(p.strip() for p in order_raw.split(",") if p.strip())

        return cls(
            api_key=api_key,
            base_url=_env_str("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
            http_referer=_env_str("OPENROUTER_HTTP_REFERER") or None,
            app_title=_env_str("OPENROUTER_APP_TITLE", DEFAULT_APP_TITLE),
            timeout=_env_float("OPENROUTER_TIMEOUT", 120.0),
            max_retries=max(0, _env_int("OPENROUTER_MAX_RETRIES", 5)),
            sleep_seconds=max(0.0, _env_float("OPENROUTER_SLEEP_SECONDS", 0.0)),
            provider_order=provider_order,
            allow_fallbacks=_env_bool("OPENROUTER_ALLOW_FALLBACKS", True),
        )

    def provider_preferences(self) -> Optional[Dict[str, Any]]:
        if not self.provider_order:
            return None if self.allow_fallbacks else {"allow_fallbacks": False}
        return {"order": list(self.provider_order), "allow_fallbacks": self.allow_fallbacks}

    def describe(self) -> str:
        pin = ",".join(self.provider_order) if self.provider_order else "none"
        return (
            f"base_url={self.base_url} timeout={self.timeout:g}s "
            f"max_retries={self.max_retries} sleep={self.sleep_seconds:g}s "
            f"provider_pin={pin} allow_fallbacks={self.allow_fallbacks}"
        )


def build_client(settings: OpenRouterSettings) -> OpenAI:
    headers = {"X-Title": settings.app_title}
    if settings.http_referer:
        headers["HTTP-Referer"] = settings.http_referer

    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        default_headers=headers,
        timeout=settings.timeout,
        max_retries=0,
    )


def _as_dict(response: Any) -> Dict[str, Any]:
    for attempt in (lambda: response.model_dump(), lambda: dict(response)):
        try:
            out = attempt()
            if isinstance(out, dict):
                return out
        except Exception:
            continue
    return {}


def extract_provenance(response: Any) -> Dict[str, Any]:
    payload = _as_dict(response)
    usage = payload.get("usage") or {}
    choices = payload.get("choices") or []
    first = choices[0] if choices else {}

    return {
        "model": payload.get("model"),
        "provider": payload.get("provider"),
        "generation_id": payload.get("id"),
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost": usage.get("cost"),
    }


def _text_from(response: Any) -> str:
    payload = _as_dict(response)

    err = payload.get("error")
    if err:
        message = err.get("message") if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        try:
            status = int(code) if code is not None else None
        except (TypeError, ValueError):
            status = None
        raise OpenRouterCallError(
            f"OpenRouter returned an error payload: {message}",
            status=status,
            retryable=status in RETRYABLE_STATUS if status else False,
        )

    choices = payload.get("choices") or []
    if not choices:
        raise OpenRouterCallError("OpenRouter response contained no choices.", retryable=True)

    message = (choices[0] or {}).get("message") or {}
    text = (message.get("content") or "").strip()
    if text:
        return text

    finish = (choices[0] or {}).get("finish_reason")
    if finish == "length":
        raise OpenRouterCallError(
            "Empty completion with finish_reason='length' - max_tokens is too small "
            "for this model (reasoning tokens count against it).",
            retryable=False,
        )
    raise OpenRouterCallError(
        f"Empty completion (finish_reason={finish!r}).", retryable=True
    )


def _status_of(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, OpenRouterCallError):
        return exc.retryable
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    return exc.__class__.__name__ in {
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
        "Timeout",
        "ConnectionError",
    }


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
        try:
            raw = headers.get(key)
        except Exception:
            raw = None
        if not raw:
            continue
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
    return None


def _mentions_response_format(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_schema" in text or "structured output" in text


def _is_no_endpoints(exc: Exception) -> bool:
    text = str(exc).lower()
    return "no endpoints found" in text or "no allowed providers" in text


def _drop_require_parameters(extra_body: Dict[str, Any]) -> bool:
    provider = extra_body.get("provider")
    if not isinstance(provider, dict) or "require_parameters" not in provider:
        return False
    provider.pop("require_parameters")
    if not provider:
        extra_body.pop("provider")
    return True


def chat_completion(
    client: OpenAI,
    settings: OpenRouterSettings,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict[str, Any]] = None,
    debug_print_full_response: bool = False,
    reasoning: Optional[Dict[str, Any]] = None,
    **params: Any,
) -> Tuple[str, Dict[str, Any]]:
    extra_body: Dict[str, Any] = {"usage": {"include": True}}
    if reasoning:
        extra_body["reasoning"] = dict(reasoning)

    provider = settings.provider_preferences()
    if response_format is not None:
        provider = dict(provider or {})
        provider["require_parameters"] = True
    if provider:
        extra_body["provider"] = provider

    kwargs: Dict[str, Any] = dict(model=model, messages=messages, extra_body=extra_body, **params)
    if response_format is not None:
        kwargs["response_format"] = response_format

    attempt = 0
    relaxed_parameters = False

    while True:
        if settings.sleep_seconds > 0:
            time.sleep(settings.sleep_seconds)

        try:
            response = client.chat.completions.create(**kwargs)
            if debug_print_full_response:
                print("\n--- FULL API RESPONSE ---")
                print(response.model_dump_json(indent=2))
                sys.stdout.flush()
            return _text_from(response), extract_provenance(response)

        except Exception as exc:
            if not relaxed_parameters and _is_no_endpoints(exc) and _drop_require_parameters(extra_body):
                relaxed_parameters = True
                print(
                    f"[openrouter] {model}: no endpoint accepts every requested parameter "
                    f"({exc}) - dropping require_parameters."
                )
                sys.stdout.flush()
                continue

            if attempt >= settings.max_retries or not _is_retryable(exc):
                raise

            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(settings.backoff_max, settings.backoff_base * (2 ** attempt))
                delay += random.uniform(0.0, min(1.0, delay / 4.0))
            print(
                f"[openrouter] {type(exc).__name__} on attempt {attempt + 1}/"
                f"{settings.max_retries + 1} (status={_status_of(exc)}): {exc}"
                f" - retrying in {delay:.1f}s"
            )
            sys.stdout.flush()
            attempt += 1
            time.sleep(delay)


def is_response_format_rejection(exc: Exception) -> bool:
    status = _status_of(exc)
    if status is not None and not (400 <= status < 500):
        return False
    return _mentions_response_format(exc)

@dataclass
class ProvenanceAccumulator:
    slug: str
    calls: int = 0
    providers_seen: List[str] = field(default_factory=list)
    resolved_models: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_cost: float = 0.0

    def add(self, provenance: Optional[Dict[str, Any]]) -> None:
        if not provenance:
            return
        self.calls += 1

        provider = provenance.get("provider")
        if provider and provider not in self.providers_seen:
            self.providers_seen.append(str(provider))

        resolved = provenance.get("model")
        if resolved and resolved not in self.resolved_models:
            self.resolved_models.append(str(resolved))

        for key, attr in (("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens"), ("reasoning_tokens", "reasoning_tokens")):
            value = provenance.get(key)
            if isinstance(value, (int, float)):
                setattr(self, attr, getattr(self, attr) + int(value))

        cost = provenance.get("cost")
        if isinstance(cost, (int, float)):
            self.total_cost += float(cost)

    def summary(self) -> Optional[Dict[str, Any]]:
        if self.calls == 0:
            return None
        return {
            "slug": self.slug,
            "calls": self.calls,
            "resolved_models": self.resolved_models,
            "providers_seen": self.providers_seen,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }
