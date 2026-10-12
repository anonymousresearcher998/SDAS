from __future__ import annotations

from typing import Any, Optional

from src.backend import require_local_dir


def build_translator(
    model_id: str,
    system_prompt: str,
    *,
    remote: bool,
    language: str,
    country: str,
    mitigation_prompt: str = "",
    dtype: str = "auto",
    remote_reasoning: Optional[str] = None,
    remote_temperature: Optional[float] = 0.4,
) -> Any:
    if mitigation_prompt:
        system_prompt = f"{mitigation_prompt}\n\n{system_prompt}"
    system_prompt = system_prompt.format(language=language, country=country)

    if remote:
        from src.translator.openrouter_translator import (
            OpenRouterTranslator,
            OpenRouterTranslatorConfig,
        )

        print(f"TRANSLATOR BACKEND: REMOTE (OpenRouter) slug={model_id}")
        return OpenRouterTranslator(
            OpenRouterTranslatorConfig(
                language=language,
                model=model_id,
                system_prompt=system_prompt,
                reasoning_effort=remote_reasoning,
                temperature=remote_temperature,
            )
        )

    from src.translator.translator import MinimalTranslator, TranslatorConfig

    require_local_dir(model_id, what="translator", flag="--local_translator_model")
    print(f"TRANSLATOR BACKEND: LOCAL (MinimalTranslator) path={model_id}")
    return MinimalTranslator(
        TranslatorConfig(
            model_path=model_id,
            language=language,
            dtype=dtype,
            system_prompt=system_prompt,
        )
    )
