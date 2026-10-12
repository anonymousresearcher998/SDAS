from __future__ import annotations

from typing import Any, Optional

from src.backend import require_local_dir


def build_judge(
    *,
    local_model_path: Optional[str],
    remote_model: Optional[str],
    system_prompt: str,
    max_new_tokens: int = 256,
    dtype: str = "auto",
) -> Any:
    local_model_path = (local_model_path or "").strip()
    remote_model = (remote_model or "").strip()

    if local_model_path and remote_model:
        raise ValueError(
            "judge: --local_judge_model and --remote_judge_model are mutually exclusive - "
            "pass exactly one."
        )
    if not local_model_path and not remote_model:
        raise ValueError(
            "judge: pass exactly one of --local_judge_model (a path to local weights) or "
            "--remote_judge_model (an OpenRouter slug)."
        )

    if remote_model:
        from src.judge.openrouter_judge import OpenRouterJudge, OpenRouterJudgeConfig

        print(f"JUDGE BACKEND: REMOTE (OpenRouter) slug={remote_model}")
        return OpenRouterJudge(
            OpenRouterJudgeConfig(
                model=remote_model,
                system_prompt=system_prompt,
            )
        )

    from src.judge.judge import LocalJudge

    require_local_dir(local_model_path, what="judge", flag="--local_judge_model")
    print(f"JUDGE BACKEND: LOCAL (LocalJudge) path={local_model_path}")
    return LocalJudge(
        model_path=local_model_path,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
    )
