from __future__ import annotations

import os
from typing import Optional, Tuple

LOCAL = "LOCAL"
REMOTE = "REMOTE"


def select_backend(
    *,
    local: Optional[str],
    remote: Optional[str],
    what: str,
    local_flag: str,
    remote_flag: str,
) -> Tuple[bool, str]:
    local = (local or "").strip()
    remote = (remote or "").strip()

    if local and remote:
        raise ValueError(
            f"{what}: {local_flag} ({local!r}) and {remote_flag} ({remote!r}) are mutually "
            f"exclusive - pass exactly one. {local_flag} loads weights from disk; "
            f"{remote_flag} is an OpenRouter slug."
        )
    if not local and not remote:
        raise ValueError(
            f"{what}: pass exactly one of {local_flag} (a path to local weights) or "
            f"{remote_flag} (an OpenRouter slug)."
        )

    if remote:
        if os.sep in remote and remote.startswith(os.sep):
            raise ValueError(
                f"{what}: {remote_flag}={remote!r} looks like a filesystem path. "
                f"Remote models are OpenRouter slugs such as 'openai/gpt-5.1'; "
                f"use {local_flag} for local weights."
            )
        return True, remote

    return False, local


def require_local_dir(model_path: str, *, what: str, flag: str) -> str:
    if not os.path.isdir(model_path):
        raise ValueError(
            f"{what}: {flag}={model_path!r} is not a directory"
        )
    return model_path
