from __future__ import annotations

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_env(path: str | None = None) -> bool:
    dotenv_path = path or os.path.join(_REPO_ROOT, ".env")
    if not os.path.isfile(dotenv_path):
        return False

    try:
        from dotenv import load_dotenv
    except ImportError:
        return _load_env_fallback(dotenv_path)

    load_dotenv(dotenv_path, override=False)
    return True


def _load_env_fallback(dotenv_path: str) -> bool:
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
    return True
