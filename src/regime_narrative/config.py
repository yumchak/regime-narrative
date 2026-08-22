"""Settings loader.

Every tunable in this project lives in ``settings.yaml``. Nothing in ``src/``
hardcodes a value that appears there -- including the LLM model name, which
the submission requires to be swappable in one line.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "settings.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def load_env_file(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Read ``.env`` into the process environment. Returns the names it set.

    Written out rather than pulling in python-dotenv: it is a dozen lines, and a
    credential path is a bad place to add a dependency nobody reads.

    A real environment variable wins over the file by default, so a CI secret or
    a one-off ``ANTHROPIC_API_KEY=... python ...`` is never silently overridden
    by a stale file on disk.

    ``.env`` is gitignored and a test asserts that no tracked file contains a
    key, because the failure mode here is permanent: git remembers.
    """
    path = path or ENV_PATH
    if not path.exists():
        return []

    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Strip one matched pair of surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if override or not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


# Load on import so every script, the dashboard and the test suite all see the
# same credentials without each one remembering to ask.
load_env_file()


class Settings(dict):
    """Dict with dotted access, so ``cfg["news"]["window_days"]`` can also be
    written ``cfg.news.window_days`` at call sites where that reads better."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Settings(value) if isinstance(value, dict) else value


@lru_cache(maxsize=1)
def load_settings(path: str | Path | None = None) -> Settings:
    target = Path(path) if path else SETTINGS_PATH
    with open(target, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings(raw)


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the project root, creating parents as needed."""
    p = PROJECT_ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(*parts: str) -> Path:
    cfg = load_settings()
    p = PROJECT_ROOT / cfg["paths"]["cache"]
    p = p.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir(*parts: str) -> Path:
    cfg = load_settings()
    p = PROJECT_ROOT / cfg["paths"]["outputs"]
    p = p.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def study_end_date() -> date:
    cfg = load_settings()
    end = cfg["study"]["end_date"]
    if end:
        return date.fromisoformat(str(end))
    return date.today()


def study_start_date() -> date:
    return date.fromisoformat(str(load_settings()["study"]["start_date"]))


def require_env(name: str, *, hint: str = "") -> str:
    """Fetch a required credential, failing with an actionable message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. {hint}".strip()
        )
    return value


def has_env(name: str) -> bool:
    return bool(os.environ.get(name))
