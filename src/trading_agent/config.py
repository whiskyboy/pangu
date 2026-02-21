"""Configuration loader — reads settings.toml with env-var substitution.

Key features (per PRD §7.2 + M1.3 plan):
- `$ENV_VAR` placeholders in TOML values are resolved from `os.environ` (walks nested dicts/lists)
- Singleton access via `get_settings()`
"""

from __future__ import annotations

import os
import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENV_VAR_RE = re.compile(r"\$([A-Z_][A-Z0-9_]*)")


def _substitute_env(value: Any) -> Any:
    """Recursively substitute ``$ENV_VAR`` placeholders in strings/containers."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    return value


@dataclass
class Settings:
    """Typed wrapper around settings.toml with env-var resolution."""

    raw: dict[str, Any] = field(repr=False)

    # Convenience accessors — populated in __post_init__
    system: dict[str, Any] = field(default_factory=dict)
    stock_pool: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    notification: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for section in (
            "system",
            "stock_pool",
            "strategy",
            "llm",
            "notification",
            "scheduler",
        ):
            setattr(self, section, self.raw.get(section, {}))


def load_settings(path: str | Path = "config/settings.toml") -> Settings:
    """Load *path*, substitute env vars, return ``Settings``."""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    raw = _substitute_env(raw)
    return Settings(raw=raw)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings(path: str | Path = "config/settings.toml") -> Settings:
    """Return the singleton ``Settings`` instance (lazy-loaded, thread-safe)."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = load_settings(path)
    return _settings


def reset_settings() -> None:
    """Clear cached settings (useful for testing)."""
    global _settings  # noqa: PLW0603
    with _settings_lock:
        _settings = None
