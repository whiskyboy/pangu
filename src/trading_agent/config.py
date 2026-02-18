"""Configuration loader — reads settings.toml with env-var substitution.

Key features (per PRD §7.2 + M1.3 plan):
- `$ENV_VAR` placeholders in TOML values are resolved from `os.environ` (walks nested dicts/lists)
- LLM fallback providers are auto-pruned when their API key env var is missing
- Notification channels are auto-pruned when required config values are empty
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

# provider prefix → env var that must be non-empty
_PROVIDER_KEY_MAP: dict[str, str] = {
    "azure": "AZURE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# notification channel → list of env vars that must ALL be non-empty
_CHANNEL_REQUIRED_VARS: dict[str, list[str]] = {
    "feishu": ["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
    "email": ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"],
}


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


def _provider_prefix(provider: str) -> str:
    """Extract the provider prefix from a litellm-style model string (e.g. ``azure/gpt-4o`` → ``azure``)."""
    return provider.split("/", 1)[0] if "/" in provider else provider


def _has_api_key(provider: str) -> bool:
    """Check whether the API key env var for *provider* is set and non-empty."""
    prefix = _provider_prefix(provider)
    env_var = _PROVIDER_KEY_MAP.get(prefix)
    if env_var is None:
        # Unknown provider — keep it (user may have set it up outside our map)
        return True
    return bool(os.environ.get(env_var, "").strip())


def _channel_configured(channel: str) -> bool:
    """Check whether all required env vars for a notification channel are present."""
    required = _CHANNEL_REQUIRED_VARS.get(channel)
    if required is None:
        return True
    return all(os.environ.get(v, "").strip() for v in required)


@dataclass
class Settings:
    """Typed wrapper around settings.toml with env-var resolution."""

    raw: dict[str, Any] = field(repr=False)

    # Convenience accessors — populated in __post_init__
    system: dict[str, Any] = field(default_factory=dict)
    stock_pool: dict[str, Any] = field(default_factory=dict)
    data_source: dict[str, Any] = field(default_factory=dict)
    factor: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    notification: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    global_market: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for section in (
            "system",
            "stock_pool",
            "data_source",
            "factor",
            "strategy",
            "llm",
            "notification",
            "scheduler",
            "global_market",
        ):
            setattr(self, section, self.raw.get(section, {}))


def load_settings(path: str | Path = "config/settings.toml") -> Settings:
    """Load *path*, substitute env vars, prune providers/channels, return ``Settings``."""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    # 1. Recursive env-var substitution
    raw = _substitute_env(raw)

    # 2. LLM fallback auto-pruning
    llm = raw.get("llm", {})
    fallbacks = llm.get("fallback_providers", [])
    if fallbacks:
        if not isinstance(fallbacks, list):
            raise TypeError(f"llm.fallback_providers must be a list, got {type(fallbacks).__name__}")
        llm["fallback_providers"] = [p for p in fallbacks if _has_api_key(p)]

    # 3. Notification channel auto-detection
    notif = raw.get("notification", {})
    channels = notif.get("channels", [])
    if channels:
        if not isinstance(channels, list):
            raise TypeError(f"notification.channels must be a list, got {type(channels).__name__}")
        notif["channels"] = [ch for ch in channels if _channel_configured(ch)]

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
