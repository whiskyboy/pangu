"""Timezone utilities — single source of truth for ``now()``.

All modules should use :func:`now` instead of ``datetime.now()`` or
``datetime.now(tz=timezone.utc)`` to ensure consistent timezone handling
across the system.

The timezone is read from ``config/settings.toml`` → ``[system].timezone``.
Falls back to ``Asia/Shanghai`` when the config cannot be loaded.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_tz: ZoneInfo | None = None


def _get_tz() -> ZoneInfo:
    """Lazily resolve the system timezone from settings.toml."""
    global _tz  # noqa: PLW0603
    if _tz is not None:
        return _tz
    try:
        from trading_agent.config import get_settings

        tz_name = get_settings().system.get("timezone", "Asia/Shanghai")
        _tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        _tz = ZoneInfo("Asia/Shanghai")
    return _tz


def now() -> datetime:
    """Return the current time in the configured system timezone."""
    return datetime.now(tz=_get_tz())


def today_str() -> str:
    """Return today's date as ``YYYY-MM-DD`` in the system timezone."""
    return now().strftime("%Y-%m-%d")
