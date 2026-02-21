"""Shared utilities — CircuitBreaker, retry, throttle, date helpers."""

from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry / circuit-breaker infrastructure — PRD §4.1.1
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Simple consecutive-failure circuit breaker (thread-safe)."""

    def __init__(self, threshold: int = 5, cooldown: float = 300.0) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._consecutive_failures = 0
        self._open_until: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._consecutive_failures >= self._threshold:
                if time.monotonic() < self._open_until:
                    return True
                # Cooldown expired — reset so next failure doesn't re-open immediately
                self._consecutive_failures = 0
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._open_until = time.monotonic() + self._cooldown
                logger.warning(
                    "Circuit breaker OPEN — %d consecutive failures, cooling down %.0fs",
                    self._consecutive_failures,
                    self._cooldown,
                )


def retry_call(fn, *, max_retries: int = 3, backoff_base: float = 2.0,
               circuit: CircuitBreaker | None = None):
    """Call *fn()* with exponential back-off and optional circuit breaker.

    Returns the result of *fn()* on success, raises the last exception on
    exhaustion.
    """
    if circuit and circuit.is_open:
        raise RuntimeError("Circuit breaker is open — skipping call")

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = fn()
            if circuit:
                circuit.record_success()
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if circuit:
                circuit.record_failure()
            if attempt < max_retries - 1:
                wait = backoff_base ** (attempt + 1)
                logger.warning("Retry %d/%d after %.1fs: %s", attempt + 1, max_retries, wait, exc)
                time.sleep(wait)
            else:
                logger.warning("Retry %d/%d exhausted: %s", attempt + 1, max_retries, exc)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ThrottleMixin — shared rate-limiting for AkShare providers
# ---------------------------------------------------------------------------

class ThrottleMixin:
    """Mixin providing ``_throttle()`` for AkShare API rate limiting."""

    def __init_throttle__(self, interval: float = 0.5) -> None:
        self._interval = interval
        self._last_call: float = 0.0
        self._throttle_lock = threading.Lock()

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def date_str(days_ago: int = 0) -> str:
    """Return ``(now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")``."""
    from pangu.tz import now

    return (now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
