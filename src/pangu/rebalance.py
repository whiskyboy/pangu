"""Rebalance cadence configuration.

A ``RebalanceSchedule`` decides whether a given trading day is a rebalance
day under one of two patterns:

* ``weekly:N`` — every week on weekday ``N`` (1=Mon..5=Fri). If that
  weekday is non-trading, defer to the next trading day.
* ``monthly:N`` — every month on day-of-month ``N`` (1..28). If that
  day is non-trading, defer to the next trading day.

The schedule does **not** dedup cross-period defers. If a Friday
rebalance defers to next Monday, *and* the next Friday is also a
rebalance day, both fire — accepted by design (matches ETF 定投 style
cadence and keeps the algorithm trivially predictable).

Used by both production (``utils.is_rebalance_day``) and the backtest
engine. Production and backtest can carry **different** values but share
the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta
from typing import Any, Callable, Literal

Mode = Literal["weekly", "monthly"]

_VALID_MODES: tuple[Mode, ...] = ("weekly", "monthly")


@dataclass(frozen=True)
class RebalanceSchedule:
    """Frozen schedule descriptor (weekly day-of-week / monthly day-of-month).

    Attributes
    ----------
    mode :
        ``"weekly"`` or ``"monthly"``.
    day :
        ``weekly`` ⇒ 1..5 (1=Mon..5=Fri).  ``monthly`` ⇒ 1..28 (28 cap
        avoids February edge cases).
    """

    mode: Mode
    day: int

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Invalid rebalance mode {self.mode!r}; must be one of {_VALID_MODES}")
        if self.mode == "weekly":
            if not (1 <= self.day <= 5):
                raise ValueError(f"weekly_day must be 1..5 (Mon..Fri); got {self.day}")
        else:  # monthly
            if not (1 <= self.day <= 28):
                raise ValueError(f"monthly_day must be 1..28; got {self.day}")

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_string(cls, spec: str) -> "RebalanceSchedule":
        """Parse a CLI-friendly ``"<mode>:<day>"`` spec.

        Examples
        --------
        >>> RebalanceSchedule.from_string("weekly:1")
        RebalanceSchedule(mode='weekly', day=1)
        >>> RebalanceSchedule.from_string("monthly:15")
        RebalanceSchedule(mode='monthly', day=15)
        """
        if not isinstance(spec, str) or ":" not in spec:
            raise ValueError(f"Invalid --rebalance spec {spec!r}; expected 'weekly:N' or 'monthly:N'")
        mode_str, _, day_str = spec.partition(":")
        try:
            day = int(day_str)
        except ValueError as exc:  # noqa: BLE001
            raise ValueError(f"Invalid --rebalance day {day_str!r}; must be int") from exc
        return cls(mode=mode_str, day=day)  # type: ignore[arg-type]

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "RebalanceSchedule":
        """Build from a ``[rebalance]`` TOML section (or ``None`` for defaults).

        Defaults: ``mode="weekly", weekly_day=1`` — preserves the original
        ISO-week-first-trading-day behaviour for existing deployments.
        """
        cfg = cfg or {}
        mode = cfg.get("mode", "weekly")
        if mode == "weekly":
            return cls(mode="weekly", day=int(cfg.get("weekly_day", 1)))
        if mode == "monthly":
            return cls(mode="monthly", day=int(cfg.get("monthly_day", 1)))
        # Let __post_init__ raise a clear error
        return cls(mode=mode, day=int(cfg.get("weekly_day", 1)))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Core predicate
    # ------------------------------------------------------------------

    def matches(self, today: _date, is_trading_day: Callable[[_date], bool]) -> bool:
        """Return True iff *today* is a rebalance day under this schedule.

        Algorithm
        ---------
        1. ``today`` must itself be a trading day.
        2. Compute the *latest target date* ``T ≤ today`` whose calendar
           position matches the schedule (weekday for weekly mode,
           day-of-month for monthly mode).
        3. Scan forward from ``T``; the first trading day reached is the
           materialised rebalance day for that target. Return ``today == T*``.
        4. If no trading day is reached on or before ``today`` (calendar
           is sparse / cold start with partial history), return False.

        Parameters
        ----------
        today :
            Candidate trading day.
        is_trading_day :
            Predicate ``date -> bool``. In production this wraps
            ``Database.is_trading_day``; in backtest it wraps a set of
            known trading dates from the price index.
        """
        if not is_trading_day(today):
            return False

        target = self._latest_target_on_or_before(today)

        d = target
        while d <= today:
            if is_trading_day(d):
                return d == today
            d += timedelta(days=1)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _latest_target_on_or_before(self, today: _date) -> _date:
        """Find the most recent ``T ≤ today`` matching the schedule's
        calendar position (weekday or day-of-month).
        """
        if self.mode == "weekly":
            # isoweekday(): Mon=1 .. Sun=7; self.day in 1..5
            delta = (today.isoweekday() - self.day) % 7
            return today - timedelta(days=delta)
        # monthly: latest occurrence of `self.day` on or before today.
        if today.day >= self.day:
            return today.replace(day=self.day)
        # Wrap to previous month
        if today.month == 1:
            return today.replace(year=today.year - 1, month=12, day=self.day)
        return today.replace(month=today.month - 1, day=self.day)
