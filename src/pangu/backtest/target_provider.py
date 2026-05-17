"""Target-portfolio providers for the backtest engine.

The backtest engine simulates day-by-day NAV given a sequence of
**target portfolios** (which stocks to hold on each rebalance day).
This module decouples that "what to hold" decision from the engine
itself so the same engine can be reused for two purposes:

1. **Score-based research backtest** (``ScoreBasedProvider``) — selects
   targets from a Score matrix via TopkDropout, point-in-time universe
   filtering, and sector caps.

2. **Production decision replay** (``ReplayProvider``) — replays a
   stored sequence of rebalance decisions (``portfolio_snapshots`` table)
   to evaluate how the production pipeline would have performed.

Both providers satisfy the same Protocol; the engine code is agnostic.
"""

from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd


class TargetProvider(Protocol):
    """Strategy for picking the target portfolio on a rebalance day."""

    def get_target(
        self,
        rebalance_date: pd.Timestamp,
        score_date: pd.Timestamp,
        current_holdings: dict[str, int],
    ) -> list[str]:
        """Return the desired list of symbols to hold after this rebalance.

        Parameters
        ----------
        rebalance_date :
            The day we are placing orders (trades execute at this day's open).
        score_date :
            The reference date for "current information" (T-1 close).
            The provider must NOT peek at any data after this date.
        current_holdings :
            Map of ``symbol → shares`` immediately before this rebalance.
            Used by TopkDropout-style providers to decide what to keep.
        """
        ...


class ScoreBasedProvider:
    """TopkDropout target selection driven by a score matrix.

    Encapsulates the original engine selection logic: top_n selection,
    optional TopkDropout (only swap the worst n_drop held stocks),
    point-in-time constituent filtering, and per-sector caps.
    """

    def __init__(
        self,
        scores: pd.DataFrame,
        *,
        top_n: int,
        n_drop: int = 0,
        universe_fn: Callable[[str], set[str]] | None = None,
        sector_map: dict[str, str] | None = None,
        max_per_sector: int | None = None,
    ) -> None:
        self._scores = scores
        self._top_n = top_n
        self._n_drop = n_drop
        self._universe_fn = universe_fn
        self._sector_map = sector_map
        self._max_per_sector = max_per_sector

    def get_target(
        self,
        rebalance_date: pd.Timestamp,
        score_date: pd.Timestamp,
        current_holdings: dict[str, int],
    ) -> list[str]:
        prev_scores = self._scores.loc[score_date].dropna()

        if self._universe_fn is not None:
            universe = self._universe_fn(score_date.strftime("%Y-%m-%d"))
            prev_scores = prev_scores[prev_scores.index.isin(universe)]

        if len(prev_scores) >= self._top_n:
            if self._n_drop > 0 and current_holdings:
                held = set(current_holdings.keys())
                held_scores = {s: prev_scores.get(s, float("-inf")) for s in held}
                sorted_held = sorted(held_scores, key=held_scores.get, reverse=True)
                n_keep = max(0, min(len(sorted_held), self._top_n - self._n_drop))
                kept = sorted_held[:n_keep]
                n_buy = self._top_n - len(kept)
                top_ranked = prev_scores.nlargest(self._top_n + self._n_drop + n_buy)
                kept_set = set(kept)
                to_buy = [s for s in top_ranked.index if s not in kept_set][:n_buy]
                target = kept + to_buy
            else:
                target = prev_scores.nlargest(self._top_n).index.tolist()
        else:
            target = prev_scores.sort_values(ascending=False).index.tolist()

        if self._max_per_sector and self._sector_map is not None:
            target = _apply_sector_cap(
                target,
                prev_scores,
                self._sector_map,
                top_n=self._top_n,
                max_per_sector=self._max_per_sector,
            )

        return target


class ReplayProvider:
    """Replay a stored sequence of historical rebalance decisions.

    The engine asks for the target on each rebalance date; this provider
    simply looks up the recorded decision. Unlike ``ScoreBasedProvider``
    it does not re-derive targets — production already made the call
    (including LLM judging, sector constraints, etc.) and that decision
    was persisted to ``portfolio_snapshots``.
    """

    def __init__(self, decisions: dict[str, list[str]]) -> None:
        """*decisions*: map ``date_str (YYYY-MM-DD) → list[symbol]``."""
        self._decisions = {str(k): list(v) for k, v in decisions.items()}
        # Sorted dates allow forward search for the latest decision ≤ date
        self._dates_sorted = sorted(self._decisions.keys())

    def get_target(
        self,
        rebalance_date: pd.Timestamp,
        score_date: pd.Timestamp,
        current_holdings: dict[str, int],
    ) -> list[str]:
        # Prefer exact match on rebalance date; else last decision on/before it
        key = rebalance_date.strftime("%Y-%m-%d")
        if key in self._decisions:
            return self._decisions[key]
        # Binary-search the latest decision ≤ rebalance_date
        from bisect import bisect_right

        idx = bisect_right(self._dates_sorted, key) - 1
        if idx < 0:
            return list(current_holdings.keys())
        return self._decisions[self._dates_sorted[idx]]


# ---------------------------------------------------------------------------
# Internal helpers (shared by providers and the engine)
# ---------------------------------------------------------------------------


def _apply_sector_cap(
    target: list[str],
    scores: pd.Series,
    sector_map: dict[str, str],
    *,
    top_n: int,
    max_per_sector: int,
) -> list[str]:
    """Trim target so no sector exceeds *max_per_sector* stocks.

    Mirrors the original ``BacktestEngine._apply_sector_cap`` logic but
    lives here so it's accessible to ``ScoreBasedProvider`` without
    coupling to the engine.
    """
    target_sorted = sorted(target, key=lambda s: scores.get(s, float("-inf")), reverse=True)
    sector_counts: dict[str, int] = {}
    result: list[str] = []

    for sym in target_sorted:
        sector = sector_map.get(sym, "未知")
        cnt = sector_counts.get(sector, 0)
        if cnt < max_per_sector:
            result.append(sym)
            sector_counts[sector] = cnt + 1
        if len(result) >= top_n:
            break

    if len(result) < top_n:
        result_set = set(result)
        for sym in scores.sort_values(ascending=False).index:
            if sym in result_set:
                continue
            sector = sector_map.get(sym, "未知")
            cnt = sector_counts.get(sector, 0)
            if cnt < max_per_sector:
                result.append(sym)
                result_set.add(sym)
                sector_counts[sector] = cnt + 1
            if len(result) >= top_n:
                break

    return result
