"""ML-based stock scoring strategy.

Replaces the fixed-weight z-score ranking (MultiFactorStrategy) with
LightGBM model predictions on Alpha158 factors.

Production rebalance uses LLM-TopkDropout (two-stage TopkDropout):
ML coarsely selects candidate pools, LLM finely picks within them.
Falls back to classic TopkDropout when LLM fails.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from pangu.models import Action, SignalStatus, TradeSignal

if TYPE_CHECKING:
    from pangu.ml.scorer import MLScorer

logger = logging.getLogger(__name__)

_STAR_PREFIXES = ("688", "689")


class MLScoringStrategy:
    """Score stocks via ML model ensemble; expose candidate-pool APIs.

    Parameters
    ----------
    scorer : MLScorer instance (models must be pre-loaded)
    top_n : target portfolio size (default 25, matches backtest optimum)
    buy_candidate_size : non-held top-ranked pool size for LLM BUY decision
    sell_candidate_size : held bottom-ranked pool size for LLM SELL decision
    n_drop : per-rebalance turnover (fallback uses this when LLM under-fills)
    buy_threshold : minimum normalized score to trigger BUY (legacy path only)
    exclude_star : exclude STAR Market stocks (688/689 prefix)
    """

    def __init__(
        self,
        scorer: MLScorer,
        *,
        top_n: int = 25,
        buy_candidate_size: int = 10,
        sell_candidate_size: int = 5,
        n_drop: int = 3,
        buy_threshold: float = 0.0,
        exclude_star: bool = True,
    ) -> None:
        if buy_candidate_size < n_drop:
            raise ValueError(
                f"buy_candidate_size ({buy_candidate_size}) must be >= n_drop ({n_drop})"
            )
        if sell_candidate_size < n_drop:
            raise ValueError(
                f"sell_candidate_size ({sell_candidate_size}) must be >= n_drop ({n_drop})"
            )
        self._scorer = scorer
        self._top_n = top_n
        self._buy_candidate_size = buy_candidate_size
        self._sell_candidate_size = sell_candidate_size
        self._n_drop = n_drop
        self._buy_threshold = buy_threshold
        self._exclude_star = exclude_star

    # ------------------------------------------------------------------
    # Public properties (read-only)
    # ------------------------------------------------------------------

    @property
    def top_n(self) -> int:
        return self._top_n

    @property
    def buy_candidate_size(self) -> int:
        return self._buy_candidate_size

    @property
    def sell_candidate_size(self) -> int:
        return self._sell_candidate_size

    @property
    def n_drop(self) -> int:
        return self._n_drop

    # ------------------------------------------------------------------
    # Score → pool_df
    # ------------------------------------------------------------------

    def score_pool(self, date: str, pool: list[str]) -> pd.DataFrame:
        """Score *pool* on *date* and return ``[symbol, score, rank]`` DataFrame.

        Scores are min-max normalised to ``[0, 1]`` for downstream display.
        ``rank`` starts at 1 (best); ties resolved by ``method='min'``.
        STAR Market symbols are filtered when ``exclude_star`` is True.
        """
        if self._exclude_star:
            pool = [s for s in pool if not s.startswith(_STAR_PREFIXES)]

        raw_scores = self._scorer.score(date, pool)
        if raw_scores.empty:
            logger.warning("MLScoringStrategy.score_pool: empty scores for %s", date)
            return pd.DataFrame(columns=["symbol", "score", "rank"])

        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            scores = pd.Series(0.5, index=raw_scores.index, name="score")

        ranks = scores.rank(ascending=False, method="min").astype(int)
        return pd.DataFrame({
            "symbol": scores.index,
            "score": scores.values,
            "rank": ranks.values,
        })

    # ------------------------------------------------------------------
    # Candidate pools
    # ------------------------------------------------------------------

    def get_buy_candidate_pool(
        self, pool_df: pd.DataFrame, holdings: list[str],
    ) -> list[str]:
        """Top ``buy_candidate_size`` non-held symbols by ML rank (ascending)."""
        if pool_df.empty:
            return []
        held = set(holdings)
        non_held = pool_df[~pool_df["symbol"].isin(held)]
        if non_held.empty:
            return []
        ordered = non_held.sort_values(
            ["rank", "symbol"], ascending=[True, True],
        )
        return ordered["symbol"].head(self._buy_candidate_size).tolist()

    def get_sell_candidate_pool(
        self, pool_df: pd.DataFrame, holdings: list[str],
    ) -> list[str]:
        """Worst ``sell_candidate_size`` held symbols by ML rank (descending).

        Holdings missing from ``pool_df`` (suspended, ST handling, etc.) are
        treated as "worst possible" and placed at the front of the pool — they
        deserve immediate review.
        """
        if not holdings:
            return []
        pool_syms = set(pool_df["symbol"]) if not pool_df.empty else set()
        missing = sorted(s for s in holdings if s not in pool_syms)

        if pool_df.empty:
            held_ordered: list[str] = []
        else:
            held_df = pool_df[pool_df["symbol"].isin(holdings)]
            ordered = held_df.sort_values(
                ["rank", "symbol"], ascending=[False, True],
            )
            held_ordered = ordered["symbol"].tolist()

        combined = missing + held_ordered
        return combined[: self._sell_candidate_size]

    # ------------------------------------------------------------------
    # Fallback (degenerates LLM-TopkDropout into classic TopkDropout)
    # ------------------------------------------------------------------

    def fallback_sells(
        self,
        sell_pool: list[str],
        excluded: set[str],
        pool_df: pd.DataFrame,
        n: int,
    ) -> list[str]:
        """Pick *n* worst-ranked symbols from ``sell_pool``, skipping excluded.

        Symbols already absent from ``pool_df`` retain their pool ordering
        (they were placed at the front of the sell pool as "worst possible").
        """
        if n <= 0:
            return []
        rank_map = (
            dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
            if not pool_df.empty else {}
        )
        # Highest rank number first (worst), missing → +inf so they sort first
        candidates = [s for s in sell_pool if s not in excluded]
        candidates.sort(key=lambda s: (-rank_map.get(s, float("inf")), s))
        return candidates[:n]

    def fallback_buys(
        self,
        buy_pool: list[str],
        excluded: set[str],
        pool_df: pd.DataFrame,
        n: int,
    ) -> list[str]:
        """Pick *n* best-ranked symbols from ``buy_pool``, skipping excluded."""
        if n <= 0:
            return []
        rank_map = (
            dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
            if not pool_df.empty else {}
        )
        candidates = [s for s in buy_pool if s not in excluded]
        candidates.sort(key=lambda s: (rank_map.get(s, float("inf")), s))
        return candidates[:n]

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------

    def cold_start_portfolio(self, pool_df: pd.DataFrame) -> list[str]:
        """Pick ``top_n`` best-ranked symbols as the initial portfolio."""
        if pool_df.empty:
            return []
        ordered = pool_df.sort_values(
            ["rank", "symbol"], ascending=[True, True],
        )
        return ordered["symbol"].head(self._top_n).tolist()

    # ------------------------------------------------------------------
    # Legacy single-call signal generation (CLI `pangu score` only)
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        date: str,
        pool: list[str],
        *,
        prev_pool: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, list[TradeSignal]]:
        """Score stocks and generate naive BUY/SELL signals (legacy CLI path).

        For the production rebalance pipeline, use ``score_pool`` +
        ``get_buy_candidate_pool`` + ``get_sell_candidate_pool`` instead.
        """
        pool_df = self.score_pool(date, pool)
        if pool_df.empty:
            return pool_df, []

        now = datetime.now()
        prev_top: set[str] = set()
        if prev_pool is not None and not prev_pool.empty:
            prev_top = set(
                prev_pool[prev_pool["rank"] <= self._top_n]["symbol"].tolist()
            )

        signals: list[TradeSignal] = []
        score_map = dict(zip(pool_df["symbol"], pool_df["score"], strict=False))
        rank_map = dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
        for sym in sorted(score_map):
            score = float(score_map[sym])
            rank = int(rank_map[sym])
            in_top = rank <= self._top_n
            was_in_top = sym in prev_top

            if in_top and score >= self._buy_threshold:
                status = SignalStatus.SUSTAINED if was_in_top else SignalStatus.NEW_ENTRY
                signals.append(self._make_signal(
                    now, sym, Action.BUY, status, score,
                    f"ML rank={rank} score={score:.3f}",
                ))
            elif was_in_top and not in_top:
                signals.append(self._make_signal(
                    now, sym, Action.SELL, SignalStatus.EXIT, score,
                    f"ML exit top-{self._top_n}: rank={rank} score={score:.3f}",
                ))

        return pool_df, signals

    @staticmethod
    def _make_signal(
        now: datetime,
        symbol: str,
        action: Action,
        status: SignalStatus,
        score: float,
        reason: str,
    ) -> TradeSignal:
        return TradeSignal(
            timestamp=now,
            symbol=symbol,
            name=symbol,
            action=action,
            signal_status=status,
            days_in_top_n=0,
            price=0.0,
            confidence=score,
            source="ml",
            reason=reason,
            factor_score=score,
        )
