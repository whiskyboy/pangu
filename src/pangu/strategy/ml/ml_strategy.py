"""ML-based stock scoring strategy.

Replaces the fixed-weight z-score ranking (MultiFactorStrategy) with
LightGBM model predictions on Alpha158 factors.
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
    """Score stocks via ML model ensemble, generate BUY/SELL signals.

    Unlike :class:`MultiFactorStrategy` which uses fixed-weight z-scores,
    this strategy delegates scoring entirely to an :class:`MLScorer`
    (Alpha158 + LightGBM ensemble).

    Parameters
    ----------
    scorer : MLScorer instance (models must be pre-loaded)
    top_n : number of top-ranked stocks to select
    buy_threshold : minimum normalized score to trigger BUY (0-1)
    exclude_star : exclude STAR Market stocks (688/689 prefix)
    """

    def __init__(
        self,
        scorer: MLScorer,
        *,
        top_n: int = 25,
        buy_threshold: float = 0.0,
        exclude_star: bool = True,
    ) -> None:
        self._scorer = scorer
        self._top_n = top_n
        self._buy_threshold = buy_threshold
        self._exclude_star = exclude_star

    def generate_signals(
        self,
        date: str,
        pool: list[str],
        *,
        prev_pool: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, list[TradeSignal]]:
        """Score stocks and generate BUY/SELL signals.

        Parameters
        ----------
        date : target date (YYYY-MM-DD)
        pool : symbols to score
        prev_pool : previous pool_df with columns [symbol, score, rank]

        Returns
        -------
        (pool_df, signals) — same output shape as MultiFactorStrategy.
        pool_df has columns [symbol, score, rank].
        """
        # Filter STAR Market if configured
        if self._exclude_star:
            pool = [s for s in pool if not s.startswith(_STAR_PREFIXES)]

        # 1. ML scoring
        raw_scores = self._scorer.score(date, pool)
        if raw_scores.empty:
            logger.warning("MLScoringStrategy: no scores returned for %s", date)
            return pd.DataFrame(columns=["symbol", "score", "rank"]), []

        # 2. Normalize to [0, 1]
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            scores = pd.Series(0.5, index=raw_scores.index, name="score")

        # 3. Rank (1 = best)
        ranks = scores.rank(ascending=False, method="min").astype(int)

        # 4. Build pool_df
        pool_df = pd.DataFrame({
            "symbol": scores.index,
            "score": scores.values,
            "rank": ranks.values,
        })

        # 5. Generate signals
        now = datetime.now()
        prev_top: set[str] = set()
        if prev_pool is not None and not prev_pool.empty:
            prev_top = set(
                prev_pool[prev_pool["rank"] <= self._top_n]["symbol"].tolist()
            )

        signals: list[TradeSignal] = []
        for sym in sorted(scores.index):
            score = float(scores[sym])
            rank = int(ranks[sym])

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
