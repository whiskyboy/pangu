"""ML-based stock scoring strategy.

Replaces the fixed-weight z-score ranking (MultiFactorStrategy) with
LightGBM model predictions on Alpha158 factors.

Production rebalance uses LLM-TopkDropout (two-stage TopkDropout):
ML coarsely selects candidate pools, LLM finely picks within them.
Falls back to ML-only TopkDropout when the LLM fails.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pangu.ml.scorer import MLScorer

logger = logging.getLogger(__name__)

_STAR_PREFIXES = ("688", "689")


def pool_score_rank_maps(
    pool_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int]]:
    """Return (score_map, rank_map) keyed by symbol from a scored pool DataFrame.

    Both maps are empty when ``pool_df`` is empty or missing the expected
    columns. Used across T4 and ML strategy code to avoid repeated
    ``dict(zip(...))`` boilerplate.
    """
    if pool_df is None or pool_df.empty:
        return {}, {}
    score_map = dict(zip(pool_df["symbol"], pool_df["score"], strict=False)) if "score" in pool_df.columns else {}
    rank_map = dict(zip(pool_df["symbol"], pool_df["rank"], strict=False)) if "rank" in pool_df.columns else {}
    return score_map, rank_map


class MLScoringStrategy:
    """Score stocks via ML model ensemble; expose candidate-pool APIs.

    Parameters
    ----------
    scorer : MLScorer instance (models must be pre-loaded)
    top_n : target portfolio size (default 25, matches backtest optimum)
    buy_candidate_size : non-held top-ranked pool size for LLM BUY decision
    sell_candidate_size : held bottom-ranked pool size for LLM SELL decision
    n_drop : per-rebalance turnover (fallback uses this when LLM under-fills)
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
        exclude_star: bool = True,
    ) -> None:
        if buy_candidate_size < n_drop:
            raise ValueError(f"buy_candidate_size ({buy_candidate_size}) must be >= n_drop ({n_drop})")
        if sell_candidate_size < n_drop:
            raise ValueError(f"sell_candidate_size ({sell_candidate_size}) must be >= n_drop ({n_drop})")
        self._scorer = scorer
        self._top_n = top_n
        self._buy_candidate_size = buy_candidate_size
        self._sell_candidate_size = sell_candidate_size
        self._n_drop = n_drop
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
        return pd.DataFrame(
            {
                "symbol": scores.index,
                "score": scores.values,
                "rank": ranks.values,
            }
        )

    # ------------------------------------------------------------------
    # Candidate pools
    # ------------------------------------------------------------------

    def get_buy_candidate_pool(
        self,
        pool_df: pd.DataFrame,
        holdings: list[str],
    ) -> list[str]:
        """Top ``buy_candidate_size`` non-held symbols by ML rank (ascending)."""
        if pool_df.empty:
            return []
        held = set(holdings)
        non_held = pool_df[~pool_df["symbol"].isin(held)]
        if non_held.empty:
            return []
        ordered = non_held.sort_values(
            ["rank", "symbol"],
            ascending=[True, True],
        )
        return ordered["symbol"].head(self._buy_candidate_size).tolist()

    def get_sell_candidate_pool(
        self,
        pool_df: pd.DataFrame,
        holdings: list[str],
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
                ["rank", "symbol"],
                ascending=[False, True],
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
        _, rank_map = pool_score_rank_maps(pool_df)
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
        _, rank_map = pool_score_rank_maps(pool_df)
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
            ["rank", "symbol"],
            ascending=[True, True],
        )
        return ordered["symbol"].head(self._top_n).tolist()


# ---------------------------------------------------------------------------
# Factory helper (shared by main.py bootstrap and T5 post-train hot-load)
# ---------------------------------------------------------------------------


def try_build_ml_strategy(
    db,
    ml_cfg: dict,
    strategy_cfg: dict,
) -> "MLScoringStrategy | None":
    """Construct an MLScoringStrategy from config, returning None if models are missing.

    Centralises the bootstrap rules so callers (``main.build_components`` at
    startup and ``tasks.update_model`` after first training) stay in sync.
    """
    if not ml_cfg.get("enabled", False):
        return None
    from pangu.ml.scorer import MLScorer

    model_dir = ml_cfg.get("model_dir", "models")
    try:
        scorer = MLScorer(model_dir=model_dir, db=db)
    except FileNotFoundError:
        return None
    return MLScoringStrategy(
        scorer,
        top_n=strategy_cfg.get("top_n", 25),
        buy_candidate_size=ml_cfg.get("buy_candidate_size", 10),
        sell_candidate_size=ml_cfg.get("sell_candidate_size", 5),
        n_drop=ml_cfg.get("n_drop", 3),
    )
