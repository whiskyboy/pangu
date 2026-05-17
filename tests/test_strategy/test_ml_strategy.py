"""Tests for MLScoringStrategy — ML-based candidate pools + fallbacks."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from pangu.strategy.ml.ml_strategy import MLScoringStrategy


def _make_scorer_mock(scores: dict[str, float]) -> MagicMock:
    """Create a mock MLScorer that returns fixed scores.

    The mock dynamically returns only scores for symbols in the called pool.
    """
    scorer = MagicMock()

    def _score(date, pool):
        return pd.Series(
            {s: scores[s] for s in pool if s in scores},
            name="score",
        )

    scorer.score.side_effect = _score
    return scorer


# ---------------------------------------------------------------------------
# Two-stage TopkDropout: candidate pools + fallback + cold start
# ---------------------------------------------------------------------------


class TestScorePool:
    def test_score_pool_basic(self):
        scores = {"600001": 0.9, "600002": 0.5, "600003": 0.1}
        scorer = _make_scorer_mock(scores)
        strat = MLScoringStrategy(scorer, top_n=2)

        pool_df = strat.score_pool("2025-01-01", list(scores))
        assert list(pool_df.columns) == ["symbol", "score", "rank"]
        # Rank ordering: highest raw score → rank 1
        rank_of = dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
        assert rank_of["600001"] == 1
        assert rank_of["600003"] == 3

    def test_score_pool_filters_star_market(self):
        scores = {"688001": 0.95, "600001": 0.5}
        scorer = _make_scorer_mock(scores)
        strat = MLScoringStrategy(scorer, top_n=2)

        pool_df = strat.score_pool("2025-01-01", list(scores))
        assert "688001" not in pool_df["symbol"].tolist()

    def test_score_pool_empty(self):
        scorer = _make_scorer_mock({})
        strat = MLScoringStrategy(scorer, top_n=2)
        pool_df = strat.score_pool("2025-01-01", [])
        assert pool_df.empty
        assert list(pool_df.columns) == ["symbol", "score", "rank"]

    def test_score_pool_normalizes_scores(self):
        scores = {"600001": 100.0, "600002": 50.0, "600003": 0.0}
        scorer = _make_scorer_mock(scores)
        strat = MLScoringStrategy(scorer, top_n=3)
        pool_df = strat.score_pool("2025-01-01", list(scores))
        assert pool_df["score"].max() == pytest.approx(1.0)
        assert pool_df["score"].min() == pytest.approx(0.0)

    def test_score_pool_equal_scores_normalize_to_half(self):
        scores = {"600001": 5.0, "600002": 5.0}
        scorer = _make_scorer_mock(scores)
        strat = MLScoringStrategy(scorer, top_n=2)
        pool_df = strat.score_pool("2025-01-01", list(scores))
        for val in pool_df["score"]:
            assert val == pytest.approx(0.5)


class TestCandidatePools:
    def _make(self, n_buy=10, n_sell=5, n_drop=3, top_n=25):
        return MLScoringStrategy(
            _make_scorer_mock({}),
            top_n=top_n,
            buy_candidate_size=n_buy,
            sell_candidate_size=n_sell,
            n_drop=n_drop,
        )

    def test_buy_candidate_pool_top_non_held(self):
        strat = self._make(n_buy=3)
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E"],
                "score": [0.9, 0.8, 0.7, 0.6, 0.5],
                "rank": [1, 2, 3, 4, 5],
            }
        )
        # Hold B and D; non-held best-ranked: A, C, E
        out = strat.get_buy_candidate_pool(pool_df, holdings=["B", "D"])
        assert out == ["A", "C", "E"]

    def test_buy_candidate_pool_excludes_all_held(self):
        strat = self._make(n_buy=3)
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "score": [0.9, 0.8],
                "rank": [1, 2],
            }
        )
        assert strat.get_buy_candidate_pool(pool_df, ["A", "B"]) == []

    def test_sell_candidate_pool_worst_held(self):
        strat = self._make(n_sell=3)
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "score": [0.9, 0.8, 0.7, 0.6],
                "rank": [1, 2, 3, 4],
            }
        )
        # Hold all; worst-ranked first
        out = strat.get_sell_candidate_pool(pool_df, holdings=["A", "B", "C", "D"])
        assert out == ["D", "C", "B"]

    def test_sell_candidate_pool_missing_holdings_first(self):
        """Suspended / delisted holdings missing from pool_df rank first."""
        strat = self._make(n_sell=3)
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "score": [0.9, 0.8],
                "rank": [1, 2],
            }
        )
        # Hold A, B, X (X is missing — suspended)
        out = strat.get_sell_candidate_pool(pool_df, holdings=["A", "B", "X"])
        # Missing first, then worst-ranked held
        assert out[0] == "X"
        assert "B" in out

    def test_sell_candidate_pool_empty_holdings(self):
        strat = self._make()
        pool_df = pd.DataFrame(
            {
                "symbol": ["A"],
                "score": [0.9],
                "rank": [1],
            }
        )
        assert strat.get_sell_candidate_pool(pool_df, []) == []

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            MLScoringStrategy(
                _make_scorer_mock({}),
                top_n=25,
                buy_candidate_size=2,
                sell_candidate_size=5,
                n_drop=3,
            )
        with pytest.raises(ValueError):
            MLScoringStrategy(
                _make_scorer_mock({}),
                top_n=25,
                buy_candidate_size=10,
                sell_candidate_size=2,
                n_drop=3,
            )


class TestFallback:
    def _strat(self):
        return MLScoringStrategy(
            _make_scorer_mock({}),
            top_n=25,
            buy_candidate_size=5,
            sell_candidate_size=5,
            n_drop=3,
        )

    def test_fallback_sells_picks_worst_ranks_in_pool(self):
        strat = self._strat()
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "score": [0.9, 0.8, 0.7, 0.6],
                "rank": [1, 2, 3, 4],
            }
        )
        # sell pool = ["D", "C", "B", "A"] (worst first)
        out = strat.fallback_sells(["D", "C", "B", "A"], excluded=set(), pool_df=pool_df, n=2)
        # Worst rank first → D (rank 4), C (rank 3)
        assert out == ["D", "C"]

    def test_fallback_sells_skips_excluded(self):
        strat = self._strat()
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "score": [0.9, 0.8, 0.7],
                "rank": [1, 2, 3],
            }
        )
        out = strat.fallback_sells(["C", "B", "A"], excluded={"C"}, pool_df=pool_df, n=2)
        assert "C" not in out
        assert out == ["B", "A"]

    def test_fallback_buys_picks_best_ranks(self):
        strat = self._strat()
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "score": [0.9, 0.8, 0.7],
                "rank": [1, 2, 3],
            }
        )
        out = strat.fallback_buys(["A", "B", "C"], excluded={"A"}, pool_df=pool_df, n=2)
        assert out == ["B", "C"]

    def test_fallback_zero_count_returns_empty(self):
        strat = self._strat()
        pool_df = pd.DataFrame({"symbol": ["A"], "score": [0.9], "rank": [1]})
        assert strat.fallback_sells(["A"], set(), pool_df, n=0) == []
        assert strat.fallback_buys(["A"], set(), pool_df, n=0) == []

    def test_fallback_sells_missing_rank_treated_as_worst(self):
        """Symbols without a rank entry (e.g. suspended) get sorted first."""
        strat = self._strat()
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "score": [0.9, 0.8],
                "rank": [1, 2],
            }
        )
        # "X" not in pool_df → should be picked first as worst
        out = strat.fallback_sells(["X", "B", "A"], excluded=set(), pool_df=pool_df, n=2)
        assert out[0] == "X"


class TestColdStart:
    def test_cold_start_picks_top_n_by_rank(self):
        strat = MLScoringStrategy(
            _make_scorer_mock({}),
            top_n=3,
            buy_candidate_size=5,
            sell_candidate_size=5,
            n_drop=3,
        )
        pool_df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E"],
                "score": [0.9, 0.8, 0.7, 0.6, 0.5],
                "rank": [1, 2, 3, 4, 5],
            }
        )
        assert strat.cold_start_portfolio(pool_df) == ["A", "B", "C"]

    def test_cold_start_empty(self):
        strat = MLScoringStrategy(
            _make_scorer_mock({}),
            top_n=3,
            buy_candidate_size=5,
            sell_candidate_size=5,
            n_drop=3,
        )
        assert strat.cold_start_portfolio(pd.DataFrame()) == []
