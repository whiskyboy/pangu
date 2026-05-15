"""Tests for MLScoringStrategy — ML-based signal generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from pangu.models import Action, SignalStatus
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


class TestGenerateSignals:
    def test_basic_top_n(self):
        """Top-N stocks get BUY, rest get nothing."""
        scores = {f"60{i:04d}": float(i) for i in range(10)}
        scorer = _make_scorer_mock(scores)
        strategy = MLScoringStrategy(scorer, top_n=3)

        pool_df, signals = strategy.generate_signals(
            "2025-01-01", list(scores.keys()),
        )

        assert len(pool_df) == 10
        assert pool_df["rank"].min() == 1
        buys = [s for s in signals if s.action == Action.BUY]
        assert len(buys) == 3
        buy_syms = {s.symbol for s in buys}
        # Top-3 by score: 9, 8, 7
        assert buy_syms == {"600009", "600008", "600007"}

    def test_sell_signals_on_exit(self):
        """Stocks that were in prev top-N but dropped out get SELL."""
        scores = {
            "600001": 0.9,
            "600002": 0.8,
            "600003": 0.1,  # was top-2, now dropped
        }
        scorer = _make_scorer_mock(scores)
        strategy = MLScoringStrategy(scorer, top_n=2)

        prev_pool = pd.DataFrame({
            "symbol": ["600002", "600003"],
            "score": [0.8, 0.7],
            "rank": [1, 2],
        })

        pool_df, signals = strategy.generate_signals(
            "2025-01-01", list(scores.keys()), prev_pool=prev_pool,
        )

        sells = [s for s in signals if s.action == Action.SELL]
        assert len(sells) == 1
        assert sells[0].symbol == "600003"
        assert sells[0].signal_status == SignalStatus.EXIT

    def test_sustained_status(self):
        """Stocks staying in top-N get SUSTAINED status."""
        scores = {"600001": 0.9, "600002": 0.8, "600003": 0.1}
        scorer = _make_scorer_mock(scores)
        strategy = MLScoringStrategy(scorer, top_n=2)

        prev_pool = pd.DataFrame({
            "symbol": ["600001", "600002"],
            "score": [0.85, 0.75],
            "rank": [1, 2],
        })

        _, signals = strategy.generate_signals(
            "2025-01-01", list(scores.keys()), prev_pool=prev_pool,
        )

        buys = [s for s in signals if s.action == Action.BUY]
        sustained = [s for s in buys if s.signal_status == SignalStatus.SUSTAINED]
        assert len(sustained) == 2

    def test_star_exclusion(self):
        """688/689 prefix stocks should be excluded from pool."""
        all_scores = {"688001": 0.9, "600001": 0.5, "600002": 0.3}
        scorer = _make_scorer_mock(all_scores)
        strategy = MLScoringStrategy(scorer, top_n=2)

        pool_df, signals = strategy.generate_signals(
            "2025-01-01", list(all_scores.keys()),
        )

        # scorer.score is called with filtered pool (no 688)
        call_args = scorer.score.call_args
        called_pool = call_args[0][1]
        assert "688001" not in called_pool
        assert "600001" in called_pool

    def test_empty_scores(self):
        """Empty scorer response returns empty pool_df and signals."""
        scorer = MagicMock()
        scorer.score.return_value = pd.Series(dtype="float64", name="score")
        strategy = MLScoringStrategy(scorer, top_n=5)

        pool_df, signals = strategy.generate_signals("2025-01-01", [])

        assert pool_df.empty
        assert len(signals) == 0

    def test_no_prev_pool(self):
        """Without prev_pool, all top-N are NEW_ENTRY, no SELL signals."""
        scores = {"600001": 0.9, "600002": 0.5}
        scorer = _make_scorer_mock(scores)
        strategy = MLScoringStrategy(scorer, top_n=1)

        pool_df, signals = strategy.generate_signals(
            "2025-01-01", list(scores.keys()),
        )

        buys = [s for s in signals if s.action == Action.BUY]
        sells = [s for s in signals if s.action == Action.SELL]
        assert len(buys) == 1
        assert buys[0].signal_status == SignalStatus.NEW_ENTRY
        assert len(sells) == 0

    def test_pool_df_columns(self):
        """pool_df should have symbol, score, rank columns."""
        scores = {"600001": 0.9, "600002": 0.5}
        scorer = _make_scorer_mock(scores)
        strategy = MLScoringStrategy(scorer, top_n=2)

        pool_df, _ = strategy.generate_signals(
            "2025-01-01", list(scores.keys()),
        )

        assert set(pool_df.columns) >= {"symbol", "score", "rank"}
        assert pool_df["rank"].min() == 1
        assert pool_df["rank"].max() == 2

    def test_normalization(self):
        """Scores should be normalized to [0, 1]."""
        scores = {"600001": 100.0, "600002": 50.0, "600003": 0.0}
        scorer = _make_scorer_mock(scores)
        strategy = MLScoringStrategy(scorer, top_n=3)

        pool_df, _ = strategy.generate_signals(
            "2025-01-01", list(scores.keys()),
        )

        assert pool_df["score"].max() == pytest.approx(1.0)
        assert pool_df["score"].min() == pytest.approx(0.0)

    def test_equal_scores_normalization(self):
        """When all scores are equal, normalization should return 0.5."""
        scores = {"600001": 5.0, "600002": 5.0}
        scorer = _make_scorer_mock(scores)
        strategy = MLScoringStrategy(scorer, top_n=2)

        pool_df, _ = strategy.generate_signals(
            "2025-01-01", list(scores.keys()),
        )

        for val in pool_df["score"]:
            assert val == pytest.approx(0.5)
