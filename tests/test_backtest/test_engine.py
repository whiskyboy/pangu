"""Tests for BacktestEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.backtest.engine import BacktestEngine, BacktestResult


@pytest.fixture
def sample_data():
    """Create minimal test data: 5 stocks, 30 trading days."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=30, freq="B")
    stocks = ["A", "B", "C", "D", "E"]

    close = pd.DataFrame(
        100 + np.cumsum(np.random.randn(30, 5), axis=0),
        index=dates, columns=stocks,
    )
    open_ = close.shift(1).bfill()

    # adj_factor = 1.0 (no adjustments in test data)

    scores = pd.DataFrame(
        {s: np.linspace(5 - i, 5 - i + 0.1, 30) for i, s in enumerate(stocks)},
        index=dates,
    )

    benchmark = close.mean(axis=1)

    return scores, open_, close, benchmark


def test_backtest_basic(sample_data):
    scores, open_, close, benchmark = sample_data
    engine = BacktestEngine(top_n=2)
    result = engine.run(scores, open_, close, benchmark)

    assert isinstance(result, BacktestResult)
    assert len(result.nav) == 30
    assert result.nav.iloc[0] == pytest.approx(1_000_000, rel=0.01)
    assert "total_return" in result.metrics
    assert "sharpe" in result.metrics
    assert "max_drawdown" in result.metrics
    assert result.metrics["max_drawdown"] <= 0


def test_backtest_rebalance_happens(sample_data):
    scores, open_, close, benchmark = sample_data
    # Randomize scores so TopN changes each week → triggers rebalances
    np.random.seed(99)
    scores = pd.DataFrame(
        np.random.randn(30, 5),
        index=scores.index, columns=scores.columns,
    )
    engine = BacktestEngine(top_n=2)
    result = engine.run(scores, open_, close, benchmark)

    assert len(result.rebalance_log) >= 3


def test_backtest_top_n_respected(sample_data):
    scores, open_, close, benchmark = sample_data
    engine = BacktestEngine(top_n=3)
    result = engine.run(scores, open_, close, benchmark)

    for entry in result.rebalance_log:
        assert entry["n_holdings"] <= 3


def test_backtest_date_range(sample_data):
    scores, open_, close, benchmark = sample_data
    engine = BacktestEngine(top_n=2)
    result = engine.run(scores, open_, close, benchmark,
                        start="2024-01-15", end="2024-01-25")

    assert len(result.nav) < 30
    assert result.nav.index[0] >= pd.Timestamp("2024-01-15")
    assert result.nav.index[-1] <= pd.Timestamp("2024-01-25")


def test_backtest_benchmark_nav(sample_data):
    scores, open_, close, benchmark = sample_data
    engine = BacktestEngine(top_n=2)
    result = engine.run(scores, open_, close, benchmark)

    assert result.benchmark_nav.iloc[0] == pytest.approx(1_000_000, rel=0.01)
