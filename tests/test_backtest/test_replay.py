"""Tests for ReplayProvider + run_with_provider API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.backtest.engine import BacktestEngine, BacktestResult
from pangu.backtest.target_provider import ReplayProvider, ScoreBasedProvider


@pytest.fixture
def sample_panel():
    """5 stocks × 40 trading days of synthetic OHLCV."""
    np.random.seed(7)
    dates = pd.bdate_range("2024-01-01", periods=40, freq="B")
    stocks = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(
        100 + np.cumsum(np.random.randn(40, 5) * 0.5, axis=0),
        index=dates,
        columns=stocks,
    )
    open_ = close.shift(1).bfill()
    benchmark = close.mean(axis=1)
    return dates, stocks, open_, close, benchmark


def test_replay_provider_lookup_exact_date(sample_panel):
    dates, _, _, _, _ = sample_panel
    decisions = {
        dates[0].strftime("%Y-%m-%d"): ["A", "B"],
        dates[5].strftime("%Y-%m-%d"): ["C", "D"],
    }
    rp = ReplayProvider(decisions)
    assert rp.get_target(dates[0], dates[0], {}) == ["A", "B"]
    assert rp.get_target(dates[5], dates[4], {}) == ["C", "D"]


def test_replay_provider_falls_back_to_latest_prior(sample_panel):
    dates, _, _, _, _ = sample_panel
    decisions = {
        dates[0].strftime("%Y-%m-%d"): ["A", "B"],
        dates[10].strftime("%Y-%m-%d"): ["C", "D"],
    }
    rp = ReplayProvider(decisions)
    # mid-week date with no exact match → carries the dates[0] decision
    assert rp.get_target(dates[3], dates[2], {}) == ["A", "B"]
    # past dates[10] → carries that decision
    assert rp.get_target(dates[15], dates[14], {}) == ["C", "D"]


def test_replay_provider_returns_current_when_no_prior_decision(sample_panel):
    dates, _, _, _, _ = sample_panel
    decisions = {dates[20].strftime("%Y-%m-%d"): ["A"]}
    rp = ReplayProvider(decisions)
    holdings = {"X": 100, "Y": 200}
    # before any decision → keeps current holdings (no info)
    assert sorted(rp.get_target(dates[0], dates[0], holdings)) == ["X", "Y"]


def test_run_with_provider_matches_run_for_scores(sample_panel):
    """The legacy run(scores=...) should produce the same NAV as
    run_with_provider(ScoreBasedProvider(scores))."""
    dates, stocks, open_, close, benchmark = sample_panel
    scores = pd.DataFrame(
        {s: np.linspace(5 - i, 5 - i + 0.1, 40) for i, s in enumerate(stocks)},
        index=dates,
    )

    engine_legacy = BacktestEngine(top_n=2)
    legacy = engine_legacy.run(scores, open_, close, benchmark)

    engine_new = BacktestEngine(top_n=2)
    provider = ScoreBasedProvider(scores, top_n=2)
    new = engine_new.run_with_provider(
        provider,
        open_prices=open_,
        close_prices=close,
        benchmark_close=benchmark,
    )

    assert isinstance(legacy, BacktestResult)
    assert isinstance(new, BacktestResult)
    pd.testing.assert_series_equal(legacy.nav, new.nav)


def test_replay_provider_drives_backtest(sample_panel):
    """ReplayProvider over a couple of rebalance points produces a valid NAV."""
    dates, _, open_, close, benchmark = sample_panel
    # Decisions on the first day of each ISO-week (Monday)
    monday_dates = [d for d in dates if d.weekday() == 0]
    decisions = {}
    for i, mon in enumerate(monday_dates):
        holdings = ["A", "B"] if i % 2 == 0 else ["C", "D"]
        decisions[mon.strftime("%Y-%m-%d")] = holdings

    rp = ReplayProvider(decisions)
    engine = BacktestEngine(top_n=2, initial_capital=1_000_000.0)
    result = engine.run_with_provider(
        rp,
        open_prices=open_,
        close_prices=close,
        benchmark_close=benchmark,
    )

    assert len(result.nav) == len(dates)
    assert result.nav.iloc[0] == pytest.approx(1_000_000, rel=0.05)
    assert len(result.rebalance_log) >= 1
