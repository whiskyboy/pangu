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


# ---------------------------------------------------------------------------
# Rebalance strategy edge-case tests
# ---------------------------------------------------------------------------


class TestRebalanceEdgeCases:
    """Tests for the 5-step rebalance allocation strategy."""

    @staticmethod
    def _make_engine(**kw):
        return BacktestEngine(top_n=kw.pop("top_n", 3), **kw)

    # -- helper: call _rebalance directly with controlled inputs -----------

    def test_limit_up_target_can_sell_excess(self):
        """Limit-up NeedToAdjust stock that is overweight should be sold down."""
        engine = self._make_engine()
        # Stock A at limit-up (open = prev_close * 1.1), currently overweight
        open_prices = pd.Series({"A": 11.0, "B": 10.0, "C": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6, "C": 1e6})
        holdings = {"A": 5000}  # overweight A
        target = ["A", "B", "C"]  # equal-weight target
        cash = 50_000.0

        new_cash, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # A is limit-up but overweight → should sell excess (can sell at limit-up)
        assert new_h["A"] < 5000, "Should sell excess of limit-up overweight stock"
        # B and C should be bought
        assert "B" in new_h and new_h["B"] > 0
        assert "C" in new_h and new_h["C"] > 0

    def test_limit_up_target_cannot_buy_more(self):
        """Limit-up NeedToAdjust stock that is underweight cannot buy more."""
        engine = self._make_engine()
        open_prices = pd.Series({"A": 11.0, "B": 10.0, "C": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6, "C": 1e6})
        # A underweight (only 100 shares, target ≈ per_stock/11 ≈ much more)
        holdings = {"A": 100}
        target = ["A", "B", "C"]
        cash = 100_000.0

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # A is limit-up and underweight → cannot buy more, stays at 100
        assert new_h["A"] == 100

    def test_limit_up_needtobuy_excluded(self):
        """Limit-up NeedToBuy stock should not enter TargetSet at all."""
        engine = self._make_engine()
        # D at limit-up, not currently held → NeedToBuy → excluded from CanBuy
        open_prices = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0, "D": 11.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6, "C": 1e6, "D": 1e6})
        holdings = {}
        target = ["D", "A", "B", "C"]
        cash = 300_000.0

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # D should NOT be bought (limit-up NeedToBuy)
        assert "D" not in new_h or new_h.get("D", 0) == 0
        # A, B, C should be bought with allocation over 3 stocks (not 4)
        assert new_h.get("A", 0) > 0
        assert new_h.get("B", 0) > 0
        assert new_h.get("C", 0) > 0

    def test_suspended_needtoadjust_excluded_from_allocation(self):
        """Suspended NeedToAdjust excluded from both allocatable and TargetSet."""
        engine = self._make_engine()
        # A is suspended (volume=0), held and in target
        open_prices = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        volume = pd.Series({"A": 0, "B": 1e6, "C": 1e6})
        holdings = {"A": 1000}  # held, suspended
        target = ["A", "B", "C"]
        cash = 100_000.0

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # A stays at 1000 (can't trade)
        assert new_h["A"] == 1000
        # B and C allocated from (cash only, A's value excluded)
        # per_stock = (100_000 * 0.99) / 2 = 49_500
        # Each gets ~49_500/10 = 4950 → 4900 shares (round to 100)
        assert new_h["B"] > 0
        assert new_h["C"] > 0

    def test_stuck_needtosell_excluded_from_total(self):
        """NeedToSell stocks that can't be sold are excluded from allocatable."""
        engine = self._make_engine()
        # X is limit-down, not in target → NeedToSell but stuck
        open_prices = pd.Series({"X": 9.0, "A": 10.0, "B": 10.0})
        prev_close = pd.Series({"X": 10.0, "A": 10.0, "B": 10.0})
        volume = pd.Series({"X": 1e6, "A": 1e6, "B": 1e6})
        holdings = {"X": 1000}  # limit-down, can't sell
        target = ["A", "B"]
        cash = 50_000.0

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # X stays (stuck)
        assert new_h["X"] == 1000
        # Allocatable = cash only (50k), NOT cash + X's value
        # per_stock = 50_000 * 0.99 / 2 = 24_750
        # A gets ~24_750/10 = 2475 → 2400 shares
        assert new_h["A"] > 0 and new_h["A"] <= 2500
        assert new_h["B"] > 0 and new_h["B"] <= 2500

    def test_no_over_allocation_with_frozen_target(self):
        """No over-allocation when a target stock is frozen (limit-up held)."""
        engine = self._make_engine()
        # A is limit-up, held, in target, worth 50k. B and C are normal.
        # Old bug: total includes A's 50k, denominator excludes A
        #   → per_stock = 150k/2 = 75k → needs 150k but only 100k free → over-allocate
        # New: A in denominator → per_stock = 150k/3 = 50k → no overflow
        open_prices = pd.Series({"A": 11.0, "B": 10.0, "C": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6, "C": 1e6})
        holdings = {"A": 5000}  # A worth 55k at open, limit-up
        target = ["A", "B", "C"]
        cash = 100_000.0

        new_cash, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # Should NOT go negative on cash
        assert new_cash >= 0, f"Cash went negative: {new_cash}"
        # All three should have positions
        total_mv = sum(
            new_h[s] * open_prices[s] for s in new_h if s in open_prices.index
        )
        assert total_mv + new_cash <= 155_000 * 1.01  # no over-allocation

    def test_partial_fill_on_cash_shortage(self):
        """When cash is low, buy as many lots as affordable."""
        engine = self._make_engine(top_n=2)
        open_prices = pd.Series({"A": 10.0, "B": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6})
        holdings = {}
        target = ["A", "B"]
        cash = 15_000.0  # enough for ~1500 shares total, ~750 each

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # A (higher score) should get full allocation: ~7425/10 = 742 → 700
        assert new_h.get("A", 0) == 700
        # B gets whatever is left (partial fill)
        assert new_h.get("B", 0) > 0  # should get some shares

    def test_limit_down_can_buy(self):
        """NeedToBuy stock at limit-down CAN be bought."""
        engine = self._make_engine(top_n=2)
        open_prices = pd.Series({"A": 10.0, "B": 9.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6})
        holdings = {}
        target = ["A", "B"]
        cash = 100_000.0

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # B is at limit-down but can be bought
        assert new_h.get("B", 0) > 0

    def test_needtosell_limit_up_can_sell(self):
        """NeedToSell stock at limit-up should be sold (can sell at limit-up)."""
        engine = self._make_engine(top_n=1)
        open_prices = pd.Series({"X": 11.0, "A": 10.0})
        prev_close = pd.Series({"X": 10.0, "A": 10.0})
        volume = pd.Series({"X": 1e6, "A": 1e6})
        holdings = {"X": 1000}
        target = ["A"]
        cash = 0.0

        new_cash, new_h, log = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        # X at limit-up, not in target → should be sold
        assert "X" not in new_h
        assert log is not None and "X" in log["sells"]


class TestSTStockHandling:
    """Tests for ST stock price limits, buy filtering, and force-sell."""

    @staticmethod
    def _make_engine(**kw):
        return BacktestEngine(top_n=kw.pop("top_n", 3), **kw)

    def test_st_price_limit_5pct(self):
        """Main-board ST stock uses ±5% limit instead of ±10%."""
        engine = self._make_engine()
        # Stock at 5% above prev_close → limit-up for ST, not for normal
        assert engine._is_at_limit(10.50, 10.0, "up", "600000", is_st=True)
        assert not engine._is_at_limit(10.50, 10.0, "up", "600000", is_st=False)

    def test_st_limit_down_5pct(self):
        """Main-board ST stock at -5% is limit-down."""
        engine = self._make_engine()
        assert engine._is_at_limit(9.50, 10.0, "down", "600000", is_st=True)
        assert not engine._is_at_limit(9.50, 10.0, "down", "600000", is_st=False)

    def test_gem_st_still_20pct(self):
        """GEM (300xxx) ST stock still uses ±20% limit."""
        engine = self._make_engine()
        # 10% above prev_close → NOT limit-up for GEM (needs 20%)
        assert not engine._is_at_limit(11.0, 10.0, "up", "300001", is_st=True)
        # 20% above → limit-up
        assert engine._is_at_limit(12.0, 10.0, "up", "300001", is_st=True)

    def test_st_excluded_from_buy(self):
        """ST stocks in NeedToBuy are excluded from the buy pool."""
        engine = self._make_engine(top_n=3)
        open_prices = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6, "C": 1e6})
        is_st = pd.Series({"A": 0, "B": 1, "C": 0})
        holdings = {}
        target = ["A", "B", "C"]
        cash = 300_000.0

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume, is_st,
        )
        assert "B" not in new_h, "ST stock should not be bought"
        assert new_h.get("A", 0) > 0
        assert new_h.get("C", 0) > 0

    def test_held_st_force_sold(self):
        """Held stock that becomes ST is force-classified as NeedToSell."""
        engine = self._make_engine(top_n=3)
        open_prices = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6, "C": 1e6})
        is_st = pd.Series({"A": 0, "B": 1, "C": 0})
        holdings = {"B": 1000}  # B is held but now ST
        target = ["A", "B", "C"]  # B still in target by score
        cash = 200_000.0

        _, new_h, log = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume, is_st,
        )
        assert "B" not in new_h, "ST stock should be force-sold"
        assert log is not None and "B" in log["sells"]

    def test_no_is_st_backward_compat(self):
        """Without is_st data, behavior is unchanged (no filtering)."""
        engine = self._make_engine(top_n=2)
        open_prices = pd.Series({"A": 10.0, "B": 10.0})
        prev_close = pd.Series({"A": 10.0, "B": 10.0})
        volume = pd.Series({"A": 1e6, "B": 1e6})
        holdings = {}
        target = ["A", "B"]
        cash = 200_000.0

        _, new_h, _ = engine._rebalance(
            pd.Timestamp("2024-01-08"), cash, holdings, target,
            open_prices, prev_close, volume,
        )
        assert new_h.get("A", 0) > 0
        assert new_h.get("B", 0) > 0
