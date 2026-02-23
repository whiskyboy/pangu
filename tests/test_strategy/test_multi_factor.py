"""Tests for MultiFactorStrategy — M3.4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.models import Action, SignalStatus
from pangu.strategy.factor.multi_factor import (
    MultiFactorStrategy,
    _minmax_normalize,
    _weighted_score,
    _zscore_normalize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tech_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Generate bars with fake technical factor columns."""
    rng = np.random.default_rng(seed)
    close = 50 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d").tolist(),
        "open": close + rng.normal(0, 0.2, n),
        "high": close + rng.uniform(0.1, 1.0, n),
        "low": close - rng.uniform(0.1, 1.0, n),
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n),
        "rsi_14": rng.normal(50, 10, n),
        "macd_hist": rng.normal(0, 1, n),
        "bias_20": rng.normal(0, 0.02, n),
        "obv": rng.normal(5e7, 1e7, n),
        "atr_14": rng.uniform(0.5, 2.0, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n),
    })
    return df


def _make_fund_df() -> pd.DataFrame:
    return pd.DataFrame({
        "pe_ttm": [15.0, 25.0, 8.0],
        "pb": [3.0, 1.5, 0.8],
        "roe_ttm": [20.0, 12.0, -5.0],
    }, index=pd.Index(["A", "B", "C"], name="symbol"))


_MACRO_FACTORS = {
    "gold_chg": 1.0, "silver_chg": 2.0, "oil_chg": -0.5,
    "copper_chg": 1.2, "iron_chg": -0.3, "ng_chg": -2.0,
    "cotton_chg": 0.1, "us_overnight": 0.5, "hk_intraday": -0.8,
    "hk_tech": -1.2, "vhsi": 22.0, "global_risk": 0.3,
}


def _build_inputs():
    """Build standard test inputs for 3 stocks."""
    tech_df = {
        "A": _make_tech_df(60, 42),
        "B": _make_tech_df(60, 43),
        "C": _make_tech_df(60, 44),
    }
    fund_df = _make_fund_df()
    return tech_df, fund_df, dict(_MACRO_FACTORS)


@pytest.fixture
def strategy():
    return MultiFactorStrategy(
        top_n=2,
        buy_threshold=0.5,
        sell_threshold=0.2,
    )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


class TestZscoreNormalize:
    def test_mean_zero_std_one(self) -> None:
        df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [100.0, 200.0, 300.0]})
        result = _zscore_normalize(df)
        assert pytest.approx(result["a"].mean(), abs=1e-10) == 0
        assert result["a"].std(ddof=0) > 0.5

    def test_single_row_unchanged(self) -> None:
        df = pd.DataFrame({"a": [10.0]})
        result = _zscore_normalize(df)
        assert result["a"].iloc[0] == 10.0

    def test_nan_preserved(self) -> None:
        df = pd.DataFrame({"a": [10.0, float("nan"), 30.0]})
        result = _zscore_normalize(df)
        assert pd.isna(result["a"].iloc[1])

    def test_constant_column(self) -> None:
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        result = _zscore_normalize(df)
        assert (result["a"] == 0.0).all()


class TestWeightedScore:
    def test_weighted_sum(self) -> None:
        df = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        weights = {"a": 0.6, "b": 0.4}
        scores = _weighted_score(df, weights)
        assert pytest.approx(scores.iloc[0]) == 0.6
        assert pytest.approx(scores.iloc[1]) == 0.4

    def test_missing_column_skipped(self) -> None:
        df = pd.DataFrame({"a": [1.0]})
        weights = {"a": 0.5, "missing": 0.5}
        scores = _weighted_score(df, weights)
        assert pytest.approx(scores.iloc[0]) == 1.0

    def test_negative_weights(self) -> None:
        """Negative weight: higher factor value → lower score (e.g. PE_TTM)."""
        df = pd.DataFrame({"val": [1.0, -1.0]}, index=["cheap", "expensive"])
        weights = {"val": -0.10}
        scores = _weighted_score(df, weights)
        assert scores["cheap"] < scores["expensive"]

    def test_nan_excluded_from_weight(self) -> None:
        """NaN factors should not count toward total_w for that stock."""
        df = pd.DataFrame({
            "a": [1.0, 1.0],
            "b": [1.0, float("nan")],
        }, index=["full", "partial"])
        weights = {"a": 0.5, "b": 0.5}
        s = _weighted_score(df, weights)
        # "full" uses both factors; "partial" uses only "a"
        # full: (1*0.5 + 1*0.5) / (0.5+0.5) = 1.0
        # partial: (1*0.5 + 0*0.5) / (0.5) = 1.0  (NaN weight excluded)
        assert pytest.approx(s["full"]) == 1.0
        assert pytest.approx(s["partial"]) == 1.0

    def test_all_nan_returns_zero(self) -> None:
        """Stock with all-NaN factors should score 0."""
        df = pd.DataFrame({"a": [float("nan")]})
        weights = {"a": 0.5}
        s = _weighted_score(df, weights)
        assert pytest.approx(s.iloc[0]) == 0.0


class TestMinmaxNormalize:
    def test_range_zero_one(self) -> None:
        s = pd.Series([10.0, 20.0, 30.0])
        result = _minmax_normalize(s)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_constant_returns_half(self) -> None:
        s = pd.Series([5.0, 5.0, 5.0])
        result = _minmax_normalize(s)
        assert (result == 0.5).all()

    def test_single_element_returns_half(self) -> None:
        s = pd.Series([42.0])
        result = _minmax_normalize(s)
        assert result.iloc[0] == 0.5

    def test_winsorize_reduces_outlier_impact(self) -> None:
        """An extreme outlier should not compress normal scores to near zero."""
        normal = list(np.linspace(0.1, 0.4, 100))
        outlier = [5.0]  # extreme high
        s = pd.Series(normal + outlier)
        result = _minmax_normalize(s)
        # Without winsorize: max normal ≈ 0.4/5.0 ≈ 0.08
        # With winsorize: outlier clipped, normal scores spread across [0, 1]
        normal_scores = result.iloc[:100]
        assert normal_scores.max() > 0.5, "Normal scores should not be compressed"


# ---------------------------------------------------------------------------
# MultiFactorStrategy
# ---------------------------------------------------------------------------


class TestGenerateSignals:
    def test_produces_signals(self, strategy: MultiFactorStrategy) -> None:
        tech_df, fund_df, macro = _build_inputs()
        _pool_df, signals = strategy.generate_signals(tech_df, fund_df, macro)
        assert len(signals) > 0

    def test_signal_fields(self, strategy: MultiFactorStrategy) -> None:
        tech_df, fund_df, macro = _build_inputs()
        _pool_df, signals = strategy.generate_signals(tech_df, fund_df, macro)
        for s in signals:
            assert s.source == "factor"
            assert 0 <= s.confidence <= 1
            assert s.factor_score is not None

    def test_buy_signals_in_top_n(self, strategy: MultiFactorStrategy) -> None:
        tech_df, fund_df, macro = _build_inputs()
        _pool_df, signals = strategy.generate_signals(tech_df, fund_df, macro)
        buys = [s for s in signals if s.action == Action.BUY]
        assert len(buys) <= 2

    def test_empty_pool(self, strategy: MultiFactorStrategy) -> None:
        pool_df, signals = strategy.generate_signals({}, pd.DataFrame(), {})
        assert signals == []
        assert pool_df.empty

    def test_stop_loss_take_profit_on_buy(self, strategy: MultiFactorStrategy) -> None:
        tech_df, fund_df, macro = _build_inputs()
        _pool_df, signals = strategy.generate_signals(tech_df, fund_df, macro)
        buys = [s for s in signals if s.action == Action.BUY]
        for s in buys:
            if s.stop_loss is not None:
                assert s.stop_loss < s.price
                assert s.take_profit > s.price

    def test_returns_factor_pool(self, strategy: MultiFactorStrategy) -> None:
        tech_df, fund_df, macro = _build_inputs()
        pool_df, _signals = strategy.generate_signals(tech_df, fund_df, macro)
        assert len(pool_df) == 3
        assert "score" in pool_df.columns
        assert "rank" in pool_df.columns
        assert "symbol" in pool_df.columns


class TestSignalStatusTracking:
    def test_new_entry_then_sustained(self, strategy: MultiFactorStrategy) -> None:
        tech_df, fund_df, macro = _build_inputs()

        # First run: all new_entry (no prev_pool)
        pool_df1, signals1 = strategy.generate_signals(tech_df, fund_df, macro)
        buys1 = [s for s in signals1 if s.action == Action.BUY]
        for s in buys1:
            assert s.signal_status == SignalStatus.NEW_ENTRY

        # Second run: pass first pool as prev_pool -> sustained
        _pool_df2, signals2 = strategy.generate_signals(
            tech_df, fund_df, macro, prev_pool=pool_df1
        )
        buys2 = [s for s in signals2 if s.action == Action.BUY]
        for s in buys2:
            if s.symbol in {b.symbol for b in buys1}:
                assert s.signal_status == SignalStatus.SUSTAINED


class TestGlobalRiskDampen:
    def test_high_risk_raises_threshold(self) -> None:
        tech_df, fund_df, _ = _build_inputs()
        macro = {k: 0.0 for k in _MACRO_FACTORS}
        macro["global_risk"] = -2.0

        strategy = MultiFactorStrategy(
            top_n=2, buy_threshold=0.95, sell_threshold=0.2,
            risk_dampen_threshold=-1.0,
        )
        _pool_df, signals = strategy.generate_signals(tech_df, fund_df, macro)
        buys = [s for s in signals if s.action == Action.BUY]
        assert len(buys) == 0
