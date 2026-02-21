"""Tests for PandasTAFactorEngine — M3.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.factor.technical import (
    PandasTAFactorEngine,
    _compute_bias,
    _compute_ma_alignment_score,
    _compute_volume_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bars(n: int = 100) -> pd.DataFrame:
    """Generate deterministic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    base = 50.0
    closes = base + np.cumsum(rng.normal(0, 0.5, n))
    opens = closes + rng.normal(0, 0.2, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.1, 1.0, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.1, 1.0, n)
    volumes = rng.integers(1_000_000, 10_000_000, n)
    dates = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d").tolist()

    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture
def engine() -> PandasTAFactorEngine:
    return PandasTAFactorEngine()


@pytest.fixture
def bars_100() -> pd.DataFrame:
    return _make_bars(100)


@pytest.fixture
def bars_5() -> pd.DataFrame:
    return _make_bars(5)


# ---------------------------------------------------------------------------
# compute() — full pipeline
# ---------------------------------------------------------------------------


class TestCompute:
    def test_returns_all_factor_columns(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        for name in engine.get_factor_names():
            assert name in result.columns, f"Missing factor column: {name}"

    def test_preserves_original_columns(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        for col in ("date", "open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_does_not_mutate_input(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        original_cols = list(bars_100.columns)
        engine.compute(bars_100)
        assert list(bars_100.columns) == original_cols

    def test_row_count_unchanged(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        assert len(result) == len(bars_100)

    def test_empty_dataframe(self, engine: PandasTAFactorEngine) -> None:
        empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        result = engine.compute(empty)
        assert result.empty

    def test_factor_names_match_columns(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        names = set(engine.get_factor_names())
        computed = set(result.columns) - {"date", "open", "high", "low", "close", "volume", "amount", "adj_factor"}
        assert names == computed


# ---------------------------------------------------------------------------
# NaN handling — short data
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_ma60_nan_with_short_data(self, engine: PandasTAFactorEngine, bars_5: pd.DataFrame) -> None:
        result = engine.compute(bars_5)
        assert result["ma60"].isna().all()

    def test_ma5_computed_on_5_rows(self, engine: PandasTAFactorEngine, bars_5: pd.DataFrame) -> None:
        result = engine.compute(bars_5)
        # MA5 should have at least the last row non-NaN with 5 data points
        assert result["ma5"].notna().any()

    def test_rsi_nan_with_short_data(self, engine: PandasTAFactorEngine, bars_5: pd.DataFrame) -> None:
        result = engine.compute(bars_5)
        # RSI(14) needs 14+ rows
        assert result["rsi_14"].isna().all()

    def test_atr_nan_with_short_data(self, engine: PandasTAFactorEngine, bars_5: pd.DataFrame) -> None:
        result = engine.compute(bars_5)
        assert result["atr_14"].isna().all()


# ---------------------------------------------------------------------------
# Trend factors
# ---------------------------------------------------------------------------


class TestTrend:
    def test_ma5_values(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        # Manual check: MA5 at row 4 should equal mean of close[0:5]
        expected = bars_100["close"].iloc[:5].mean()
        assert pytest.approx(result["ma5"].iloc[4], rel=1e-6) == expected

    def test_ema_responds_to_trend(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        # EMA12 should have values after warmup
        assert result["ema12"].iloc[-1] is not None
        assert not np.isnan(result["ema12"].iloc[-1])


# ---------------------------------------------------------------------------
# Momentum factors
# ---------------------------------------------------------------------------


class TestMomentum:
    def test_rsi_range(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        valid_rsi = result["rsi_14"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd_columns(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        assert result["macd"].notna().any()
        assert result["macd_signal"].notna().any()
        assert result["macd_hist"].notna().any()

    def test_roc_values(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        valid_roc = result["roc_10"].dropna()
        assert len(valid_roc) > 0


# ---------------------------------------------------------------------------
# Volatility factors
# ---------------------------------------------------------------------------


class TestVolatility:
    def test_atr_positive(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        valid_atr = result["atr_14"].dropna()
        assert (valid_atr > 0).all()

    def test_bbands_order(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        valid = result.dropna(subset=["bb_upper", "bb_mid", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    def test_hv_positive(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        valid_hv = result["hv_20"].dropna()
        assert (valid_hv >= 0).all()


# ---------------------------------------------------------------------------
# Volume factors
# ---------------------------------------------------------------------------


class TestVolume:
    def test_obv_computed(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        assert result["obv"].notna().any()

    def test_vwap_computed(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        assert result["vwap"].notna().any()

    def test_volume_ratio_computed(self, engine: PandasTAFactorEngine, bars_100: pd.DataFrame) -> None:
        result = engine.compute(bars_100)
        valid_vr = result["volume_ratio"].dropna()
        assert len(valid_vr) > 0
        assert (valid_vr > 0).all()


# ---------------------------------------------------------------------------
# Custom factors
# ---------------------------------------------------------------------------


class TestBias:
    def test_bias_zero_when_at_ma(self) -> None:
        close = pd.Series([10.0] * 25)
        bias = _compute_bias(close, period=20)
        valid = bias.dropna()
        assert pytest.approx(valid.iloc[-1], abs=1e-10) == 0.0

    def test_bias_positive_above_ma(self) -> None:
        close = pd.Series([10.0] * 20 + [15.0])
        bias = _compute_bias(close, period=20)
        assert bias.iloc[-1] > 0

    def test_bias_negative_below_ma(self) -> None:
        close = pd.Series([10.0] * 20 + [5.0])
        bias = _compute_bias(close, period=20)
        assert bias.iloc[-1] < 0


class TestMAAlignment:
    def test_perfect_bull_alignment(self) -> None:
        df = pd.DataFrame({
            "ma5": [50.0],
            "ma10": [40.0],
            "ma20": [30.0],
            "ma60": [20.0],
        })
        score = _compute_ma_alignment_score(df)
        assert score.iloc[0] == 4.0

    def test_perfect_bear_alignment(self) -> None:
        df = pd.DataFrame({
            "ma5": [20.0],
            "ma10": [30.0],
            "ma20": [40.0],
            "ma60": [50.0],
        })
        score = _compute_ma_alignment_score(df)
        assert score.iloc[0] == 0.0

    def test_partial_alignment(self) -> None:
        df = pd.DataFrame({
            "ma5": [50.0],
            "ma10": [40.0],
            "ma20": [45.0],  # MA20 > MA10 breaks chain
            "ma60": [20.0],
        })
        score = _compute_ma_alignment_score(df)
        # MA5>MA10=1, MA10>MA20=0, MA20>MA60=1, MA5>MA60=1 → 3
        assert score.iloc[0] == 3.0

    def test_missing_ma_columns(self) -> None:
        df = pd.DataFrame({"ma5": [50.0]})
        score = _compute_ma_alignment_score(df)
        assert score.iloc[0] == 0.0


class TestVolumeRatio:
    def test_volume_ratio_basic(self) -> None:
        volume = pd.Series([100, 100, 100, 100, 100, 200])
        vr = _compute_volume_ratio(volume, period=5)
        # Last value: 200 / mean([100]*5) = 200/100 = 2.0
        assert pytest.approx(vr.iloc[-1]) == 2.0

    def test_volume_ratio_first_rows_nan(self) -> None:
        volume = pd.Series([100, 200, 300, 400, 500])
        vr = _compute_volume_ratio(volume, period=5)
        # shift(1) means first 5 rows have NaN in rolling mean
        assert vr.iloc[0] != vr.iloc[0]  # NaN check


# ---------------------------------------------------------------------------
# Pseudo-bar
# ---------------------------------------------------------------------------


class TestPseudoBar:
    def test_appends_one_row(self, bars_100: pd.DataFrame) -> None:
        quote = {"open": 55.0, "high": 56.0, "low": 54.0, "price": 55.5, "volume": 5_000_000}
        result = PandasTAFactorEngine.build_pseudo_bar(bars_100, quote)
        assert len(result) == len(bars_100) + 1

    def test_pseudo_bar_close_is_price(self, bars_100: pd.DataFrame) -> None:
        quote = {"open": 55.0, "high": 56.0, "low": 54.0, "price": 55.5, "volume": 5_000_000}
        result = PandasTAFactorEngine.build_pseudo_bar(bars_100, quote)
        assert result.iloc[-1]["close"] == 55.5

    def test_pseudo_bar_with_date(self, bars_100: pd.DataFrame) -> None:
        quote = {"date": "2025-06-01", "open": 55.0, "high": 56.0, "low": 54.0, "price": 55.5, "volume": 5_000_000}
        result = PandasTAFactorEngine.build_pseudo_bar(bars_100, quote)
        assert result.iloc[-1]["date"] == "2025-06-01"
