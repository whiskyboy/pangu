"""Tests for Alpha158Engine — Step 3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.factor.alpha158 import (
    FACTOR_NAMES,
    Alpha158Engine,
    _compute_kbar,
    _compute_price,
    _compute_rolling_complex,
    _compute_rolling_regression,
    _compute_rolling_simple,
    _prepare_wide_tables,
    _rolling_idxmax,
    _rolling_idxmin,
    _rolling_rank,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_multi_stock_bars(n_days: int = 120, n_stocks: int = 5) -> pd.DataFrame:
    """Generate deterministic multi-stock OHLCV data in long format."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = [f"00000{i}" for i in range(n_stocks)]

    rows = []
    for sym in symbols:
        base = 20.0 + rng.uniform(0, 30)
        closes = base + np.cumsum(rng.normal(0, 0.3, n_days))
        closes = np.maximum(closes, 1.0)  # ensure positive
        opens = closes * (1 + rng.normal(0, 0.005, n_days))
        highs = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.02, n_days))
        lows = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.02, n_days))
        volumes = rng.integers(500_000, 50_000_000, n_days).astype(float)
        amounts = volumes * closes * (1 + rng.normal(0, 0.01, n_days))
        adj_factors = np.ones(n_days)

        for i in range(n_days):
            rows.append({
                "symbol": sym,
                "date": dates[i].strftime("%Y-%m-%d"),
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "volume": volumes[i],
                "amount": amounts[i],
                "adj_factor": adj_factors[i],
            })

    return pd.DataFrame(rows)


def _make_fundamentals(bars: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic fundamental data matching bars."""
    rng = np.random.default_rng(99)
    symbols = bars["symbol"].unique()
    dates = sorted(bars["date"].unique())

    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({
                "symbol": sym,
                "date": d,
                "pe_ttm": rng.uniform(5, 50),
                "pb": rng.uniform(0.5, 5),
                "roe_ttm": rng.uniform(0.02, 0.25),
                "revenue_yoy": rng.uniform(-0.3, 0.5),
                "profit_yoy": rng.uniform(-0.5, 1.0),
                "market_cap": rng.uniform(1e9, 1e11),
                "gross_margin": rng.uniform(0.1, 0.6),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def bars() -> pd.DataFrame:
    return _make_multi_stock_bars()


@pytest.fixture
def fundamentals(bars: pd.DataFrame) -> pd.DataFrame:
    return _make_fundamentals(bars)


@pytest.fixture
def engine() -> Alpha158Engine:
    return Alpha158Engine()


@pytest.fixture
def wide_tables(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return _prepare_wide_tables(bars)


# ---------------------------------------------------------------------------
# Factor names
# ---------------------------------------------------------------------------

class TestFactorNames:
    def test_count(self):
        assert len(FACTOR_NAMES) == 166

    def test_no_duplicates(self):
        assert len(set(FACTOR_NAMES)) == 166

    def test_kbar_first_9(self):
        expected = ["KMID", "KLEN", "KMID2", "KUP", "KUP2",
                     "KLOW", "KLOW2", "KSFT", "KSFT2"]
        assert FACTOR_NAMES[:9] == expected

    def test_price_next_4(self):
        assert FACTOR_NAMES[9:13] == ["OPEN0", "HIGH0", "LOW0", "VWAP0"]

    def test_fundamentals_last_8(self):
        expected = ["PE", "PB", "ROE", "REVENUE_YOY", "PROFIT_YOY",
                     "LN_MKTCAP", "TURNOVER", "GROSS_MARGIN"]
        assert FACTOR_NAMES[-8:] == expected

    def test_get_factor_names_matches(self, engine: Alpha158Engine):
        assert engine.get_factor_names() == FACTOR_NAMES


# ---------------------------------------------------------------------------
# KBar factors
# ---------------------------------------------------------------------------

class TestKBar:
    def test_all_9_returned(self, wide_tables):
        result = _compute_kbar(
            wide_tables["O"], wide_tables["H"],
            wide_tables["L"], wide_tables["C"],
        )
        assert len(result) == 9
        expected_keys = {"KMID", "KLEN", "KMID2", "KUP", "KUP2",
                         "KLOW", "KLOW2", "KSFT", "KSFT2"}
        assert set(result.keys()) == expected_keys

    def test_klen_positive(self, wide_tables):
        """KLEN = (H-L)/O should always be positive (H >= L)."""
        result = _compute_kbar(
            wide_tables["O"], wide_tables["H"],
            wide_tables["L"], wide_tables["C"],
        )
        klen = result["KLEN"]
        assert (klen.dropna() >= 0).all().all()

    def test_kup_klow_non_negative(self, wide_tables):
        """Upper and lower shadows should be non-negative."""
        result = _compute_kbar(
            wide_tables["O"], wide_tables["H"],
            wide_tables["L"], wide_tables["C"],
        )
        assert (result["KUP"].dropna() >= -1e-10).all().all()
        assert (result["KLOW"].dropna() >= -1e-10).all().all()

    def test_ksft2_range(self, wide_tables):
        """KSFT2 = (2C-H-L)/(H-L+eps) should be in [-1, 1]."""
        result = _compute_kbar(
            wide_tables["O"], wide_tables["H"],
            wide_tables["L"], wide_tables["C"],
        )
        ksft2 = result["KSFT2"].dropna()
        assert (ksft2 >= -1.01).all().all()
        assert (ksft2 <= 1.01).all().all()

    def test_known_values(self):
        """Test with a single known candlestick."""
        idx = pd.to_datetime(["2024-01-01"])
        cols = ["A"]
        O = pd.DataFrame([[10.0]], index=idx, columns=cols)  # noqa: E741
        H = pd.DataFrame([[12.0]], index=idx, columns=cols)
        L = pd.DataFrame([[8.0]], index=idx, columns=cols)
        C = pd.DataFrame([[11.0]], index=idx, columns=cols)

        result = _compute_kbar(O, H, L, C)
        assert abs(result["KMID"].iloc[0, 0] - 0.1) < 1e-6    # (11-10)/10
        assert abs(result["KLEN"].iloc[0, 0] - 0.4) < 1e-6    # (12-8)/10
        assert abs(result["KUP"].iloc[0, 0] - 0.1) < 1e-6     # (12-11)/10
        assert abs(result["KLOW"].iloc[0, 0] - 0.2) < 1e-6    # (10-8)/10


# ---------------------------------------------------------------------------
# Price factors
# ---------------------------------------------------------------------------

class TestPrice:
    def test_all_4_returned(self, wide_tables):
        result = _compute_price(
            wide_tables["O"], wide_tables["H"], wide_tables["L"],
            wide_tables["C"], wide_tables["VWAP"],
        )
        assert set(result.keys()) == {"OPEN0", "HIGH0", "LOW0", "VWAP0"}

    def test_high0_ge_low0(self, wide_tables):
        """HIGH0 >= LOW0 always."""
        result = _compute_price(
            wide_tables["O"], wide_tables["H"], wide_tables["L"],
            wide_tables["C"], wide_tables["VWAP"],
        )
        diff = result["HIGH0"] - result["LOW0"]
        assert (diff.dropna() >= -1e-10).all().all()

    def test_close_ratio_near_1(self, wide_tables):
        """OPEN0, HIGH0, LOW0 should be reasonable ratios near 1."""
        result = _compute_price(
            wide_tables["O"], wide_tables["H"], wide_tables["L"],
            wide_tables["C"], wide_tables["VWAP"],
        )
        for name in ["OPEN0", "HIGH0", "LOW0", "VWAP0"]:
            vals = result[name].dropna()
            assert (vals > 0.5).all().all(), f"{name} has values too low"
            assert (vals < 2.0).all().all(), f"{name} has values too high"


# ---------------------------------------------------------------------------
# Rolling simple
# ---------------------------------------------------------------------------

class TestRollingSimple:
    def test_factor_count(self, wide_tables):
        result = _compute_rolling_simple(
            wide_tables["C"], wide_tables["H"], wide_tables["L"], wide_tables["V"],
        )
        # 11 ops × 5 windows = 55
        assert len(result) == 55

    def test_nan_warmup(self, wide_tables):
        """First 59 rows should be NaN for window=60 factors."""
        result = _compute_rolling_simple(
            wide_tables["C"], wide_tables["H"], wide_tables["L"], wide_tables["V"],
        )
        ma60 = result["MA60"]
        # First 58 rows (0-indexed) should be NaN (need 60 values for window=60)
        assert ma60.iloc[:59].isna().all().all()
        # Row 59 (60th) should have values
        assert ma60.iloc[59].notna().any()

    def test_ma_reasonable(self, wide_tables):
        """MA_d / C should be roughly near 1 for stable prices."""
        result = _compute_rolling_simple(
            wide_tables["C"], wide_tables["H"], wide_tables["L"], wide_tables["V"],
        )
        ma5 = result["MA5"].dropna()
        # MA5/C should be close to 1 for prices with small drift
        assert (ma5 > 0.8).all().all()
        assert (ma5 < 1.2).all().all()

    def test_rsv_range(self, wide_tables):
        """RSV should be in [0, 1]."""
        result = _compute_rolling_simple(
            wide_tables["C"], wide_tables["H"], wide_tables["L"], wide_tables["V"],
        )
        for d in [5, 10, 20, 30, 60]:
            rsv = result[f"RSV{d}"].dropna()
            assert (rsv >= -0.01).all().all(), f"RSV{d} has negative values"
            assert (rsv <= 1.01).all().all(), f"RSV{d} exceeds 1"


# ---------------------------------------------------------------------------
# Rolling complex
# ---------------------------------------------------------------------------

class TestRollingComplex:
    def test_factor_count(self, wide_tables):
        result = _compute_rolling_complex(wide_tables["C"], wide_tables["V"])
        # 15 ops × 5 windows = 75
        assert len(result) == 75

    def test_rank_range(self, wide_tables):
        """RANK should be in [0, 1]."""
        result = _compute_rolling_complex(wide_tables["C"], wide_tables["V"])
        for d in [5, 10, 20, 30, 60]:
            rank = result[f"RANK{d}"].dropna()
            assert (rank >= 0).all().all()
            assert (rank <= 1.01).all().all()

    def test_cntp_cntn_sum_le_1(self, wide_tables):
        """CNTP + CNTN should be <= 1 (some days could be flat)."""
        result = _compute_rolling_complex(wide_tables["C"], wide_tables["V"])
        for d in [5, 10]:
            total = result[f"CNTP{d}"] + result[f"CNTN{d}"]
            assert (total.dropna() <= 1.01).all().all()

    def test_cntd_is_cntp_minus_cntn(self, wide_tables):
        result = _compute_rolling_complex(wide_tables["C"], wide_tables["V"])
        for d in [5, 10, 20]:
            diff = result[f"CNTD{d}"] - (result[f"CNTP{d}"] - result[f"CNTN{d}"])
            assert (diff.dropna().abs() < 1e-6).all().all()

    def test_sumd_is_sump_minus_sumn(self, wide_tables):
        result = _compute_rolling_complex(wide_tables["C"], wide_tables["V"])
        for d in [5, 10]:
            diff = result[f"SUMD{d}"] - (result[f"SUMP{d}"] - result[f"SUMN{d}"])
            assert (diff.dropna().abs() < 1e-6).all().all()

    def test_imax_imin_range(self, wide_tables):
        result = _compute_rolling_complex(wide_tables["C"], wide_tables["V"])
        for d in [5, 10, 20]:
            imax = result[f"IMAX{d}"].dropna()
            imin = result[f"IMIN{d}"].dropna()
            assert (imax >= 0).all().all()
            assert (imax <= 1).all().all()
            assert (imin >= 0).all().all()
            assert (imin <= 1).all().all()


# ---------------------------------------------------------------------------
# Rolling regression
# ---------------------------------------------------------------------------

class TestRollingRegression:
    def test_factor_count(self, wide_tables):
        result = _compute_rolling_regression(wide_tables["C"])
        # 3 ops × 5 windows = 15
        assert len(result) == 15

    def test_rsqr_range(self, wide_tables):
        """R² should be in [0, 1] (approximately)."""
        result = _compute_rolling_regression(wide_tables["C"])
        for d in [5, 10, 20]:
            rsqr = result[f"RSQR{d}"].dropna()
            assert (rsqr >= -0.05).all().all(), f"RSQR{d} negative"
            assert (rsqr <= 1.05).all().all(), f"RSQR{d} exceeds 1"

    def test_perfect_linear(self):
        """For perfectly linear data, R²≈1 and RESI≈0."""
        n = 30
        idx = pd.to_datetime(pd.bdate_range("2024-01-01", periods=n))
        vals = np.linspace(10, 20, n)
        C = pd.DataFrame({"A": vals}, index=idx)

        result = _compute_rolling_regression(C)
        rsqr10 = result["RSQR10"].dropna()
        resi10 = result["RESI10"].dropna()

        assert (rsqr10 > 0.99).all().all(), "R² should be ~1 for linear data"
        assert (resi10.abs() < 0.01).all().all(), "Residuals should be ~0"

    def test_beta_sign_for_uptrend(self):
        """Upward trend should have positive slope (BETA/C > 0)."""
        n = 30
        idx = pd.to_datetime(pd.bdate_range("2024-01-01", periods=n))
        vals = np.linspace(10, 20, n)
        C = pd.DataFrame({"A": vals}, index=idx)

        result = _compute_rolling_regression(C)
        beta10 = result["BETA10"].dropna()
        assert (beta10 > 0).all().all()


# ---------------------------------------------------------------------------
# RANK / IMAX / IMIN helpers
# ---------------------------------------------------------------------------

class TestRollingHelpers:
    def test_rolling_rank_known(self):
        """Rank of max value in window should be 1.0."""
        s = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = _rolling_rank(s, 5)
        # For monotonically increasing, rank of last value = 1.0
        assert abs(result.iloc[4, 0] - 1.0) < 1e-6
        assert abs(result.iloc[9, 0] - 1.0) < 1e-6

    def test_rolling_rank_min(self):
        """Rank of min value should be 1/d."""
        s = pd.DataFrame({"A": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]})
        result = _rolling_rank(s, 5)
        # For monotonically decreasing, rank of last value = 1/5 = 0.2
        assert abs(result.iloc[4, 0] - 0.2) < 1e-6

    def test_rolling_idxmax_recent(self):
        """Max at latest position → (d-1)/d."""
        s = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        result = _rolling_idxmax(s, 5)
        assert abs(result.iloc[4, 0] - 4 / 5) < 1e-6

    def test_rolling_idxmin_recent(self):
        """Min at latest position → (d-1)/d."""
        s = pd.DataFrame({"A": [5, 4, 3, 2, 1]})
        result = _rolling_idxmin(s, 5)
        assert abs(result.iloc[4, 0] - 4 / 5) < 1e-6


# ---------------------------------------------------------------------------
# Full compute pipeline
# ---------------------------------------------------------------------------

class TestFullCompute:
    def test_output_shape(self, bars, fundamentals, engine):
        panel = engine.compute(bars, fundamentals)
        assert panel.columns.tolist() == FACTOR_NAMES
        assert panel.index.names == ["date", "symbol"]

    def test_output_dtype(self, bars, fundamentals, engine):
        panel = engine.compute(bars, fundamentals)
        for col in panel.columns:
            assert panel[col].dtype == np.float32, f"{col} is {panel[col].dtype}"

    def test_no_all_nan_columns(self, bars, fundamentals, engine):
        """After warmup, no column should be entirely NaN."""
        panel = engine.compute(bars, fundamentals)
        # Filter to dates after warmup (last 30 days should be fully populated)
        dates = panel.index.get_level_values("date").unique()
        late_dates = dates[-30:]
        late = panel.loc[late_dates]
        for col in FACTOR_NAMES:
            assert late[col].notna().any(), f"{col} is all NaN after warmup"

    def test_determinism(self, bars, fundamentals, engine):
        """Same input → identical output."""
        panel1 = engine.compute(bars, fundamentals)
        panel2 = engine.compute(bars, fundamentals)
        pd.testing.assert_frame_equal(panel1, panel2)


# ---------------------------------------------------------------------------
# Forward-adjusted prices
# ---------------------------------------------------------------------------

class TestAdjFactor:
    def test_split_continuity(self, engine):
        """Stock split: adj_factor < 1 for pre-split, adjusted prices continuous."""
        n = 10
        dates = pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d").tolist()
        # Simulate 2:1 split at day 5
        # Forward adj: pre-split adj_factor = 0.5 (halve old prices to match new scale)
        closes = [100.0] * 5 + [50.0] * 5
        adj_factors = [0.5] * 5 + [1.0] * 5  # forward adjustment
        opens = [c * 1.01 for c in closes]
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]
        volumes = [1_000_000] * n
        amounts = [c * v for c, v in zip(closes, volumes)]

        bars = pd.DataFrame({
            "symbol": ["A"] * n,
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "amount": amounts,
            "adj_factor": adj_factors,
        })

        wide = _prepare_wide_tables(bars)
        adj_close = wide["C"]
        # Adjusted close should be ~50 throughout (100*0.5=50, 50*1.0=50)
        assert (adj_close.dropna() > 40).all().all()
        assert (adj_close.dropna() < 60).all().all()
        # No big jump at split point
        diff_pct = adj_close.pct_change().dropna().abs()
        assert (diff_pct < 0.05).all().all()
