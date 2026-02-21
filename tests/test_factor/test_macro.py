"""Tests for MacroFactorEngine — M3.3."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pangu.factor.macro import MacroFactorEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(overrides: dict | None = None, exclude: set | None = None) -> pd.DataFrame:
    """Build a full global snapshot DataFrame."""
    exclude = exclude or set()
    rows = [
        {"symbol": "SPX", "name": "S&P 500", "close": 5000, "change_pct": 1.0, "source": "us_index"},
        {"symbol": "DJI", "name": "Dow Jones", "close": 40000, "change_pct": 0.5, "source": "us_index"},
        {"symbol": "IXIC", "name": "NASDAQ", "close": 16000, "change_pct": 1.5, "source": "us_index"},
        {"symbol": "HSI", "name": "恒生指数", "close": 26000, "change_pct": -0.8, "source": "hk_index"},
        {"symbol": "HSTECH", "name": "恒生科技指数", "close": 5000, "change_pct": -1.2, "source": "hk_index"},
        {"symbol": "VHSI", "name": "恒指波幅指数", "close": 22.5, "change_pct": 3.0, "source": "hk_index"},
        {"symbol": "GC", "name": "COMEX黄金", "close": 4900, "change_pct": 0.8, "source": "commodity"},
        {"symbol": "SI", "name": "COMEX白银", "close": 75, "change_pct": 2.5, "source": "commodity"},
        {"symbol": "CL", "name": "WTI原油", "close": 62, "change_pct": -0.5, "source": "commodity"},
        {"symbol": "HG", "name": "LME铜", "close": 570, "change_pct": 1.2, "source": "commodity"},
        {"symbol": "FEF", "name": "铁矿石", "close": 96, "change_pct": -0.3, "source": "commodity"},
        {"symbol": "NG", "name": "NYMEX天然气", "close": 3.0, "change_pct": -2.0, "source": "commodity"},
        {"symbol": "CT", "name": "NYBOT棉花", "close": 64, "change_pct": 0.1, "source": "commodity"},
    ]
    rows = [r for r in rows if r["symbol"] not in exclude]
    if overrides:
        for r in rows:
            if r["symbol"] in overrides:
                r.update(overrides[r["symbol"]])
    return pd.DataFrame(rows)


@pytest.fixture
def engine() -> MacroFactorEngine:
    return MacroFactorEngine()


@pytest.fixture
def full_snapshot() -> pd.DataFrame:
    return _make_snapshot()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompute:
    def test_all_factors_present(self, engine: MacroFactorEngine, full_snapshot: pd.DataFrame) -> None:
        result = engine.compute(full_snapshot)
        for name in engine.get_factor_names():
            assert name in result, f"Missing factor: {name}"

    def test_us_overnight_weighted(self, engine: MacroFactorEngine, full_snapshot: pd.DataFrame) -> None:
        result = engine.compute(full_snapshot)
        # SPX=1.0*0.4 + DJI=0.5*0.3 + IXIC=1.5*0.3 = 0.4+0.15+0.45 = 1.0
        expected = (1.0 * 0.4 + 0.5 * 0.3 + 1.5 * 0.3) / (0.4 + 0.3 + 0.3)
        assert result["us_overnight"] == pytest.approx(expected)

    def test_single_symbol_factors(self, engine: MacroFactorEngine, full_snapshot: pd.DataFrame) -> None:
        result = engine.compute(full_snapshot)
        assert result["hk_intraday"] == pytest.approx(-0.8)
        assert result["hk_tech"] == pytest.approx(-1.2)
        assert result["gold_chg"] == pytest.approx(0.8)
        assert result["silver_chg"] == pytest.approx(2.5)
        assert result["oil_chg"] == pytest.approx(-0.5)
        assert result["copper_chg"] == pytest.approx(1.2)
        assert result["iron_chg"] == pytest.approx(-0.3)
        assert result["ng_chg"] == pytest.approx(-2.0)
        assert result["cotton_chg"] == pytest.approx(0.1)

    def test_vhsi_uses_close_not_change(self, engine: MacroFactorEngine, full_snapshot: pd.DataFrame) -> None:
        result = engine.compute(full_snapshot)
        assert result["vhsi"] == pytest.approx(22.5)  # close, not change_pct=3.0

    def test_global_risk_computed(self, engine: MacroFactorEngine, full_snapshot: pd.DataFrame) -> None:
        result = engine.compute(full_snapshot)
        assert not math.isnan(result["global_risk"])

    def test_global_risk_formula(self, engine: MacroFactorEngine, full_snapshot: pd.DataFrame) -> None:
        result = engine.compute(full_snapshot)
        # Uses VHSI change_pct (3.0) not close (22.5) for scale consistency
        # us=1.0*0.3 + hk=-0.8*0.2 + gold=0.8*(-0.15) + oil=-0.5*0.1 + copper=1.2*0.1 + vhsi_chg=3.0*(-0.15)
        # = 0.3 + (-0.16) + (-0.12) + (-0.05) + 0.12 + (-0.45) = -0.36
        # total_w = 0.3+0.2+0.15+0.1+0.1+0.15 = 1.0
        us = 1.0
        vhsi_chg = 3.0  # change_pct, not close=22.5
        weighted = us * 0.3 + (-0.8) * 0.2 + 0.8 * (-0.15) + (-0.5) * 0.1 + 1.2 * 0.1 + vhsi_chg * (-0.15)
        total_w = 0.3 + 0.2 + 0.15 + 0.1 + 0.1 + 0.15
        expected = weighted / total_w
        assert result["global_risk"] == pytest.approx(expected)


class TestPartialData:
    def test_missing_us_indices(self, engine: MacroFactorEngine) -> None:
        snapshot = _make_snapshot(exclude={"SPX", "DJI", "IXIC"})
        result = engine.compute(snapshot)
        assert math.isnan(result["us_overnight"])
        # Other factors still work
        assert result["hk_intraday"] == pytest.approx(-0.8)
        assert result["gold_chg"] == pytest.approx(0.8)

    def test_missing_some_commodities(self, engine: MacroFactorEngine) -> None:
        snapshot = _make_snapshot(exclude={"NG", "CT", "SI"})
        result = engine.compute(snapshot)
        assert math.isnan(result["ng_chg"])
        assert math.isnan(result["cotton_chg"])
        assert math.isnan(result["silver_chg"])
        # Others still work
        assert result["gold_chg"] == pytest.approx(0.8)

    def test_missing_vhsi(self, engine: MacroFactorEngine) -> None:
        snapshot = _make_snapshot(exclude={"VHSI"})
        result = engine.compute(snapshot)
        assert math.isnan(result["vhsi"])
        # global_risk still computable (VHSI skipped)
        assert not math.isnan(result["global_risk"])

    def test_partial_us_indices(self, engine: MacroFactorEngine) -> None:
        snapshot = _make_snapshot(exclude={"DJI"})
        result = engine.compute(snapshot)
        # Only SPX(0.4) and IXIC(0.3) → weighted / sum_w
        expected = (1.0 * 0.4 + 1.5 * 0.3) / (0.4 + 0.3)
        assert result["us_overnight"] == pytest.approx(expected)


class TestEmptyData:
    def test_empty_snapshot(self, engine: MacroFactorEngine) -> None:
        empty = pd.DataFrame(columns=["symbol", "change_pct", "close", "source"])
        result = engine.compute(empty)
        assert all(math.isnan(v) for v in result.values())

    def test_none_snapshot(self, engine: MacroFactorEngine) -> None:
        result = engine.compute(None)
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == len(engine.get_factor_names())


class TestGetFactorNames:
    def test_returns_twelve_factors(self, engine: MacroFactorEngine) -> None:
        names = engine.get_factor_names()
        assert len(names) == 12

    def test_returns_copy(self, engine: MacroFactorEngine) -> None:
        names = engine.get_factor_names()
        names.append("extra")
        assert "extra" not in engine.get_factor_names()
