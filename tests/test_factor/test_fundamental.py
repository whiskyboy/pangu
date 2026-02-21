"""Tests for FundamentalFactorEngine — M3.2."""

from __future__ import annotations

import pandas as pd
import pytest

from pangu.factor.fundamental import FundamentalFactorEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_DF = pd.DataFrame({
    "symbol": ["601899", "600967", "000750"],
    "pe_ttm": [15.2, 22.5, -8.0],
    "pb": [2.8, 1.5, 0.9],
    "roe_ttm": [18.5, 12.0, -5.0],
    "revenue_yoy": [25.3, 10.5, -15.0],
    "profit_yoy": [30.1, 8.2, -40.0],
})


@pytest.fixture
def engine() -> FundamentalFactorEngine:
    return FundamentalFactorEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompute:
    def test_normal_three_stocks(self, engine: FundamentalFactorEngine) -> None:
        result = engine.compute(_SAMPLE_DF)

        assert list(result.index) == ["601899", "600967", "000750"]
        assert set(result.columns) == set(engine.get_factor_names())

        assert result.loc["601899", "pe_ttm"] == pytest.approx(15.2)
        assert result.loc["601899", "roe_ttm"] == pytest.approx(18.5)
        assert result.loc["600967", "pb"] == pytest.approx(1.5)
        assert result.loc["000750", "pe_ttm"] == pytest.approx(-8.0)
        assert result.loc["000750", "profit_yoy"] == pytest.approx(-40.0)

    def test_empty_input(self, engine: FundamentalFactorEngine) -> None:
        result = engine.compute(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == engine.get_factor_names()

    def test_none_input(self, engine: FundamentalFactorEngine) -> None:
        result = engine.compute(None)
        assert result.empty

    def test_single_stock(self, engine: FundamentalFactorEngine) -> None:
        df = _SAMPLE_DF[_SAMPLE_DF["symbol"] == "601899"]
        result = engine.compute(df)
        assert len(result) == 1
        assert result.loc["601899", "revenue_yoy"] == pytest.approx(25.3)

    def test_symbol_as_index(self, engine: FundamentalFactorEngine) -> None:
        df = _SAMPLE_DF.set_index("symbol")
        result = engine.compute(df)
        assert result.index.name == "symbol"
        assert len(result) == 3

    def test_missing_columns_filled_nan(self, engine: FundamentalFactorEngine) -> None:
        df = pd.DataFrame({"symbol": ["601899"], "pe_ttm": [10.0]})
        result = engine.compute(df)
        assert result.loc["601899", "pe_ttm"] == pytest.approx(10.0)
        assert pd.isna(result.loc["601899", "pb"])
        assert pd.isna(result.loc["601899", "roe_ttm"])


class TestGetFactorNames:
    def test_returns_five_factors(self, engine: FundamentalFactorEngine) -> None:
        names = engine.get_factor_names()
        assert len(names) == 5
        assert "pe_ttm" in names
        assert "roe_ttm" in names

    def test_returns_copy(self, engine: FundamentalFactorEngine) -> None:
        names = engine.get_factor_names()
        names.append("extra")
        assert "extra" not in engine.get_factor_names()
