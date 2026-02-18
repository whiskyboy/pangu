"""Tests for FundamentalFactorEngine — M3.2."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading_agent.factor.fundamental import FundamentalFactorEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(
    valuations: dict[str, dict] | None = None,
    financials: dict[str, pd.DataFrame] | None = None,
    fail_symbols: set[str] | None = None,
) -> MagicMock:
    """Build a mock FundamentalDataProvider."""
    valuations = valuations or {}
    financials = financials or {}
    fail_symbols = fail_symbols or set()

    provider = MagicMock()

    def get_val(symbol: str) -> dict:
        if symbol in fail_symbols:
            raise RuntimeError("API error")
        return valuations.get(symbol, {})

    def get_fin(symbol: str) -> pd.DataFrame:
        if symbol in fail_symbols:
            raise RuntimeError("API error")
        return financials.get(symbol, pd.DataFrame())

    provider.get_valuation.side_effect = get_val
    provider.get_financial_indicator.side_effect = get_fin
    return provider


_SAMPLE_VALUATIONS = {
    "601899": {"pe_ttm": 15.2, "pb": 2.8, "ps": 3.0, "market_cap": 3.5e11},
    "600967": {"pe_ttm": 22.5, "pb": 1.5, "ps": 2.0, "market_cap": 1.0e11},
    "000750": {"pe_ttm": -8.0, "pb": 0.9, "ps": 1.0, "market_cap": 5.0e10},
}

_SAMPLE_FINANCIALS = {
    "601899": pd.DataFrame({
        "symbol": ["601899"], "date": ["2025-12-31"],
        "roe_ttm": [18.5], "revenue_yoy": [25.3], "profit_yoy": [30.1],
    }),
    "600967": pd.DataFrame({
        "symbol": ["600967"], "date": ["2025-12-31"],
        "roe_ttm": [12.0], "revenue_yoy": [10.5], "profit_yoy": [8.2],
    }),
    "000750": pd.DataFrame({
        "symbol": ["000750"], "date": ["2025-12-31"],
        "roe_ttm": [-5.0], "revenue_yoy": [-15.0], "profit_yoy": [-40.0],
    }),
}


@pytest.fixture
def engine() -> FundamentalFactorEngine:
    return FundamentalFactorEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompute:
    def test_normal_three_stocks(self, engine: FundamentalFactorEngine) -> None:
        provider = _mock_provider(_SAMPLE_VALUATIONS, _SAMPLE_FINANCIALS)
        result = engine.compute(["601899", "600967", "000750"], provider)

        assert list(result.index) == ["601899", "600967", "000750"]
        assert set(result.columns) == set(engine.get_factor_names())

        # Spot check values
        assert result.loc["601899", "pe_ttm"] == pytest.approx(15.2)
        assert result.loc["601899", "roe_ttm"] == pytest.approx(18.5)
        assert result.loc["600967", "pb"] == pytest.approx(1.5)
        assert result.loc["000750", "pe_ttm"] == pytest.approx(-8.0)
        assert result.loc["000750", "profit_yoy"] == pytest.approx(-40.0)

    def test_empty_symbols(self, engine: FundamentalFactorEngine) -> None:
        provider = _mock_provider()
        result = engine.compute([], provider)
        assert result.empty
        assert list(result.columns) == engine.get_factor_names()

    def test_single_stock(self, engine: FundamentalFactorEngine) -> None:
        provider = _mock_provider(_SAMPLE_VALUATIONS, _SAMPLE_FINANCIALS)
        result = engine.compute(["601899"], provider)
        assert len(result) == 1
        assert result.loc["601899", "revenue_yoy"] == pytest.approx(25.3)


class TestFailureHandling:
    def test_one_stock_fails_others_ok(self, engine: FundamentalFactorEngine) -> None:
        provider = _mock_provider(
            _SAMPLE_VALUATIONS, _SAMPLE_FINANCIALS, fail_symbols={"600967"}
        )
        result = engine.compute(["601899", "600967", "000750"], provider)

        # 601899 and 000750 should have data
        assert result.loc["601899", "pe_ttm"] == pytest.approx(15.2)
        assert result.loc["000750", "pb"] == pytest.approx(0.9)

        # 600967 should be all NaN
        assert result.loc["600967"].isna().all()

    def test_all_stocks_fail(self, engine: FundamentalFactorEngine) -> None:
        provider = _mock_provider(fail_symbols={"601899", "600967"})
        result = engine.compute(["601899", "600967"], provider)
        assert len(result) == 2
        assert result.isna().all().all()

    def test_missing_valuation_fields(self, engine: FundamentalFactorEngine) -> None:
        provider = _mock_provider(
            valuations={"601899": {"pe_ttm": 10.0}},  # missing pb
            financials=_SAMPLE_FINANCIALS,
        )
        result = engine.compute(["601899"], provider)
        assert result.loc["601899", "pe_ttm"] == pytest.approx(10.0)
        assert pd.isna(result.loc["601899", "pb"])

    def test_empty_financial_dataframe(self, engine: FundamentalFactorEngine) -> None:
        provider = _mock_provider(
            valuations=_SAMPLE_VALUATIONS,
            financials={"601899": pd.DataFrame()},
        )
        result = engine.compute(["601899"], provider)
        assert result.loc["601899", "pe_ttm"] == pytest.approx(15.2)
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
