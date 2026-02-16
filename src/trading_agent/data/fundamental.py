"""FundamentalDataProvider Protocol — PRD §4.1.2 / §6."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class FundamentalDataProvider(Protocol):
    """Interface for A-share fundamental / valuation data."""

    def get_valuation(self, symbol: str) -> dict[str, Any]:
        """Return valuation metrics: PE, PB, PS, market cap."""
        ...

    def get_financial_indicator(self, symbol: str) -> pd.DataFrame:
        """Return financial indicators: ROE, revenue growth, net profit growth."""
        ...


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------

_VALUATIONS: dict[str, dict[str, Any]] = {
    "600519": {"pe_ttm": 30.5, "pb": 10.2, "ps": 15.0, "market_cap": 2.2e12},
    "000858": {"pe_ttm": 22.0, "pb": 5.8, "ps": 8.0, "market_cap": 5.8e11},
    "300750": {"pe_ttm": 25.0, "pb": 4.5, "ps": 6.0, "market_cap": 9.0e11},
    "601318": {"pe_ttm": 8.5, "pb": 1.1, "ps": 1.5, "market_cap": 8.7e11},
    "000001": {"pe_ttm": 6.0, "pb": 0.6, "ps": 2.0, "market_cap": 2.3e11},
}

_DEFAULT_VALUATION: dict[str, Any] = {
    "pe_ttm": 15.0, "pb": 2.0, "ps": 3.0, "market_cap": 1.0e11,
}


class FakeFundamentalDataProvider:
    """Deterministic fake data for testing."""

    def get_valuation(self, symbol: str) -> dict[str, Any]:
        return dict(_VALUATIONS.get(symbol, _DEFAULT_VALUATION))

    def get_financial_indicator(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "symbol": symbol,
            "date": "2025-12-31",
            "roe_ttm": 0.18,
            "revenue_yoy": 0.12,
            "profit_yoy": 0.15,
        }])

