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


# Backward-compatible re-export
from trading_agent.data.fundamental_akshare import AkShareFundamentalProvider  # noqa: E402, F401

