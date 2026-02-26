"""FundamentalDataProvider Protocol — PRD §4.1.2 / §6."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class FundamentalDataProvider(Protocol):
    """Interface for A-share fundamental data."""

    def get_financial_indicator(
        self, symbol: str, start: str | None = None, end: str | None = None,
    ) -> pd.DataFrame:
        """Return financial indicators: ROE, revenue growth, net profit growth."""
        ...
