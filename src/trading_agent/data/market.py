"""MarketDataProvider Protocol — PRD §4.1.1 / §6."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    """Unified interface for A-share and international market data."""

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return A-share daily OHLCV bars with adjustment factor."""
        ...

    def get_realtime_quote(self, symbols: list[str]) -> pd.DataFrame:
        """Return real-time quote snapshot for given symbols."""
        ...

    def get_stock_list(self) -> pd.DataFrame:
        """Return full A-share stock list (code, name, industry, listing_date)."""
        ...

    def get_us_indices(self) -> pd.DataFrame:
        """Return latest US major indices (S&P 500, DJIA, NASDAQ)."""
        ...

    def get_hk_indices(self) -> pd.DataFrame:
        """Return latest Hang Seng Index data."""
        ...

    def get_commodity_futures(self) -> pd.DataFrame:
        """Return international commodity futures (gold, silver, crude oil, copper, iron ore)."""
        ...

    def get_global_snapshot(self) -> pd.DataFrame:
        """Aggregate all international quotes into a single snapshot."""
        ...
