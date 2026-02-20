"""MarketDataProvider Protocol — PRD §4.1.1 / §6."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from trading_agent.utils import CircuitBreaker, retry_call


class MarketDataProvider(Protocol):
    """Unified interface for A-share and international market data."""

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return A-share daily OHLCV bars with adjustment factor."""
        ...

    def get_global_snapshot(self) -> pd.DataFrame:
        """Aggregate all international quotes into a single snapshot."""
        ...


# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# ---------------------------------------------------------------------------

from trading_agent.data.market_akshare import AkShareMarketDataProvider  # noqa: E402, F401
from trading_agent.data.market_baostock import (  # noqa: E402, F401
    BaoStockMarketDataProvider,
    _to_bs_code,
)

# Backward-compatible alias
_retry_call = retry_call

