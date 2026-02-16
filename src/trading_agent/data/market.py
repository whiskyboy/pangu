"""MarketDataProvider Protocol — PRD §4.1.1 / §6."""

from __future__ import annotations

from typing import Protocol

import numpy as np
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


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------

_STOCKS = {
    "600519": ("贵州茅台", "白酒"),
    "000858": ("五粮液", "白酒"),
    "300750": ("宁德时代", "新能源"),
    "601318": ("中国平安", "保险"),
    "000001": ("平安银行", "银行"),
}

_BASE_PRICES: dict[str, float] = {
    "600519": 1800.0,
    "000858": 150.0,
    "300750": 220.0,
    "601318": 48.0,
    "000001": 12.0,
}


class FakeMarketDataProvider:
    """Deterministic fake data for testing."""

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(hash(symbol) % 2**31)
        base = _BASE_PRICES.get(symbol, 100.0)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        if n == 0:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
            )
        close = base * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        open_ = close * rng.uniform(0.98, 1.02, n)
        high = np.maximum(open_, close) * rng.uniform(1.0, 1.03, n)
        low = np.minimum(open_, close) * rng.uniform(0.97, 1.0, n)
        volume = rng.integers(1_000_000, 10_000_000, n)
        return pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "adj_factor": 1.0,
        })

    def get_realtime_quote(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        for s in symbols:
            name, _ = _STOCKS.get(s, (s, "未知"))
            base = _BASE_PRICES.get(s, 100.0)
            rows.append({
                "symbol": s,
                "name": name,
                "price": base,
                "change_pct": 0.5,
                "volume": 5_000_000,
                "amount": base * 5_000_000,
                "volume_ratio": 1.2,
            })
        return pd.DataFrame(rows)

    def get_stock_list(self) -> pd.DataFrame:
        rows = [
            {"symbol": sym, "name": name, "industry": ind, "listing_date": "2000-01-01"}
            for sym, (name, ind) in _STOCKS.items()
        ]
        return pd.DataFrame(rows)

    def get_us_indices(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "SPX", "name": "S&P 500", "price": 5200.0, "change_pct": 0.3},
            {"symbol": "DJI", "name": "Dow Jones", "price": 39000.0, "change_pct": 0.2},
            {"symbol": "IXIC", "name": "NASDAQ", "price": 16500.0, "change_pct": 0.5},
        ])

    def get_hk_indices(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "HSI", "name": "恒生指数", "price": 17500.0, "change_pct": -0.3},
        ])

    def get_commodity_futures(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "GC", "name": "黄金", "price": 2350.0, "change_pct": 0.1},
            {"symbol": "SI", "name": "白银", "price": 28.5, "change_pct": -0.2},
            {"symbol": "CL", "name": "原油", "price": 78.0, "change_pct": 0.8},
            {"symbol": "HG", "name": "铜", "price": 4.2, "change_pct": 0.4},
            {"symbol": "DCE_I", "name": "铁矿石", "price": 820.0, "change_pct": -1.0},
        ])

    def get_global_snapshot(self) -> pd.DataFrame:
        return pd.concat(
            [self.get_us_indices(), self.get_hk_indices(), self.get_commodity_futures()],
            ignore_index=True,
        )

