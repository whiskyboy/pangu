"""Fake implementations for testing — moved from src/ modules."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pangu.models import (
    Action,
    NewsCategory,
    NewsItem,
    Region,
    SignalStatus,
    TradeSignal,
)
from pangu.tz import now as _tz_now


# ---------------------------------------------------------------------------
# FakeMarketDataProvider  (was in pangu.data.market)
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

    def _get_us_indices(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "SPX", "name": "S&P 500", "date": "2026-01-02", "open": 5180.0,
             "high": 5220.0, "low": 5170.0, "close": 5200.0, "volume": 3e9,
             "change_pct": 0.3, "source": "us_index"},
            {"symbol": "DJI", "name": "Dow Jones", "date": "2026-01-02", "open": 38900.0,
             "high": 39100.0, "low": 38800.0, "close": 39000.0, "volume": 2e9,
             "change_pct": 0.2, "source": "us_index"},
            {"symbol": "IXIC", "name": "NASDAQ", "date": "2026-01-02", "open": 16400.0,
             "high": 16600.0, "low": 16350.0, "close": 16500.0, "volume": 4e9,
             "change_pct": 0.5, "source": "us_index"},
        ])

    def _get_hk_indices(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "HSI", "name": "恒生指数", "date": "2026-01-02", "open": 17600.0,
             "high": 17700.0, "low": 17400.0, "close": 17500.0, "volume": 1e9,
             "change_pct": -0.3, "source": "hk_index"},
        ])

    def _get_commodity_futures(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "GC", "name": "COMEX黄金", "date": "2026-01-02", "open": 2340.0,
             "high": 2360.0, "low": 2330.0, "close": 2350.0, "volume": 1e6,
             "change_pct": 0.1, "source": "commodity"},
            {"symbol": "SI", "name": "COMEX白银", "date": "2026-01-02", "open": 28.3,
             "high": 28.8, "low": 28.1, "close": 28.5, "volume": 5e5,
             "change_pct": -0.2, "source": "commodity"},
            {"symbol": "CL", "name": "WTI原油", "date": "2026-01-02", "open": 77.5,
             "high": 78.5, "low": 77.0, "close": 78.0, "volume": 8e5,
             "change_pct": 0.8, "source": "commodity"},
            {"symbol": "HG", "name": "LME铜", "date": "2026-01-02", "open": 4.15,
             "high": 4.25, "low": 4.1, "close": 4.2, "volume": 3e5,
             "change_pct": 0.4, "source": "commodity"},
            {"symbol": "FEF", "name": "铁矿石", "date": "2026-01-02", "open": 830.0,
             "high": 835.0, "low": 815.0, "close": 820.0, "volume": 2e5,
             "change_pct": -1.0, "source": "commodity"},
        ])

    def get_global_snapshot(self) -> pd.DataFrame:
        return pd.concat(
            [self._get_us_indices(), self._get_hk_indices(), self._get_commodity_futures()],
            ignore_index=True,
        )

    def get_index_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Generate fake index daily bars."""
        rng = np.random.default_rng(hash(symbol) % 2**31)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        if n == 0:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
            )
        base = 3500.0
        close = base * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        return pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": close * rng.uniform(0.99, 1.01, n),
            "high": close * rng.uniform(1.0, 1.02, n),
            "low": close * rng.uniform(0.98, 1.0, n),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "amount": rng.integers(10_000_000, 50_000_000, n).astype(float),
            "adj_factor": np.ones(n),
        })


# ---------------------------------------------------------------------------
# FakeFundamentalDataProvider  (was in pangu.data.fundamental)
# ---------------------------------------------------------------------------


class FakeFundamentalDataProvider:
    """Deterministic fake data for testing."""

    def get_financial_indicator(
        self, symbol: str, start: str | None = None, end: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame([{
            "symbol": symbol,
            "date": "2025-12-31",
            "roe_ttm": 0.18,
            "revenue_yoy": 0.12,
            "profit_yoy": 0.15,
        }])


# ---------------------------------------------------------------------------
# FakeNewsDataProvider  (was in pangu.data.news)
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 16, 10, 0, 0)


class FakeNewsDataProvider:
    """Deterministic fake news for testing."""

    def get_latest_news(self, limit: int = 50) -> list[NewsItem]:
        items = [
            NewsItem(
                timestamp=_NOW - timedelta(minutes=10),
                title="央行宣布降准50基点",
                content="中国人民银行决定下调金融机构存款准备金率0.5个百分点。",
                source="财联社",
                region=Region.DOMESTIC,
                symbols=[],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(minutes=30),
                title="贵州茅台发布业绩预增公告",
                content="贵州茅台预计2025年净利润同比增长15%-20%。",
                source="东方财富",
                region=Region.DOMESTIC,
                symbols=["600519"],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(hours=1),
                title="新能源汽车销量创新高",
                content="2025年新能源汽车销量突破1000万辆。",
                source="财联社",
                region=Region.DOMESTIC,
                symbols=["300750"],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(hours=2),
                title="银行板块集体上涨",
                content="多家银行股涨停，市场情绪回暖。",
                source="东方财富",
                region=Region.DOMESTIC,
                symbols=["000001", "601318"],
            ),
        ]
        return items[:limit]

    def get_stock_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        all_news = self.get_latest_news(limit=50)
        return [n for n in all_news if symbol in n.symbols][:limit]

    def get_announcements(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        return [
            NewsItem(
                timestamp=_NOW - timedelta(hours=1),
                title=f"{symbol} 关于2025年度业绩预增的公告",
                content="http://www.cninfo.com.cn/example",
                source="巨潮",
                region=Region.DOMESTIC,
                symbols=[symbol],
                category=NewsCategory.ANNOUNCEMENT,
            ),
        ][:limit]


# ---------------------------------------------------------------------------
# FakeStockPool  (was in pangu.data.stock_pool)
# ---------------------------------------------------------------------------


class FakeStockPool:
    """Loads watchlist from YAML; factor pool is an empty stub."""

    def __init__(self, watchlist_path: str | Path = "config/watchlist.yaml") -> None:
        self._path = Path(watchlist_path)
        self._symbols: list[str] = []
        if self._path.exists():
            data = yaml.safe_load(self._path.read_text())
            watchlist = (data or {}).get("watchlist") or []
            self._symbols = [
                item["symbol"] for item in watchlist if "symbol" in item
            ]

    def get_watchlist(self) -> list[str]:
        return list(self._symbols)

    def add_to_watchlist(self, symbol: str) -> None:
        if symbol not in self._symbols:
            self._symbols.append(symbol)

    def remove_from_watchlist(self, symbol: str) -> None:
        if symbol in self._symbols:
            self._symbols.remove(symbol)

    def get_all_symbols(self) -> list[str]:
        return list(self._symbols)

    def get_stock_metadata(self) -> dict:
        from pangu.models import StockMeta
        return {s: StockMeta(name=s, sector="") for s in self._symbols}

    def sync_index_constituents(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# FakeLLMJudgeEngine  (was in pangu.strategy.llm_engine)
# ---------------------------------------------------------------------------


class FakeLLMJudgeEngine:
    """Deterministic judge based on factor_score. For testing only.

    Unlike LLMJudgeEngineImpl, no error handling — expects well-formed input.
    """

    async def judge_stock(
        self,
        symbol: str,
        name: str,
        factor_score: float,
        factor_rank: int,
        factor_details: dict[str, float],
        stock_news: list[NewsItem],
        announcements: list[NewsItem],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        price: float,
    ) -> TradeSignal:

        if factor_score >= 0.7:
            action = Action.BUY
        elif factor_score <= 0.3:
            action = Action.SELL
        else:
            action = Action.HOLD

        return TradeSignal(
            timestamp=_tz_now(),
            symbol=symbol,
            name=name,
            action=action,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=price,
            confidence=factor_score,
            source="fake_llm_judge",
            reason=f"fake judge: score={factor_score:.2f}",
            factor_score=factor_score,
            metadata={},
        )

    async def judge_pool(
        self,
        candidates: list[dict[str, Any]],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
    ) -> list[TradeSignal]:
        signals = []
        for c in candidates:
            signal = await self.judge_stock(
                symbol=c["symbol"],
                name=c["name"],
                factor_score=c["factor_score"],
                factor_rank=c["factor_rank"],
                factor_details=c.get("factor_details", {}),
                stock_news=c.get("stock_news", []),
                announcements=c.get("announcements", []),
                telegraph=telegraph,
                global_market=global_market,
                price=c["price"],
            )
            signals.append(signal)
        return signals
