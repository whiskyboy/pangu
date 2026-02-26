"""Tests for Fake data providers (M1.4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.fakes import (
    FakeFundamentalDataProvider,
    FakeMarketDataProvider,
    FakeNewsDataProvider,
    FakeStockPool,
)
from pangu.models import NewsItem, Region

# ---------------------------------------------------------------------------
# FakeMarketDataProvider
# ---------------------------------------------------------------------------


class TestFakeMarketDataProvider:
    def test_daily_bars_columns(self, fake_market: FakeMarketDataProvider) -> None:
        df = fake_market.get_daily_bars("600519", "2025-01-01", "2025-06-30")
        expected = {"date", "open", "high", "low", "close", "volume", "amount", "adj_factor"}
        assert expected == set(df.columns)

    def test_daily_bars_row_count(self, fake_market: FakeMarketDataProvider) -> None:
        df = fake_market.get_daily_bars("600519", "2025-01-01", "2025-06-30")
        assert len(df) > 100  # ~120 business days in 6 months

    def test_daily_bars_empty_range(self, fake_market: FakeMarketDataProvider) -> None:
        df = fake_market.get_daily_bars("600519", "2025-01-01", "2024-01-01")
        assert len(df) == 0
        assert "close" in df.columns

    def test_daily_bars_deterministic(self, fake_market: FakeMarketDataProvider) -> None:
        df1 = fake_market.get_daily_bars("600519", "2025-01-01", "2025-03-01")
        df2 = fake_market.get_daily_bars("600519", "2025-01-01", "2025-03-01")
        pd.testing.assert_frame_equal(df1, df2)

    def test_daily_bars_prices_positive(self, fake_market: FakeMarketDataProvider) -> None:
        df = fake_market.get_daily_bars("600519", "2025-01-01", "2025-03-01")
        for col in ("open", "high", "low", "close"):
            assert (df[col] > 0).all(), f"{col} has non-positive values"

    def test_global_snapshot_aggregates(self, fake_market: FakeMarketDataProvider) -> None:
        df = fake_market.get_global_snapshot()
        # 3 US + 1 HK + 5 commodities = 9
        assert len(df) == 9


# ---------------------------------------------------------------------------
# FakeFundamentalDataProvider
# ---------------------------------------------------------------------------


class TestFakeFundamentalDataProvider:
    def test_financial_indicator_columns(self, fake_fundamental: FakeFundamentalDataProvider) -> None:
        df = fake_fundamental.get_financial_indicator("600519")
        expected = {"symbol", "date", "roe_ttm", "revenue_yoy", "profit_yoy"}
        assert expected == set(df.columns)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# FakeNewsDataProvider
# ---------------------------------------------------------------------------


class TestFakeNewsDataProvider:
    def test_latest_news_returns_news_items(self, fake_news: FakeNewsDataProvider) -> None:
        items = fake_news.get_latest_news()
        assert len(items) == 4
        assert all(isinstance(n, NewsItem) for n in items)

    def test_latest_news_limit(self, fake_news: FakeNewsDataProvider) -> None:
        items = fake_news.get_latest_news(limit=2)
        assert len(items) == 2

    def test_latest_news_region(self, fake_news: FakeNewsDataProvider) -> None:
        items = fake_news.get_latest_news()
        assert all(n.region == Region.DOMESTIC for n in items)

    def test_stock_news_filters_by_symbol(self, fake_news: FakeNewsDataProvider) -> None:
        items = fake_news.get_stock_news("600519")
        assert len(items) >= 1
        assert all("600519" in n.symbols for n in items)

    def test_stock_news_unknown_symbol_empty(self, fake_news: FakeNewsDataProvider) -> None:
        items = fake_news.get_stock_news("999999")
        assert items == []

    def test_news_items_have_required_fields(self, fake_news: FakeNewsDataProvider) -> None:
        for item in fake_news.get_latest_news():
            assert item.timestamp is not None
            assert item.title
            assert item.content
            assert item.source
            assert isinstance(item.region, Region)
            assert isinstance(item.symbols, list)


# ---------------------------------------------------------------------------
# FakeStockPool
# ---------------------------------------------------------------------------


class TestFakeStockPool:
    def test_loads_watchlist_from_yaml(self, fake_stock_pool: FakeStockPool) -> None:
        wl = fake_stock_pool.get_watchlist()
        assert wl == ["600519", "000858", "300750"]

    def test_add_to_watchlist(self, fake_stock_pool: FakeStockPool) -> None:
        fake_stock_pool.add_to_watchlist("601318")
        assert "601318" in fake_stock_pool.get_watchlist()

    def test_add_duplicate_no_effect(self, fake_stock_pool: FakeStockPool) -> None:
        fake_stock_pool.add_to_watchlist("600519")
        assert fake_stock_pool.get_watchlist().count("600519") == 1

    def test_remove_from_watchlist(self, fake_stock_pool: FakeStockPool) -> None:
        fake_stock_pool.remove_from_watchlist("600519")
        assert "600519" not in fake_stock_pool.get_watchlist()

    def test_remove_nonexistent_no_error(self, fake_stock_pool: FakeStockPool) -> None:
        fake_stock_pool.remove_from_watchlist("999999")  # should not raise

    def test_get_all_symbols(self, fake_stock_pool: FakeStockPool) -> None:
        assert fake_stock_pool.get_all_symbols() == ["600519", "000858", "300750"]

    def test_get_stock_metadata(self, fake_stock_pool: FakeStockPool) -> None:
        meta = fake_stock_pool.get_stock_metadata()
        assert isinstance(meta, dict)

    def test_watchlist_returns_copy(self, fake_stock_pool: FakeStockPool) -> None:
        wl1 = fake_stock_pool.get_watchlist()
        wl2 = fake_stock_pool.get_watchlist()
        assert wl1 is not wl2

    def test_missing_yaml_file(self, tmp_path: Path) -> None:
        sp = FakeStockPool(watchlist_path=tmp_path / "nonexistent.yaml")
        assert sp.get_watchlist() == []
