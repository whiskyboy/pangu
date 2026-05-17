"""Tests for IndexStockPool — DB-backed index-constituent pool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pangu.data.stock_pool.index_pool import IndexStockPool
from pangu.data.storage import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Database:
    d = Database(":memory:")
    d.init_tables()
    return d


# ---------------------------------------------------------------------------
# sync_trading_calendar
# ---------------------------------------------------------------------------


class TestSyncTradingCalendar:
    @patch("akshare.tool_trade_date_hist_sina")
    def test_syncs_dates(self, mock_cal: MagicMock, db: Database) -> None:
        mock_cal.return_value = pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06"],
            }
        )
        pool = IndexStockPool(db)
        count = pool.sync_trading_calendar()

        assert count == 3
        assert db.is_trading_day("2026-01-02") is True
        assert db.is_trading_day("2026-01-04") is False

    @patch("akshare.tool_trade_date_hist_sina")
    def test_empty_returns_zero(self, mock_cal: MagicMock, db: Database) -> None:
        mock_cal.return_value = pd.DataFrame()
        pool = IndexStockPool(db)
        assert pool.sync_trading_calendar() == 0


# ---------------------------------------------------------------------------
# Index constituents
# ---------------------------------------------------------------------------


class TestIndexConstituents:
    @patch("akshare.stock_individual_info_em")
    @patch("akshare.index_stock_cons")
    def test_sync_index_constituents(
        self,
        mock_cons: MagicMock,
        mock_info: MagicMock,
        db: Database,
    ) -> None:
        mock_cons.return_value = pd.DataFrame(
            {
                "品种代码": ["600519", "000858"],
                "品种名称": ["贵州茅台", "五粮液"],
                "纳入日期": ["2020-01-01", "2020-01-01"],
            }
        )
        mock_info.side_effect = lambda symbol: pd.DataFrame(
            {
                "item": ["股票简称", "上市时间", "行业"],
                "value": [
                    "茅台" if symbol == "600519" else "五粮液",
                    "20010827",
                    "白酒" if symbol == "600519" else "食品饮料",
                ],
            }
        )
        pool = IndexStockPool(db)
        count = pool.sync_index_constituents()
        assert count == 2

    @patch("akshare.stock_individual_info_em")
    @patch("akshare.index_stock_cons")
    def test_sync_multi_index(
        self,
        mock_cons: MagicMock,
        mock_info: MagicMock,
        db: Database,
    ) -> None:
        """Sync multiple indices — rows from both are saved."""

        def fake_cons(symbol: str) -> pd.DataFrame:
            if symbol == "000300":
                return pd.DataFrame(
                    {
                        "品种代码": ["600519"],
                        "品种名称": ["贵州茅台"],
                        "纳入日期": ["2020-01-01"],
                    }
                )
            return pd.DataFrame(
                {
                    "品种代码": ["002475"],
                    "品种名称": ["立讯精密"],
                    "纳入日期": ["2021-06-01"],
                }
            )

        mock_cons.side_effect = fake_cons
        mock_info.return_value = pd.DataFrame(
            {
                "item": ["行业"],
                "value": ["电子"],
            }
        )
        pool = IndexStockPool(db, indices=["000300", "000905"])
        count = pool.sync_index_constituents()
        assert count == 2
        assert len(db.load_index_constituents("000300")) == 1
        assert len(db.load_index_constituents("000905")) == 1

    @patch("akshare.stock_individual_info_em")
    @patch("akshare.index_stock_cons")
    def test_sync_removes_unconfigured_index(
        self,
        mock_cons: MagicMock,
        mock_info: MagicMock,
        db: Database,
    ) -> None:
        """After removing an index from config, its constituents are cleaned up."""
        # Pre-populate DB with two indices
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
                {
                    "symbol": "002475",
                    "name": "立讯精密",
                    "index_code": "000905",
                    "sector": "电子",
                    "date": "2026-02-20",
                },
            ]
        )
        assert len(db.load_index_constituents("000905")) == 1

        # Now create pool with only 000300 configured
        mock_cons.return_value = pd.DataFrame(
            {
                "品种代码": ["600519"],
                "品种名称": ["贵州茅台"],
                "纳入日期": ["2020-01-01"],
            }
        )
        mock_info.return_value = pd.DataFrame(
            {
                "item": ["行业"],
                "value": ["白酒"],
            }
        )
        pool = IndexStockPool(db, indices=["000300"])
        pool.sync_index_constituents()

        # 000905 constituents should be removed
        assert len(db.load_index_constituents("000905")) == 0
        assert len(db.load_index_constituents("000300")) == 1


# ---------------------------------------------------------------------------
# _get_index_stocks / get_all_symbols
# ---------------------------------------------------------------------------


class TestGetSymbols:
    def test_get_index_stocks_from_db(self, db: Database) -> None:
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
                {
                    "symbol": "000858",
                    "name": "五粮液",
                    "index_code": "000300",
                    "sector": "食品饮料",
                    "date": "2026-02-20",
                },
            ]
        )
        pool = IndexStockPool(db)
        stocks = pool._get_index_stocks()
        assert set(stocks) == {"600519", "000858"}

    def test_get_index_stocks_multi_index_dedup(self, db: Database) -> None:
        """Same stock in two indices is deduplicated."""
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000905",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
                {
                    "symbol": "002475",
                    "name": "立讯精密",
                    "index_code": "000905",
                    "sector": "电子",
                    "date": "2026-02-20",
                },
            ]
        )
        pool = IndexStockPool(db, indices=["000300", "000905"])
        stocks = pool._get_index_stocks()
        assert len(stocks) == 2
        assert set(stocks) == {"600519", "002475"}

    def test_get_all_symbols_equals_index_stocks(self, db: Database) -> None:
        """get_all_symbols is a thin wrapper around _get_index_stocks."""
        db.save_index_constituents(
            [
                {
                    "symbol": "601899",
                    "name": "紫金矿业",
                    "index_code": "000300",
                    "sector": "有色金属",
                    "date": "2026-02-20",
                },
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
            ]
        )
        pool = IndexStockPool(db)
        assert pool.get_all_symbols() == pool._get_index_stocks()
        assert set(pool.get_all_symbols()) == {"601899", "600519"}

    def test_get_all_symbols_empty_db(self, db: Database) -> None:
        pool = IndexStockPool(db)
        assert pool.get_all_symbols() == []


# ---------------------------------------------------------------------------
# get_stock_metadata
# ---------------------------------------------------------------------------


class TestGetStockMetadata:
    def test_returns_metadata_from_db(self, db: Database) -> None:
        db.save_index_constituents(
            [
                {
                    "symbol": "601899",
                    "name": "紫金矿业",
                    "index_code": "000300",
                    "sector": "有色金属",
                    "date": "2026-02-20",
                },
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
            ]
        )
        pool = IndexStockPool(db)
        meta = pool.get_stock_metadata()
        assert meta["601899"].name == "紫金矿业"
        assert meta["601899"].sector == "有色金属"
        assert meta["600519"].name == "贵州茅台"
        assert meta["600519"].sector == "白酒"

    def test_empty_db_returns_empty_dict(self, db: Database) -> None:
        pool = IndexStockPool(db)
        assert pool.get_stock_metadata() == {}

    def test_multi_index_dedup_picks_first(self, db: Database) -> None:
        """If a stock appears in multiple indices, the first row wins."""
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000905",
                    "sector": "白酒",
                    "date": "2026-02-20",
                },
            ]
        )
        pool = IndexStockPool(db)
        meta = pool.get_stock_metadata()
        assert "600519" in meta
        assert len(meta) == 1
