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


def _fake_cninfo_row(symbol: str, sector: str, **overrides: str) -> pd.DataFrame:
    """Build a 1-row cninfo DataFrame with 26 columns (default empty)."""
    base = {
        "公司名称": f"{symbol} Inc.",
        "英文名称": "",
        "曾用简称": "",
        "A股代码": symbol,
        "A股简称": f"Stock-{symbol}",
        "B股代码": "",
        "B股简称": "",
        "H股代码": "",
        "H股简称": "",
        "入选指数": "",
        "所属市场": "",
        "所属行业": sector,
        "法人代表": "",
        "注册资金": "",
        "成立日期": "",
        "上市日期": "2001-08-27",
        "官方网站": "",
        "电子邮箱": "",
        "联系电话": "",
        "传真": "",
        "注册地址": "贵州省仁怀市",
        "办公地址": "",
        "邮政编码": "",
        "主营业务": f"{symbol} 主营业务描述",
        "经营范围": "",
        "机构简介": "",
    }
    base.update(overrides)
    return pd.DataFrame([base])


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
# Index constituents (now via cninfo)
# ---------------------------------------------------------------------------


class TestIndexConstituents:
    @patch("akshare.stock_profile_cninfo")
    @patch("akshare.index_stock_cons")
    def test_sync_index_constituents_writes_both_tables(
        self,
        mock_cons: MagicMock,
        mock_cninfo: MagicMock,
        db: Database,
    ) -> None:
        """sync_index_constituents dual-writes: index_constituents + stock_profiles."""
        mock_cons.return_value = pd.DataFrame(
            {
                "品种代码": ["600519", "000858"],
                "品种名称": ["贵州茅台", "五粮液"],
                "纳入日期": ["2020-01-01", "2020-01-01"],
            }
        )
        mock_cninfo.side_effect = lambda symbol: _fake_cninfo_row(
            symbol,
            "酒、饮料和精制茶制造业" if symbol == "600519" else "酒类业",
        )
        pool = IndexStockPool(db)
        count = pool.sync_index_constituents()

        # index_constituents
        assert count == 2
        const = db.load_index_constituents("000300")
        assert len(const) == 2
        sectors = {r["symbol"]: r["sector"] for r in const}
        assert sectors["600519"] == "酒、饮料和精制茶制造业"

        # stock_profiles
        assert db.count_stock_profiles() == 2
        p = db.load_stock_profile("600519")
        assert p is not None
        assert p["full_name"] == "600519 Inc."
        assert p["sector"] == "酒、饮料和精制茶制造业"
        assert p["list_date"] == "2001-08-27"
        assert p["main_business"] == "600519 主营业务描述"
        assert p["registered_area"] == "贵州省仁怀市"

    @patch("akshare.stock_profile_cninfo")
    @patch("akshare.index_stock_cons")
    def test_cninfo_failure_leaves_sector_empty_no_profile(
        self,
        mock_cons: MagicMock,
        mock_cninfo: MagicMock,
        db: Database,
    ) -> None:
        """cninfo returning empty DataFrame → sector="" in index_constituents, no profile row."""
        mock_cons.return_value = pd.DataFrame(
            {
                "品种代码": ["600519", "999999"],
                "品种名称": ["贵州茅台", "未知股"],
                "纳入日期": ["2020-01-01", "2020-01-01"],
            }
        )

        def fake_cninfo(symbol: str) -> pd.DataFrame:
            if symbol == "999999":
                return pd.DataFrame()  # delisted / not found
            return _fake_cninfo_row(symbol, "酒类业")

        mock_cninfo.side_effect = fake_cninfo
        pool = IndexStockPool(db)
        pool.sync_index_constituents()

        const = db.load_index_constituents("000300")
        sectors = {r["symbol"]: r["sector"] for r in const}
        assert sectors["600519"] == "酒类业"
        assert sectors["999999"] == ""

        assert db.count_stock_profiles() == 1
        assert db.load_stock_profile("999999") is None
        assert db.load_stock_profile("600519") is not None

    @patch("akshare.stock_profile_cninfo")
    @patch("akshare.index_stock_cons")
    def test_sync_multi_index(
        self,
        mock_cons: MagicMock,
        mock_cninfo: MagicMock,
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
        mock_cninfo.side_effect = lambda symbol: _fake_cninfo_row(symbol, "电子业")
        pool = IndexStockPool(db, indices=["000300", "000905"])
        count = pool.sync_index_constituents()
        assert count == 2
        assert len(db.load_index_constituents("000300")) == 1
        assert len(db.load_index_constituents("000905")) == 1
        assert db.count_stock_profiles() == 2

    @patch("akshare.stock_profile_cninfo")
    @patch("akshare.index_stock_cons")
    def test_sync_removes_unconfigured_index(
        self,
        mock_cons: MagicMock,
        mock_cninfo: MagicMock,
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
        mock_cninfo.return_value = _fake_cninfo_row("600519", "酒类业")
        pool = IndexStockPool(db, indices=["000300"])
        pool.sync_index_constituents()

        # 000905 constituents should be removed
        assert len(db.load_index_constituents("000905")) == 0
        assert len(db.load_index_constituents("000300")) == 1


# ---------------------------------------------------------------------------
# backfill_sectors (also via cninfo now, dual-writes both tables)
# ---------------------------------------------------------------------------


class TestBackfillSectors:
    @patch("akshare.stock_profile_cninfo")
    def test_backfills_sector_and_profile(
        self,
        mock_cninfo: MagicMock,
        db: Database,
    ) -> None:
        """backfill_sectors updates index_constituents.sector AND writes stock_profiles."""
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "",  # missing — to be backfilled
                    "date": "2020-01-01",
                },
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "",
                    "date": "2021-01-01",
                },
            ]
        )
        mock_cninfo.return_value = _fake_cninfo_row("600519", "酒类业")
        pool = IndexStockPool(db)
        updated = pool.backfill_sectors()
        # Both historical rows get sector via broadcast UPDATE
        assert updated == 2
        # stock_profiles also written (CLI-self-sufficient: no need to also run T1)
        p = db.load_stock_profile("600519")
        assert p is not None
        assert p["sector"] == "酒类业"
        assert p["main_business"] == "600519 主营业务描述"

    @patch("akshare.stock_profile_cninfo")
    def test_skips_when_no_missing_sectors(
        self,
        mock_cninfo: MagicMock,
        db: Database,
    ) -> None:
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "贵州茅台",
                    "index_code": "000300",
                    "sector": "酒类业",
                    "date": "2020-01-01",
                },
            ]
        )
        pool = IndexStockPool(db)
        assert pool.backfill_sectors() == 0
        mock_cninfo.assert_not_called()

    @patch("akshare.stock_profile_cninfo")
    def test_cninfo_failure_skips_symbol(
        self,
        mock_cninfo: MagicMock,
        db: Database,
    ) -> None:
        """cninfo returning empty for a symbol → its sector stays empty, no profile row."""
        db.save_index_constituents(
            [
                {
                    "symbol": "999999",
                    "name": "未知",
                    "index_code": "000300",
                    "sector": "",
                    "date": "2020-01-01",
                },
            ]
        )
        mock_cninfo.return_value = pd.DataFrame()
        pool = IndexStockPool(db)
        updated = pool.backfill_sectors()
        assert updated == 0
        assert db.load_stock_profile("999999") is None
        # Sector stays empty
        rows = db.load_index_constituents("000300")
        assert rows[0]["sector"] in (None, "")


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
# get_stock_metadata (now reads stock_profiles, falls back to index_constituents)
# ---------------------------------------------------------------------------


class TestGetStockMetadata:
    def test_returns_metadata_from_constituents_when_no_profile(self, db: Database) -> None:
        """Cold start: stock_profiles empty → falls back to index_constituents (name+sector only)."""
        db.save_index_constituents(
            [
                {
                    "symbol": "601899",
                    "name": "紫金矿业",
                    "index_code": "000300",
                    "sector": "有色金属",
                    "date": "2026-02-20",
                },
            ]
        )
        pool = IndexStockPool(db)
        meta = pool.get_stock_metadata()
        assert meta["601899"].name == "紫金矿业"
        assert meta["601899"].sector == "有色金属"
        # Extended fields default to empty strings
        assert meta["601899"].full_name == ""
        assert meta["601899"].list_date == ""
        assert meta["601899"].main_business == ""
        assert meta["601899"].registered_area == ""

    def test_returns_full_metadata_from_profile(self, db: Database) -> None:
        """When stock_profiles has the row, all 6 fields come through."""
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "茅台 (constituents row)",
                    "index_code": "000300",
                    "sector": "constituents sector",
                    "date": "2026-02-20",
                },
            ]
        )
        db.save_stock_profile(
            "600519",
            {
                "name": "贵州茅台",
                "full_name": "贵州茅台酒股份有限公司",
                "sector": "酒、饮料和精制茶制造业",
                "list_date": "2001-08-27",
                "main_business": "茅台酒及系列酒",
                "registered_area": "贵州省仁怀市",
            },
        )
        pool = IndexStockPool(db)
        m = pool.get_stock_metadata()["600519"]
        # Profile fields prevail over index_constituents
        assert m.name == "贵州茅台"
        assert m.sector == "酒、饮料和精制茶制造业"
        assert m.full_name == "贵州茅台酒股份有限公司"
        assert m.list_date == "2001-08-27"
        assert m.main_business == "茅台酒及系列酒"
        assert m.registered_area == "贵州省仁怀市"

    def test_profile_for_non_constituent_is_ignored(self, db: Database) -> None:
        """Universe is the constituents table — orphan profile rows aren't surfaced."""
        db.save_stock_profile("999999", {"name": "孤儿股"})
        pool = IndexStockPool(db)
        assert pool.get_stock_metadata() == {}

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
