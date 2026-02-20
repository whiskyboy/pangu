"""Tests for StockPoolManager — YAML persistence + data initialization."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from trading_agent.data.stock_pool_yaml import StockPoolManager
from trading_agent.data.storage import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Database:
    d = Database(":memory:")
    d.init_tables()
    return d


@pytest.fixture()
def tmp_yaml(tmp_path: Path) -> Path:
    """Create a temporary watchlist.yaml with 2 stocks."""
    p = tmp_path / "watchlist.yaml"
    data = {
        "watchlist": [
            {"symbol": "601899", "name": "紫金矿业", "sector": "有色金属"},
            {"symbol": "600967", "name": "内蒙一机", "sector": "军工"},
        ]
    }
    p.write_text(yaml.dump(data, allow_unicode=True))
    return p


@pytest.fixture()
def empty_yaml(tmp_path: Path) -> Path:
    """An empty watchlist.yaml."""
    p = tmp_path / "watchlist.yaml"
    p.write_text("watchlist: []\n")
    return p


def _mock_providers() -> tuple[MagicMock, MagicMock, MagicMock]:
    market = MagicMock()
    market.get_daily_bars.return_value = pd.DataFrame({
        "date": ["2026-02-10", "2026-02-11"],
        "open": [10.0, 10.5],
        "high": [11.0, 11.5],
        "low": [9.5, 10.0],
        "close": [10.5, 11.0],
        "volume": [1000, 1200],
    })

    news = MagicMock()
    news.get_stock_news.return_value = []
    news.get_announcements.return_value = []

    fundamental = MagicMock()
    fundamental.get_valuation.return_value = {
        "pe_ttm": 20.0, "pb": 3.0, "market_cap": 5000.0,
    }
    fundamental.get_financial_indicator.return_value = pd.DataFrame([{
        "roe_ttm": 0.15, "revenue_yoy": 0.10, "profit_yoy": 0.12,
    }])

    return market, news, fundamental


# ---------------------------------------------------------------------------
# get_watchlist
# ---------------------------------------------------------------------------


class TestGetWatchlist:
    def test_loads_from_yaml(self, tmp_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        assert pool.get_watchlist() == ["601899", "600967"]

    def test_empty_yaml(self, empty_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(empty_yaml, db, m, n, f)
        assert pool.get_watchlist() == []

    def test_missing_file(self, tmp_path: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_path / "nonexistent.yaml", db, m, n, f)
        assert pool.get_watchlist() == []

    def test_corrupted_yaml(self, tmp_path: Path, db: Database) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("watchlist: [\n  - symbol: 601899\n  missing close bracket")
        m, n, f = _mock_providers()
        pool = StockPoolManager(bad, db, m, n, f)
        assert pool.get_watchlist() == []


# ---------------------------------------------------------------------------
# add_to_watchlist
# ---------------------------------------------------------------------------


class TestAddToWatchlist:
    def test_adds_and_persists(self, empty_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(empty_yaml, db, m, n, f)
        pool.add_to_watchlist("601899", name="紫金矿业", sector="有色金属")

        assert "601899" in pool.get_watchlist()
        # Verify YAML persistence
        data = yaml.safe_load(empty_yaml.read_text())
        symbols = [e["symbol"] for e in data["watchlist"]]
        assert "601899" in symbols
        assert data["watchlist"][0]["name"] == "紫金矿业"

    def test_duplicate_no_effect(self, tmp_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        pool.add_to_watchlist("601899")  # already exists
        assert pool.get_watchlist().count("601899") == 1

    def test_triggers_data_init(self, empty_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(empty_yaml, db, m, n, f)
        pool.add_to_watchlist("601899", name="紫金矿业")

        m.get_daily_bars.assert_called_once()
        f.get_valuation.assert_called_once_with("601899")
        f.get_financial_indicator.assert_called_once_with("601899")
        n.get_stock_news.assert_called_once_with("601899", limit=20)
        n.get_announcements.assert_called_once_with("601899", limit=20)

    def test_saves_daily_bars_to_db(self, empty_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(empty_yaml, db, m, n, f)
        pool.add_to_watchlist("601899")

        bars = db.load_daily_bars("601899", "2026-01-01", "2026-12-31")
        assert len(bars) == 2

    def test_saves_fundamentals_to_db(self, empty_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(empty_yaml, db, m, n, f)
        pool.add_to_watchlist("601899")

        funds = db.load_fundamentals("601899", "2026-01-01", "2026-12-31")
        assert len(funds) == 1
        assert funds.iloc[0]["pe_ttm"] == 20.0

    def test_init_failure_does_not_block_add(
        self, empty_yaml: Path, db: Database
    ) -> None:
        m, n, f = _mock_providers()
        m.get_daily_bars.side_effect = ConnectionError("network error")
        f.get_valuation.side_effect = ConnectionError("network error")

        pool = StockPoolManager(empty_yaml, db, m, n, f)
        pool.add_to_watchlist("601899")

        # Stock still added despite init failures
        assert "601899" in pool.get_watchlist()
        data = yaml.safe_load(empty_yaml.read_text())
        assert data["watchlist"][0]["symbol"] == "601899"

    def test_add_without_name_sector(self, empty_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(empty_yaml, db, m, n, f)
        pool.add_to_watchlist("601899")

        data = yaml.safe_load(empty_yaml.read_text())
        entry = data["watchlist"][0]
        assert entry["symbol"] == "601899"
        assert "name" not in entry  # not persisted if empty


# ---------------------------------------------------------------------------
# remove_from_watchlist
# ---------------------------------------------------------------------------


class TestRemoveFromWatchlist:
    def test_removes_and_persists(self, tmp_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        pool.remove_from_watchlist("601899")

        assert "601899" not in pool.get_watchlist()
        data = yaml.safe_load(tmp_yaml.read_text())
        symbols = [e["symbol"] for e in data["watchlist"]]
        assert "601899" not in symbols
        assert "600967" in symbols  # other stock remains

    def test_remove_nonexistent_no_error(
        self, tmp_yaml: Path, db: Database
    ) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        pool.remove_from_watchlist("999999")  # no error
        assert len(pool.get_watchlist()) == 2


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_add_remove_round_trip(self, empty_yaml: Path, db: Database) -> None:
        m, n, f = _mock_providers()
        pool = StockPoolManager(empty_yaml, db, m, n, f)

        pool.add_to_watchlist("601899", name="紫金矿业", sector="有色金属")
        pool.add_to_watchlist("600967", name="内蒙一机", sector="军工")
        assert pool.get_watchlist() == ["601899", "600967"]

        # Re-load from YAML
        pool2 = StockPoolManager(empty_yaml, db, m, n, f)
        assert pool2.get_watchlist() == ["601899", "600967"]

        pool2.remove_from_watchlist("601899")
        pool3 = StockPoolManager(empty_yaml, db, m, n, f)
        assert pool3.get_watchlist() == ["600967"]


# ---------------------------------------------------------------------------
# _backfill_missing_data
# ---------------------------------------------------------------------------


class TestBackfillMissingData:
    def test_backfills_when_no_bars(self, tmp_yaml: Path, db: Database) -> None:
        """Stocks in YAML with no DB data should trigger _init_stock_data."""
        m, n, f = _mock_providers()
        # Constructor triggers backfill — both stocks have no bars in DB
        _pool = StockPoolManager(tmp_yaml, db, m, n, f)

        # get_daily_bars called once per stock without data
        assert m.get_daily_bars.call_count == 2

    def test_skips_when_bars_exist(self, tmp_yaml: Path, db: Database) -> None:
        """Stocks with existing DB data should not trigger backfill."""
        m, n, f = _mock_providers()
        # Pre-populate bars for both stocks with recent data
        recent = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        db.save_daily_bars("601899", pd.DataFrame({
            "date": [recent], "open": [10], "high": [11],
            "low": [9], "close": [10.5], "volume": [1000],
        }))
        db.save_daily_bars("600967", pd.DataFrame({
            "date": [recent], "open": [10], "high": [11],
            "low": [9], "close": [10.5], "volume": [1000],
        }))
        _pool = StockPoolManager(tmp_yaml, db, m, n, f)

        # No backfill calls since both stocks have data
        m.get_daily_bars.assert_not_called()

    def test_backfills_only_missing(self, tmp_yaml: Path, db: Database) -> None:
        """Only stocks without data should trigger backfill."""
        m, n, f = _mock_providers()
        # Pre-populate bars for only one stock with recent data
        recent = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        db.save_daily_bars("601899", pd.DataFrame({
            "date": [recent], "open": [10], "high": [11],
            "low": [9], "close": [10.5], "volume": [1000],
        }))
        _pool = StockPoolManager(tmp_yaml, db, m, n, f)

        # Only 600967 should be backfilled
        assert m.get_daily_bars.call_count == 1

    def test_backfills_stale_data(self, tmp_path: Path, db: Database) -> None:
        """Stocks with data older than 7 days should trigger backfill."""
        p = tmp_path / "watchlist.yaml"
        p.write_text(yaml.dump({"watchlist": [
            {"symbol": "601899", "name": "紫金矿业"},
        ]}, allow_unicode=True))
        m, n, f = _mock_providers()
        # Data from 30 days ago — stale
        db.save_daily_bars("601899", pd.DataFrame({
            "date": ["2025-12-01"], "open": [10], "high": [11],
            "low": [9], "close": [10.5], "volume": [1000],
        }))
        _pool = StockPoolManager(p, db, m, n, f)

        # Should trigger backfill due to stale data
        assert m.get_daily_bars.call_count == 1


# ---------------------------------------------------------------------------
# _filter_stocks
# ---------------------------------------------------------------------------


def _info_df(name: str = "紫金矿业", ipo_date: str = "20100426") -> pd.DataFrame:
    """Build a fake stock_individual_info_em DataFrame."""
    return pd.DataFrame({
        "item": ["股票简称", "上市时间", "总市值", "行业"],
        "value": [name, ipo_date, "100000000000", "有色金属"],
    })


class TestFilterStocks:
    @patch("akshare.stock_individual_info_em")
    def test_normal_stock_passes(self, mock_info: MagicMock, tmp_yaml: Path, db: Database) -> None:
        mock_info.return_value = _info_df("紫金矿业", "20100426")
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        result = pool._filter_stocks(["601899"])
        assert result == ["601899"]

    @patch("akshare.stock_individual_info_em")
    def test_st_stock_excluded(self, mock_info: MagicMock, tmp_yaml: Path, db: Database) -> None:
        mock_info.return_value = _info_df("*ST某某", "20100426")
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        result = pool._filter_stocks(["600123"])
        assert result == []

    @patch("akshare.stock_individual_info_em")
    def test_new_ipo_excluded(self, mock_info: MagicMock, tmp_yaml: Path, db: Database) -> None:
        from datetime import datetime, timedelta
        recent = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        mock_info.return_value = _info_df("新股科技", recent)
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        result = pool._filter_stocks(["688xxx"])
        assert result == []

    @patch("akshare.stock_individual_info_em")
    def test_old_ipo_passes(self, mock_info: MagicMock, tmp_yaml: Path, db: Database) -> None:
        mock_info.return_value = _info_df("老牌股份", "20200101")
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        result = pool._filter_stocks(["600001"])
        assert result == ["600001"]

    @patch("akshare.stock_individual_info_em")
    def test_api_failure_keeps_stock(self, mock_info: MagicMock, tmp_yaml: Path, db: Database) -> None:
        mock_info.side_effect = ConnectionError("network error")
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        with patch("trading_agent.utils.CircuitBreaker", return_value=MagicMock(allow_request=MagicMock(return_value=True), record_success=MagicMock(), record_failure=MagicMock())):
            result = pool._filter_stocks(["601899"])
        assert result == ["601899"]

    @patch("akshare.stock_individual_info_em")
    def test_mixed_filtering(self, mock_info: MagicMock, tmp_yaml: Path, db: Database) -> None:
        """Multiple stocks: one normal, one ST, one new IPO."""
        from datetime import datetime, timedelta
        recent = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")

        def side_effect(symbol: str) -> pd.DataFrame:
            if symbol == "601899":
                return _info_df("紫金矿业", "20100426")
            if symbol == "600123":
                return _info_df("ST退市", "20050101")
            if symbol == "688999":
                return _info_df("新股科技", recent)
            return _info_df()

        mock_info.side_effect = side_effect
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        result = pool._filter_stocks(["601899", "600123", "688999"])
        assert result == ["601899"]


# ---------------------------------------------------------------------------
# sync_trading_calendar
# ---------------------------------------------------------------------------


class TestSyncTradingCalendar:
    @patch("akshare.tool_trade_date_hist_sina")
    def test_syncs_dates(self, mock_cal: MagicMock, tmp_yaml: Path, db: Database) -> None:
        mock_cal.return_value = pd.DataFrame({
            "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06"],
        })
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        count = pool.sync_trading_calendar()

        assert count == 3
        assert db.is_trading_day("2026-01-02") is True
        assert db.is_trading_day("2026-01-04") is False

    @patch("akshare.tool_trade_date_hist_sina")
    def test_empty_returns_zero(self, mock_cal: MagicMock, tmp_yaml: Path, db: Database) -> None:
        mock_cal.return_value = pd.DataFrame()
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        assert pool.sync_trading_calendar() == 0


# ---------------------------------------------------------------------------
# CSI300 constituents
# ---------------------------------------------------------------------------


class TestCSI300:
    @patch("akshare.stock_individual_info_em")
    @patch("akshare.index_stock_cons")
    def test_sync_csi300(
        self, mock_cons: MagicMock, mock_info: MagicMock,
        tmp_yaml: Path, db: Database,
    ) -> None:
        mock_cons.return_value = pd.DataFrame({
            "品种代码": ["600519", "000858"],
            "品种名称": ["贵州茅台", "五粮液"],
            "纳入日期": ["2020-01-01", "2020-01-01"],
        })
        mock_info.side_effect = lambda symbol: pd.DataFrame({
            "item": ["股票简称", "上市时间", "行业"],
            "value": [
                "茅台" if symbol == "600519" else "五粮液",
                "20010827",
                "白酒" if symbol == "600519" else "食品饮料",
            ],
        })
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        count = pool.sync_csi300_constituents()
        assert count == 2

    def test_get_csi300_stocks_from_db(self, tmp_yaml: Path, db: Database) -> None:
        db.save_index_constituents([
            {"symbol": "600519", "name": "贵州茅台", "index_code": "000300",
             "sector": "白酒", "updated_date": "2026-02-20"},
            {"symbol": "000858", "name": "五粮液", "index_code": "000300",
             "sector": "食品饮料", "updated_date": "2026-02-20"},
        ])
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        stocks = pool._get_csi300_stocks()
        assert set(stocks) == {"600519", "000858"}

    def test_get_factor_universe(self, tmp_yaml: Path, db: Database) -> None:
        """Factor universe = watchlist + CSI300, deduplicated."""
        db.save_index_constituents([
            {"symbol": "601899", "name": "紫金矿业", "index_code": "000300",
             "sector": "有色金属", "updated_date": "2026-02-20"},
            {"symbol": "600519", "name": "贵州茅台", "index_code": "000300",
             "sector": "白酒", "updated_date": "2026-02-20"},
        ])
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        universe = pool.get_factor_universe()
        # watchlist: [601899, 600967], csi300: [601899, 600519]
        # merged: [601899, 600967, 600519] — 601899 deduped
        assert universe == ["601899", "600967", "600519"]

    def test_get_name_sector_maps_unified(self, tmp_yaml: Path, db: Database) -> None:
        """DB provides CSI300 names/sectors; YAML fills gaps for watchlist-only stocks."""
        db.save_index_constituents([
            {"symbol": "601899", "name": "紫金矿业DB", "index_code": "000300",
             "sector": "工业金属", "updated_date": "2026-02-20"},
            {"symbol": "600519", "name": "贵州茅台", "index_code": "000300",
             "sector": "白酒", "updated_date": "2026-02-20"},
        ])
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        name_map, sector_map = pool.get_name_sector_maps()
        # 601899 in both DB and YAML — DB wins
        assert name_map["601899"] == "紫金矿业DB"
        assert sector_map["601899"] == "工业金属"
        # 600519 only in DB
        assert name_map["600519"] == "贵州茅台"
        # 600967 only in YAML — YAML fills gap
        assert name_map["600967"] == "内蒙一机"
        assert sector_map["600967"] == "军工"

    def test_get_name_sector_maps_empty_db(self, tmp_yaml: Path, db: Database) -> None:
        """When DB has no constituents, falls back to YAML only."""
        m, n, f = _mock_providers()
        pool = StockPoolManager(tmp_yaml, db, m, n, f)
        name_map, sector_map = pool.get_name_sector_maps()
        assert name_map == {"601899": "紫金矿业", "600967": "内蒙一机"}
        assert sector_map == {"601899": "有色金属", "600967": "军工"}
