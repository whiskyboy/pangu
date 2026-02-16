"""Tests for AkShareMarketDataProvider — mocked AkShare calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.data.market import (
    AkShareMarketDataProvider,
    BaoStockMarketDataProvider,
    CircuitBreaker,
    _retry_call,
    _to_bs_code,
)
from trading_agent.data.storage import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Database:
    d = Database(":memory:")
    d.init_tables()
    return d


def _fake_hist_df() -> pd.DataFrame:
    """Simulate ak.stock_zh_a_hist() return with Chinese columns."""
    return pd.DataFrame({
        "日期": ["2026-01-02", "2026-01-03"],
        "开盘": [10.0, 10.5],
        "收盘": [10.8, 11.2],
        "最高": [11.0, 11.5],
        "最低": [9.5, 10.0],
        "成交量": [100000, 120000],
        "成交额": [1_080_000, 1_344_000],
        "振幅": [1.5, 1.4],
        "涨跌幅": [0.8, 3.7],
        "涨跌额": [0.08, 0.4],
        "换手率": [1.0, 1.2],
    })



# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_initially_closed(self) -> None:
        cb = CircuitBreaker(threshold=3)
        assert cb.is_open is False

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(threshold=3, cooldown=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True

    def test_resets_on_success(self) -> None:
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.is_open is False

    def test_closes_after_cooldown(self) -> None:
        cb = CircuitBreaker(threshold=2, cooldown=0.0)
        cb.record_failure()
        cb.record_failure()
        # cooldown=0 → should close immediately
        assert cb.is_open is False

    def test_resets_counter_after_cooldown(self) -> None:
        """After cooldown expires, a single failure should NOT re-open the circuit."""
        cb = CircuitBreaker(threshold=2, cooldown=0.0)
        cb.record_failure()
        cb.record_failure()
        # cooldown expired → circuit closed, counter reset
        assert cb.is_open is False
        # One more failure should NOT re-open (counter was reset to 0)
        cb.record_failure()
        assert cb.is_open is False


# ---------------------------------------------------------------------------
# _retry_call
# ---------------------------------------------------------------------------


class TestRetryCall:
    def test_success_first_try(self) -> None:
        assert _retry_call(lambda: 42) == 42

    def test_retries_on_failure(self) -> None:
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("fail")
            return "ok"

        result = _retry_call(flaky, max_retries=3, backoff_base=0.01)
        assert result == "ok"

    def test_exhausted_retries(self) -> None:
        with pytest.raises(ValueError, match="always"):
            _retry_call(
                lambda: (_ for _ in ()).throw(ValueError("always")),
                max_retries=2,
                backoff_base=0.01,
            )

    def test_circuit_open_skips(self) -> None:
        cb = CircuitBreaker(threshold=1, cooldown=999)
        cb.record_failure()
        with pytest.raises(RuntimeError, match="Circuit breaker"):
            _retry_call(lambda: 1, circuit=cb)


# ---------------------------------------------------------------------------
# AkShareMarketDataProvider — get_daily_bars
# ---------------------------------------------------------------------------


class TestGetDailyBars:
    @patch("akshare.stock_zh_a_hist")
    def test_fetches_and_cleans(self, mock_hist: MagicMock) -> None:
        mock_hist.return_value = _fake_hist_df()
        provider = AkShareMarketDataProvider(request_interval=0)
        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")

        assert list(df.columns) == [
            "date", "open", "high", "low", "close", "volume", "amount", "adj_factor",
        ]
        assert len(df) == 2
        assert df.iloc[0]["close"] == pytest.approx(10.8)
        mock_hist.assert_called_once()

    @patch("akshare.stock_zh_a_hist")
    def test_caches_to_sqlite(self, mock_hist: MagicMock, db: Database) -> None:
        mock_hist.return_value = _fake_hist_df()
        provider = AkShareMarketDataProvider(storage=db, request_interval=0)

        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")
        assert len(df) == 2

        # Sync log should be updated
        assert db.get_last_sync_date("600519", "daily_bars") == "2026-01-03"

    @patch("akshare.stock_zh_a_hist")
    def test_serves_from_cache(self, mock_hist: MagicMock, db: Database) -> None:
        """When cache is up-to-date, AkShare should NOT be called."""
        mock_hist.return_value = _fake_hist_df()
        provider = AkShareMarketDataProvider(storage=db, request_interval=0)

        # First call populates cache
        provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")
        mock_hist.reset_mock()

        # Second call — cache hit
        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")
        assert len(df) == 2
        mock_hist.assert_not_called()

    @patch("akshare.stock_zh_a_hist")
    def test_empty_response_falls_back_to_cache(
        self, mock_hist: MagicMock, db: Database
    ) -> None:
        """If AkShare returns empty, fall back to whatever is in cache."""
        # Seed cache
        mock_hist.return_value = _fake_hist_df()
        provider = AkShareMarketDataProvider(storage=db, request_interval=0)
        provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")

        # Now request a later range — AkShare returns empty
        mock_hist.return_value = pd.DataFrame()
        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-05")
        # Should return cached data
        assert len(df) == 2

    @patch("akshare.stock_zh_a_hist")
    def test_no_storage_empty_response(self, mock_hist: MagicMock) -> None:
        mock_hist.return_value = pd.DataFrame()
        provider = AkShareMarketDataProvider(request_interval=0)
        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")
        assert df.empty

    @patch("akshare.stock_zh_a_hist")
    def test_bad_columns_raises(self, mock_hist: MagicMock) -> None:
        """_clean_hist should raise if critical columns are missing."""
        mock_hist.return_value = pd.DataFrame({"foo": [1], "bar": [2]})
        provider = AkShareMarketDataProvider(request_interval=0)
        with pytest.raises(ValueError, match="missing critical columns"):
            provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")


# ---------------------------------------------------------------------------
# AkShareMarketDataProvider — get_realtime_quote
# ---------------------------------------------------------------------------


def _fake_bid_ask_df(symbol: str = "600519") -> pd.DataFrame:
    """Simulate ak.stock_bid_ask_em() return for a single stock."""
    prices = {"600519": 1800.0, "000858": 150.0}
    p = prices.get(symbol, 100.0)
    return pd.DataFrame({
        "item": ["最新", "涨幅", "总手", "金额", "最高", "最低", "今开", "量比",
                 "buy_1", "buy_1_vol", "sell_1", "sell_1_vol"],
        "value": [p, 1.5, 5_000_000, p * 5_000_000, p * 1.01, p * 0.99,
                  p * 1.005, 1.2, p - 0.1, 100, p + 0.1, 200],
    })


class TestGetRealtimeQuote:
    @patch("akshare.stock_bid_ask_em")
    def test_single_symbol(self, mock_bid: MagicMock) -> None:
        mock_bid.return_value = _fake_bid_ask_df("600519")
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_realtime_quote(["600519"])
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "600519"
        assert df.iloc[0]["price"] == pytest.approx(1800.0)
        assert "change_pct" in df.columns
        assert "volume" in df.columns

    @patch("akshare.stock_bid_ask_em")
    def test_multiple_symbols(self, mock_bid: MagicMock) -> None:
        mock_bid.side_effect = [_fake_bid_ask_df("600519"), _fake_bid_ask_df("000858")]
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_realtime_quote(["600519", "000858"])
        assert len(df) == 2
        assert df.iloc[1]["price"] == pytest.approx(150.0)

    @patch("akshare.stock_bid_ask_em")
    def test_empty_bid_ask_returns_symbol_only(self, mock_bid: MagicMock) -> None:
        mock_bid.return_value = pd.DataFrame()
        provider = AkShareMarketDataProvider(request_interval=0)
        df = provider.get_realtime_quote(["600519"])
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "600519"


# ---------------------------------------------------------------------------
# AkShareMarketDataProvider — get_stock_list
# ---------------------------------------------------------------------------


def _fake_sh_df() -> pd.DataFrame:
    return pd.DataFrame({"证券代码": ["600519"], "证券简称": ["贵州茅台"],
                         "证券全称": [""], "公司简称": [""], "公司全称": [""], "上市日期": [""]})


def _fake_sz_df() -> pd.DataFrame:
    return pd.DataFrame({"板块": ["主板"], "A股代码": ["000858"], "A股简称": ["五粮液"],
                         "A股上市日期": [""], "A股总股本": [""], "A股流通股本": [""], "所属行业": [""]})


def _fake_bj_df() -> pd.DataFrame:
    return pd.DataFrame({"证券代码": ["836149"], "证券简称": ["旭杰科技"],
                         "总股本": [""], "流通股本": [""], "上市日期": [""],
                         "所属行业": [""], "地区": [""], "报告日期": [""]})


class TestGetStockList:
    @patch("akshare.stock_info_bj_name_code")
    @patch("akshare.stock_info_sz_name_code")
    @patch("akshare.stock_info_sh_name_code")
    def test_returns_all_exchanges(
        self, mock_sh: MagicMock, mock_sz: MagicMock, mock_bj: MagicMock,
    ) -> None:
        # SH is called twice (主板A股 + 科创板)
        mock_sh.side_effect = [_fake_sh_df(), _fake_sh_df()]
        mock_sz.return_value = _fake_sz_df()
        mock_bj.return_value = _fake_bj_df()
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_stock_list()
        assert set(df.columns) == {"symbol", "name"}
        assert len(df) == 4  # 1 SH main + 1 SH STAR + 1 SZ + 1 BJ

    @patch("akshare.stock_info_bj_name_code")
    @patch("akshare.stock_info_sz_name_code")
    @patch("akshare.stock_info_sh_name_code")
    def test_partial_failure_still_returns(
        self, mock_sh: MagicMock, mock_sz: MagicMock, mock_bj: MagicMock,
    ) -> None:
        """If BJ API fails, SH+SZ stocks are still returned."""
        mock_sh.side_effect = [_fake_sh_df(), _fake_sh_df()]
        mock_sz.return_value = _fake_sz_df()
        mock_bj.side_effect = ConnectionError("timeout")
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_stock_list()
        assert len(df) == 3  # SH main + SH STAR + SZ

    @patch("akshare.stock_info_bj_name_code")
    @patch("akshare.stock_info_sz_name_code")
    @patch("akshare.stock_info_sh_name_code")
    def test_all_fail_returns_empty(
        self, mock_sh: MagicMock, mock_sz: MagicMock, mock_bj: MagicMock,
    ) -> None:
        mock_sh.side_effect = ConnectionError("fail")
        mock_sz.side_effect = ConnectionError("fail")
        mock_bj.side_effect = ConnectionError("fail")
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_stock_list()
        assert df.empty
        assert set(df.columns) == {"symbol", "name"}


# ---------------------------------------------------------------------------
# International market data — M2.3
# ---------------------------------------------------------------------------


def _fake_us_index_df() -> pd.DataFrame:
    """Simulate ak.index_us_stock_sina() return — historical daily OHLCV."""
    return pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "open": [5100.0, 5150.0, 5180.0],
        "high": [5150.0, 5200.0, 5220.0],
        "low": [5080.0, 5130.0, 5170.0],
        "close": [5140.0, 5190.0, 5200.0],
        "volume": [3e9, 3.1e9, 2.9e9],
        "amount": [0, 0, 0],
    })


def _fake_hk_index_df() -> pd.DataFrame:
    """Simulate ak.stock_hk_index_spot_sina() — 38 HK indices (sina format)."""
    return pd.DataFrame({
        "代码": ["HSI", "HSTECH", "OTHER"],
        "名称": ["恒生指数", "恒生科技指数", "其他指数"],
        "最新价": [26700.0, 5360.0, 1000.0],
        "涨跌额": [130.0, 50.0, 10.0],
        "涨跌幅": [0.52, 0.13, 0.05],
        "昨收": [26570.0, 5310.0, 990.0],
        "今开": [26500.0, 5340.0, 990.0],
        "最高": [26750.0, 5380.0, 1010.0],
        "最低": [26400.0, 5300.0, 980.0],
    })


def _fake_commodity_df() -> pd.DataFrame:
    """Simulate ak.futures_foreign_commodity_realtime() — batch realtime."""
    return pd.DataFrame({
        "名称": ["COMEX黄金", "COMEX白银", "NYMEX原油", "COMEX铜", "新加坡铁矿石"],
        "最新价": [5040.0, 77.1, 62.5, 583.4, 96.7],
        "人民币报价": [0, 0, 0, 0, 0],
        "涨跌额": [-6.5, -0.9, -0.3, -2.8, -0.1],
        "涨跌幅": [-0.13, -1.11, -0.47, -0.48, -0.10],
        "开盘价": [5050.0, 77.6, 63.0, 583.9, 96.9],
        "最高价": [5074.0, 78.4, 63.1, 585.9, 97.8],
        "最低价": [4982.0, 74.6, 62.4, 579.7, 96.2],
        "昨日结算价": [5046.0, 78.0, 62.8, 586.3, 96.8],
        "持仓量": [0, 0, 0, 0, 0],
        "买价": [5036.0, 77.0, 62.5, 583.1, 96.7],
        "卖价": [5037.0, 77.0, 62.5, 583.2, 96.8],
        "行情时间": ["17:54:21", "17:54:15", "17:54:26", "17:53:42", "17:54:30"],
        "日期": ["2026-02-16", "2026-02-16", "2026-02-16", "2026-02-16", "2026-02-16"],
    })


class TestGetUSIndices:
    @patch("akshare.index_us_stock_sina")
    def test_fetches_three_indices(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _fake_us_index_df()
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_us_indices()
        assert len(df) == 3
        assert set(df["symbol"]) == {"SPX", "DJI", "IXIC"}
        assert "change_pct" in df.columns
        assert "close" in df.columns
        assert "source" in df.columns
        assert df.iloc[0]["source"] == "us_index"

    @patch("akshare.index_us_stock_sina")
    def test_persists_to_storage(self, mock_api: MagicMock, db: Database) -> None:
        mock_api.return_value = _fake_us_index_df()
        provider = AkShareMarketDataProvider(storage=db, request_interval=0)

        df = provider.get_us_indices()
        assert len(df) == 3
        stored = db.load_global_snapshots(source="us_index")
        assert len(stored) == 3

    @patch("akshare.index_us_stock_sina")
    def test_partial_failure(self, mock_api: MagicMock) -> None:
        """If one index fails, others are still returned."""
        mock_api.side_effect = [
            _fake_us_index_df(),
            ConnectionError("timeout"),
            _fake_us_index_df(),
        ]
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_us_indices()
        assert len(df) == 2  # 2 succeeded, 1 failed


class TestGetHKIndices:
    @patch("akshare.stock_hk_index_spot_sina")
    def test_filters_target_indices(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _fake_hk_index_df()
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_hk_indices()
        assert len(df) == 2
        assert set(df["symbol"]) == {"HSI", "HSTECH"}
        assert df.iloc[0]["source"] == "hk_index"
        assert df.iloc[0]["change_pct"] == pytest.approx(0.52)

    @patch("akshare.stock_hk_index_spot_sina")
    def test_persists_to_storage(self, mock_api: MagicMock, db: Database) -> None:
        mock_api.return_value = _fake_hk_index_df()
        provider = AkShareMarketDataProvider(storage=db, request_interval=0)

        provider.get_hk_indices()
        stored = db.load_global_snapshots(source="hk_index")
        assert len(stored) == 2

    @patch("akshare.stock_hk_index_spot_sina")
    def test_api_failure_returns_empty(self, mock_api: MagicMock) -> None:
        mock_api.side_effect = ConnectionError("timeout")
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_hk_indices()
        assert df.empty


class TestGetCommodityFutures:
    @patch("akshare.futures_foreign_commodity_realtime")
    def test_fetches_five_commodities(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _fake_commodity_df()
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_commodity_futures()
        assert len(df) == 5
        assert set(df["symbol"]) == {"GC", "SI", "CL", "HG", "FEF"}
        assert df.iloc[0]["source"] == "commodity"
        assert "change_pct" in df.columns

    @patch("akshare.futures_foreign_commodity_realtime")
    def test_persists_to_storage(self, mock_api: MagicMock, db: Database) -> None:
        mock_api.return_value = _fake_commodity_df()
        provider = AkShareMarketDataProvider(storage=db, request_interval=0)

        provider.get_commodity_futures()
        stored = db.load_global_snapshots(source="commodity")
        assert len(stored) == 5

    @patch("akshare.futures_foreign_commodity_realtime")
    def test_api_failure_returns_empty(self, mock_api: MagicMock) -> None:
        mock_api.side_effect = ConnectionError("fail")
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_commodity_futures()
        assert df.empty


class TestGetGlobalSnapshot:
    @patch("akshare.futures_foreign_commodity_realtime")
    @patch("akshare.stock_hk_index_spot_sina")
    @patch("akshare.index_us_stock_sina")
    def test_aggregates_all_sources(
        self, mock_us: MagicMock, mock_hk: MagicMock, mock_commodity: MagicMock,
    ) -> None:
        mock_us.return_value = _fake_us_index_df()
        mock_hk.return_value = _fake_hk_index_df()
        mock_commodity.return_value = _fake_commodity_df()
        provider = AkShareMarketDataProvider(request_interval=0)

        df = provider.get_global_snapshot()
        assert len(df) == 10  # 3 US + 2 HK + 5 commodities
        assert set(df["source"]) == {"us_index", "hk_index", "commodity"}

    @patch("akshare.futures_foreign_commodity_realtime")
    @patch("akshare.stock_hk_index_spot_sina")
    @patch("akshare.index_us_stock_sina")
    def test_persists_all_to_storage(
        self, mock_us: MagicMock, mock_hk: MagicMock, mock_commodity: MagicMock,
        db: Database,
    ) -> None:
        mock_us.return_value = _fake_us_index_df()
        mock_hk.return_value = _fake_hk_index_df()
        mock_commodity.return_value = _fake_commodity_df()
        provider = AkShareMarketDataProvider(storage=db, request_interval=0)

        provider.get_global_snapshot()
        all_stored = db.load_latest_global_snapshots()
        assert len(all_stored) == 10


# ---------------------------------------------------------------------------
# _to_bs_code helper
# ---------------------------------------------------------------------------


class TestToBsCode:
    def test_sh_main_board(self) -> None:
        assert _to_bs_code("600519") == "sh.600519"

    def test_sh_star_market(self) -> None:
        assert _to_bs_code("688699") == "sh.688699"

    def test_sz_main_board(self) -> None:
        assert _to_bs_code("000001") == "sz.000001"

    def test_sz_chinext(self) -> None:
        assert _to_bs_code("300750") == "sz.300750"

    def test_already_prefixed(self) -> None:
        assert _to_bs_code("sh.600519") == "sh.600519"

    def test_bj_exchange_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            _to_bs_code("830799")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            _to_bs_code("")

    def test_invalid_code_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            _to_bs_code("ABC")


# ---------------------------------------------------------------------------
# BaoStockMarketDataProvider
# ---------------------------------------------------------------------------


def _fake_bs_result(fields: list[str], rows: list[list]) -> MagicMock:
    """Create a mock BaoStock ResultData object."""
    rs = MagicMock()
    rs.error_code = "0"
    rs.fields = fields
    rs._rows = iter(rows)
    rs._has_next = True

    def _next():
        try:
            rs._current = next(rs._rows)
            return True
        except StopIteration:
            return False

    rs.next = _next
    rs.get_row_data = lambda: rs._current
    return rs


class TestBaoStockDailyBars:
    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_fetches_and_cleans(self, mock_query: MagicMock, mock_login: MagicMock) -> None:
        mock_login.return_value = MagicMock(error_code="0")
        fields = ["date", "code", "open", "high", "low", "close",
                   "preclose", "volume", "amount", "adjustflag", "pctChg"]
        rows = [
            ["2026-01-02", "sh.600519", "10.0", "11.0", "9.5", "10.8",
             "10.0", "100000", "1080000", "2", "8.00"],
            ["2026-01-03", "sh.600519", "10.5", "11.5", "10.0", "11.2",
             "10.8", "120000", "1344000", "2", "3.70"],
        ]
        mock_query.return_value = _fake_bs_result(fields, rows)

        provider = BaoStockMarketDataProvider()
        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")

        assert list(df.columns) == [
            "date", "open", "high", "low", "close", "volume", "amount", "adj_factor",
        ]
        assert len(df) == 2
        assert df.iloc[0]["close"] == pytest.approx(10.8)
        mock_query.assert_called_once()
        provider.close()

    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_saves_to_storage(self, mock_query: MagicMock, mock_login: MagicMock,
                              db: Database) -> None:
        mock_login.return_value = MagicMock(error_code="0")
        fields = ["date", "code", "open", "high", "low", "close",
                   "preclose", "volume", "amount", "adjustflag", "pctChg"]
        rows = [
            ["2026-01-02", "sh.600519", "10.0", "11.0", "9.5", "10.8",
             "10.0", "100000", "1080000", "2", "8.00"],
        ]
        mock_query.return_value = _fake_bs_result(fields, rows)

        provider = BaoStockMarketDataProvider(storage=db)
        provider.get_daily_bars("600519", "2026-01-02", "2026-01-02")

        # Sync log should show source=baostock, status=fallback
        last = db.get_last_sync_date("600519", "daily_bars")
        assert last == "2026-01-02"
        provider.close()

    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_empty_result(self, mock_query: MagicMock, mock_login: MagicMock) -> None:
        mock_login.return_value = MagicMock(error_code="0")
        mock_query.return_value = _fake_bs_result(
            ["date", "code", "open", "high", "low", "close",
             "preclose", "volume", "amount", "adjustflag", "pctChg"],
            [],
        )

        provider = BaoStockMarketDataProvider()
        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")
        assert df.empty
        provider.close()

    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_query_error(self, mock_query: MagicMock, mock_login: MagicMock) -> None:
        mock_login.return_value = MagicMock(error_code="0")
        rs = MagicMock()
        rs.error_code = "10001"
        rs.error_msg = "query failed"
        mock_query.return_value = rs

        provider = BaoStockMarketDataProvider()
        with pytest.raises(RuntimeError, match="query_history_k_data_plus failed"):
            provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")
        provider.close()


class TestBaoStockStockList:
    @patch("baostock.login")
    @patch("baostock.query_all_stock")
    def test_fetches_a_shares(self, mock_all: MagicMock, mock_login: MagicMock) -> None:
        mock_login.return_value = MagicMock(error_code="0")
        fields = ["code", "tradeStatus", "code_name"]
        rows = [
            ["sh.600519", "1", "贵州茅台"],
            ["sz.000001", "1", "平安银行"],
            ["sz.300750", "1", "宁德时代"],
            ["sh.688699", "1", "明微电子"],
            ["sh.000001", "1", "上证指数"],  # index — should be filtered out
            ["sz.200002", "1", "万科B"],     # B-share — should be filtered out
            ["sz.000002", "0", "万科A"],     # suspended — should be filtered out
        ]
        mock_all.return_value = _fake_bs_result(fields, rows)

        provider = BaoStockMarketDataProvider()
        df = provider.get_stock_list()

        assert len(df) == 4
        assert set(df["symbol"]) == {"600519", "000001", "300750", "688699"}
        provider.close()


class TestBaoStockNotImplemented:
    def test_realtime_quote(self) -> None:
        provider = BaoStockMarketDataProvider.__new__(BaoStockMarketDataProvider)
        with pytest.raises(NotImplementedError):
            provider.get_realtime_quote(["600519"])

    def test_us_indices(self) -> None:
        provider = BaoStockMarketDataProvider.__new__(BaoStockMarketDataProvider)
        with pytest.raises(NotImplementedError):
            provider.get_us_indices()

    def test_hk_indices(self) -> None:
        provider = BaoStockMarketDataProvider.__new__(BaoStockMarketDataProvider)
        with pytest.raises(NotImplementedError):
            provider.get_hk_indices()

    def test_commodity_futures(self) -> None:
        provider = BaoStockMarketDataProvider.__new__(BaoStockMarketDataProvider)
        with pytest.raises(NotImplementedError):
            provider.get_commodity_futures()

    def test_global_snapshot(self) -> None:
        provider = BaoStockMarketDataProvider.__new__(BaoStockMarketDataProvider)
        with pytest.raises(NotImplementedError):
            provider.get_global_snapshot()


class TestBaoStockSession:
    @patch("baostock.login")
    @patch("baostock.logout")
    def test_login_and_close(self, mock_logout: MagicMock, mock_login: MagicMock) -> None:
        mock_login.return_value = MagicMock(error_code="0")
        provider = BaoStockMarketDataProvider()
        provider._ensure_login()
        assert provider._logged_in is True
        provider.close()
        mock_logout.assert_called_once()
        assert provider._logged_in is False

    @patch("baostock.login")
    def test_login_failure(self, mock_login: MagicMock) -> None:
        mock_login.return_value = MagicMock(error_code="10002", error_msg="auth failed")
        provider = BaoStockMarketDataProvider()
        with pytest.raises(RuntimeError, match="BaoStock login failed"):
            provider._ensure_login()


# ---------------------------------------------------------------------------
# AkShare → BaoStock fallback integration
# ---------------------------------------------------------------------------


class TestAkShareFallback:
    @patch("akshare.stock_zh_a_hist")
    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_fallback_on_akshare_exception(
        self, mock_bs_query: MagicMock, mock_bs_login: MagicMock,
        mock_ak_hist: MagicMock,
    ) -> None:
        """When AkShare raises, BaoStock fallback should be used."""
        mock_ak_hist.side_effect = ConnectionError("AkShare down")
        mock_bs_login.return_value = MagicMock(error_code="0")
        fields = ["date", "code", "open", "high", "low", "close",
                   "preclose", "volume", "amount", "adjustflag", "pctChg"]
        rows = [
            ["2026-01-02", "sh.600519", "10.0", "11.0", "9.5", "10.8",
             "10.0", "100000", "1080000", "2", "8.00"],
        ]
        mock_bs_query.return_value = _fake_bs_result(fields, rows)

        fallback = BaoStockMarketDataProvider()
        provider = AkShareMarketDataProvider(request_interval=0, fallback=fallback)
        # Override circuit breaker to avoid long retries
        provider._circuit = CircuitBreaker(threshold=1, cooldown=0)

        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-02")
        assert len(df) == 1
        assert df.iloc[0]["close"] == pytest.approx(10.8)
        mock_bs_query.assert_called_once()
        fallback.close()

    @patch("akshare.stock_zh_a_hist")
    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_fallback_on_empty_response(
        self, mock_bs_query: MagicMock, mock_bs_login: MagicMock,
        mock_ak_hist: MagicMock,
    ) -> None:
        """When AkShare returns empty, BaoStock fallback should be used."""
        mock_ak_hist.return_value = pd.DataFrame()
        mock_bs_login.return_value = MagicMock(error_code="0")
        fields = ["date", "code", "open", "high", "low", "close",
                   "preclose", "volume", "amount", "adjustflag", "pctChg"]
        rows = [
            ["2026-01-02", "sh.600519", "10.0", "11.0", "9.5", "10.8",
             "10.0", "100000", "1080000", "2", "8.00"],
        ]
        mock_bs_query.return_value = _fake_bs_result(fields, rows)

        fallback = BaoStockMarketDataProvider()
        provider = AkShareMarketDataProvider(request_interval=0, fallback=fallback)

        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-02")
        assert len(df) == 1
        fallback.close()

    @patch("akshare.stock_zh_a_hist")
    def test_no_fallback_returns_empty(self, mock_ak_hist: MagicMock) -> None:
        """Without fallback, AkShare failure should return empty DataFrame."""
        mock_ak_hist.side_effect = ConnectionError("AkShare down")
        provider = AkShareMarketDataProvider(request_interval=0)
        # Override to speed up
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)

        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-02")
        assert df.empty

    @patch("akshare.stock_zh_a_hist")
    @patch("baostock.login")
    @patch("baostock.query_history_k_data_plus")
    def test_both_fail_falls_back_to_cache(
        self, mock_bs_query: MagicMock, mock_bs_login: MagicMock,
        mock_ak_hist: MagicMock, db: Database,
    ) -> None:
        """When both AkShare and BaoStock fail, cached data should be returned."""
        # Seed cache via a successful AkShare call
        mock_ak_hist.return_value = _fake_hist_df()
        fallback = BaoStockMarketDataProvider(storage=db)
        provider = AkShareMarketDataProvider(
            storage=db, request_interval=0, fallback=fallback,
        )
        provider.get_daily_bars("600519", "2026-01-02", "2026-01-03")

        # Now both fail
        mock_ak_hist.return_value = pd.DataFrame()
        mock_bs_login.return_value = MagicMock(error_code="0")
        mock_bs_query.return_value = _fake_bs_result(
            ["date", "code", "open", "high", "low", "close",
             "preclose", "volume", "amount", "adjustflag", "pctChg"],
            [],
        )

        df = provider.get_daily_bars("600519", "2026-01-02", "2026-01-05")
        # Should return 2 rows from cache
        assert len(df) == 2
        fallback.close()
