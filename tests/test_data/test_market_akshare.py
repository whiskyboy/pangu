"""Tests for AkShareMarketDataProvider — mocked AkShare calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.data.market import (
    AkShareMarketDataProvider,
    CircuitBreaker,
    _retry_call,
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
# International stubs (deferred to M2.3)
# ---------------------------------------------------------------------------


class TestInternationalStubs:
    def test_stubs_return_empty(self) -> None:
        provider = AkShareMarketDataProvider.__new__(AkShareMarketDataProvider)
        provider._ak = MagicMock()
        provider._interval = 0
        provider._last_call = 0
        provider._circuit = CircuitBreaker()

        assert provider.get_us_indices().empty
        assert provider.get_hk_indices().empty
        assert provider.get_commodity_futures().empty
        assert provider.get_global_snapshot().empty
