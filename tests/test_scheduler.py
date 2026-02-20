"""Tests for TradingScheduler (M5.1)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from trading_agent.scheduler import Components, TradingScheduler, _load_watchlist_maps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_components() -> Components:
    """Build a Components instance with all mocked dependencies."""
    db = MagicMock()
    db.is_trading_day.return_value = True
    db.load_factor_pool_latest.return_value = pd.DataFrame()
    db.save_factor_pool = MagicMock()
    db.save_trade_signal = MagicMock()

    market = MagicMock()
    market.get_global_snapshot.return_value = pd.DataFrame(
        {"name": ["S&P500"], "latest_price": [5000.0], "change_pct": [0.5]}
    )
    market.get_daily_bars.return_value = pd.DataFrame({
        "date": ["2026-02-19"],
        "open": [10.0], "high": [11.0], "low": [9.5], "close": [10.5],
        "volume": [1000000],
    })

    news = MagicMock()
    news.get_latest_news.return_value = []
    news.get_stock_news.return_value = []
    news.get_announcements.return_value = []

    fundamental = MagicMock()
    fundamental.get_valuation.return_value = {
        "pe_ttm": 20.0, "pb": 2.0, "roe_ttm": 0.15,
        "market_cap": 1e10,
    }

    stock_pool = MagicMock()
    stock_pool.get_watchlist.return_value = ["600519"]
    stock_pool.get_active_pool.return_value = ["600519"]
    stock_pool.sync_trading_calendar.return_value = 0

    tech_engine = MagicMock()
    tech_engine.compute.return_value = pd.DataFrame({
        "close": [10.5], "rsi_14": [55.0], "macd_hist": [0.1],
    })

    fund_engine = MagicMock()
    fund_engine.compute.return_value = pd.DataFrame(
        {"pe_ttm": [20.0], "pb": [2.0]}, index=["600519"],
    )

    macro_engine = MagicMock()
    macro_engine.compute.return_value = {"global_risk": -0.5, "macro_adj": 0.0}

    factor_strategy = MagicMock()
    factor_strategy.generate_signals.return_value = (
        pd.DataFrame({"symbol": ["600519"], "score": [0.8], "rank": [1]}),
        [],
    )
    factor_strategy._build_factor_matrix.return_value = pd.DataFrame(
        {"rsi_14": [55.0]}, index=["600519"],
    )

    judge_engine = MagicMock()
    judge_engine.judge_pool = AsyncMock(return_value=[])

    notif_manager = MagicMock()
    notif_manager.notify = AsyncMock(return_value={"FeishuNotifier": True})

    return Components(
        db=db,
        market=market,
        news=news,
        fundamental=fundamental,
        stock_pool=stock_pool,
        tech_engine=tech_engine,
        fund_engine=fund_engine,
        macro_engine=macro_engine,
        factor_strategy=factor_strategy,
        judge_engine=judge_engine,
        notif_manager=notif_manager,
        watchlist_path="config/watchlist.yaml",
    )


@pytest.fixture
def scheduler(mock_components: Components) -> TradingScheduler:
    return TradingScheduler(mock_components, timezone="Asia/Shanghai")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestJobRegistration:
    def test_five_jobs_registered(self, scheduler: TradingScheduler) -> None:
        jobs = scheduler._scheduler.get_jobs()
        assert len(jobs) == 5

    def test_job_ids(self, scheduler: TradingScheduler) -> None:
        job_ids = {j.id for j in scheduler._scheduler.get_jobs()}
        expected = {
            "t1_sync_global_market",
            "t2_poll_news",
            "t3_sync_domestic_market",
            "t4_generate_signals",
            "t5_sync_calendar",
        }
        assert job_ids == expected

    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, scheduler: TradingScheduler) -> None:
        scheduler.start()
        assert scheduler.running is True
        scheduler.shutdown()
        await asyncio.sleep(0.1)
        assert scheduler.running is False


# ---------------------------------------------------------------------------
# Trading day gate
# ---------------------------------------------------------------------------


class TestTradingDayGate:
    @pytest.mark.asyncio
    async def test_runs_on_trading_day(self, scheduler: TradingScheduler) -> None:
        scheduler._c.db.is_trading_day.return_value = True
        mock_fn = AsyncMock()
        mock_fn.__name__ = "test_task"
        await scheduler._run_if_trading_day(mock_fn)
        mock_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_on_non_trading_day(self, scheduler: TradingScheduler) -> None:
        scheduler._c.db.is_trading_day.return_value = False
        mock_fn = AsyncMock()
        mock_fn.__name__ = "test_task"
        await scheduler._run_if_trading_day(mock_fn)
        mock_fn.assert_not_awaited()


# ---------------------------------------------------------------------------
# T1: Sync global market
# ---------------------------------------------------------------------------


class TestT1:
    @pytest.mark.asyncio
    async def test_fetches_and_caches_snapshot(self, scheduler: TradingScheduler) -> None:
        await scheduler.sync_global_market()
        scheduler._c.market.get_global_snapshot.assert_called_once()
        assert not scheduler._c._global_snapshot.empty
        assert len(scheduler._c._macro_factors) > 0

    @pytest.mark.asyncio
    async def test_handles_failure(self, scheduler: TradingScheduler) -> None:
        scheduler._c.market.get_global_snapshot.side_effect = RuntimeError("network")
        await scheduler.sync_global_market()
        assert scheduler._c._global_snapshot.empty


# ---------------------------------------------------------------------------
# T2: Poll news
# ---------------------------------------------------------------------------


class TestT2:
    @pytest.mark.asyncio
    async def test_fetches_telegraph(self, scheduler: TradingScheduler) -> None:
        await scheduler.poll_news()
        scheduler._c.news.get_latest_news.assert_called_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_handles_failure(self, scheduler: TradingScheduler) -> None:
        scheduler._c.news.get_latest_news.side_effect = RuntimeError("fail")
        await scheduler.poll_news()  # should not raise


# ---------------------------------------------------------------------------
# T3: Sync domestic market
# ---------------------------------------------------------------------------


class TestT3:
    @pytest.mark.asyncio
    async def test_syncs_bars_and_fundamentals(self, scheduler: TradingScheduler) -> None:
        await scheduler.sync_domestic_market()
        scheduler._c.market.get_daily_bars.assert_called()
        scheduler._c.fundamental.get_valuation.assert_called()

    @pytest.mark.asyncio
    async def test_handles_partial_failure(self, scheduler: TradingScheduler) -> None:
        scheduler._c.market.get_daily_bars.side_effect = RuntimeError("fail")
        await scheduler.sync_domestic_market()  # should not raise


# ---------------------------------------------------------------------------
# T4: Generate signals
# ---------------------------------------------------------------------------


class TestT4:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, scheduler: TradingScheduler) -> None:
        from trading_agent.models import Action, SignalStatus, TradeSignal
        from trading_agent.tz import now

        sig = TradeSignal(
            timestamp=now(), symbol="600519", name="贵州茅台",
            action=Action.HOLD, signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0, price=1850.0, confidence=0.6,
            source="llm_judge", reason="test",
        )
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[sig])

        await scheduler.generate_signals()

        scheduler._c.db.save_trade_signal.assert_called_once()
        scheduler._c.factor_strategy.generate_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_cached_global_data(self, scheduler: TradingScheduler) -> None:
        # Pre-cache global data (as T1 would)
        scheduler._c._global_snapshot = pd.DataFrame({"name": ["test"]})
        scheduler._c._macro_factors = {"global_risk": 0.0}
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[])

        await scheduler.generate_signals()

        # Should NOT fetch global snapshot again
        scheduler._c.market.get_global_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetches_fresh_global_when_empty(self, scheduler: TradingScheduler) -> None:
        scheduler._c._global_snapshot = pd.DataFrame()  # empty
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[])

        await scheduler.generate_signals()

        scheduler._c.market.get_global_snapshot.assert_called()


# ---------------------------------------------------------------------------
# T5: Sync calendar
# ---------------------------------------------------------------------------


class TestT5:
    @pytest.mark.asyncio
    async def test_syncs_calendar(self, scheduler: TradingScheduler) -> None:
        await scheduler.sync_calendar()
        scheduler._c.stock_pool.sync_trading_calendar.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_failure(self, scheduler: TradingScheduler) -> None:
        scheduler._c.stock_pool.sync_trading_calendar.side_effect = RuntimeError("fail")
        await scheduler.sync_calendar()  # should not raise


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_runs_all_tasks(self, scheduler: TradingScheduler) -> None:
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[])

        await scheduler.run_once()

        # All 5 tasks should have been called
        scheduler._c.stock_pool.sync_trading_calendar.assert_called()  # T5
        scheduler._c.market.get_daily_bars.assert_called()  # T3
        scheduler._c.fundamental.get_valuation.assert_called()  # T3
        scheduler._c.market.get_global_snapshot.assert_called()  # T1
        scheduler._c.news.get_latest_news.assert_called()  # T2
        scheduler._c.factor_strategy.generate_signals.assert_called()  # T4


# ---------------------------------------------------------------------------
# Helper: _load_watchlist_maps
# ---------------------------------------------------------------------------


class TestLoadWatchlistMaps:
    def test_loads_from_real_file(self) -> None:
        name_map, sector_map = _load_watchlist_maps("config/watchlist.yaml")
        assert len(name_map) > 0
        # All values should be strings
        for v in name_map.values():
            assert isinstance(v, str)

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        name_map, sector_map = _load_watchlist_maps(str(tmp_path / "nonexistent.yaml"))
        assert name_map == {}
        assert sector_map == {}
