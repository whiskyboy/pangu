"""Tests for TradingScheduler — registration, trading-day gate, run_signals.

Task-specific behaviour is covered in tests/test_tasks/*. Here we focus on
the scheduler-level concerns:

* All expected jobs are registered with the right ids
* ``_run_if_trading_day`` honours the calendar
* ``run_signals`` invokes T6 (and conditionally T4)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from pangu.scheduler import Components, TradingScheduler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_components() -> Components:
    """Build a Components instance with mocked dependencies (ML path enabled)."""
    db = MagicMock()
    db.is_trading_day.return_value = True
    db.load_all_index_constituents.return_value = []
    db.cleanup_old_news.return_value = 0
    db.cleanup_old_task_runs.return_value = 0
    db.get_latest_bar_date.return_value = None
    db.get_latest_news_timestamp.return_value = None
    db.load_latest_global_snapshots.return_value = pd.DataFrame()

    market = MagicMock()
    market.get_global_snapshot.return_value = pd.DataFrame()
    market.get_daily_bars.return_value = pd.DataFrame()
    market.get_index_daily_bars = MagicMock(return_value=pd.DataFrame())

    news = MagicMock()
    news.get_latest_news.return_value = []

    fundamental = MagicMock()
    fundamental.get_financial_indicator.return_value = pd.DataFrame()
    fundamental.refresh_gross_margin = MagicMock(return_value=(0, 0))
    fundamental.refresh_pub_dates = MagicMock(return_value=(0, 0))

    stock_pool = MagicMock()
    stock_pool.get_all_symbols.return_value = ["600519"]
    stock_pool.sync_trading_calendar.return_value = 0
    stock_pool.sync_index_constituents.return_value = 0

    tech_engine = MagicMock()
    fund_engine = MagicMock()
    judge_engine = MagicMock()

    notif_manager = MagicMock()
    notif_manager.notify_text = AsyncMock(return_value={"FakeChannel": True})
    notif_manager.notify_markdown = AsyncMock(return_value={"FakeChannel": True})

    ml_strategy = MagicMock()
    portfolio_state = MagicMock()
    portfolio_state.load.return_value = None

    return Components(
        db=db,
        market=market,
        news=news,
        fundamental=fundamental,
        stock_pool=stock_pool,
        tech_engine=tech_engine,
        fund_engine=fund_engine,
        judge_engine=judge_engine,
        notif_manager=notif_manager,
        ml_strategy=ml_strategy,
        ml_enabled=True,
        portfolio_state=portfolio_state,
    )


@pytest.fixture
def scheduler(mock_components: Components) -> TradingScheduler:
    return TradingScheduler(mock_components, timezone="Asia/Shanghai")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestJobRegistration:
    def test_all_jobs_registered(self, scheduler: TradingScheduler) -> None:
        # T1-T6: data sync (T1-T4) → model (T5) → signal (T6)
        job_ids = {j.id for j in scheduler._scheduler.get_jobs()}
        assert job_ids == {
            "t1_sync_reference_data",
            "t2_poll_news",
            "t3_sync_domestic_market",
            "t4_sync_global_market",
            "t5_update_model",
            "t6_generate_signals",
        }

    def test_t5_omitted_when_ml_not_configured(self, mock_components) -> None:
        # T5 is gated by ml_enabled (not ml_strategy) so a fresh deploy
        # with ml.enabled=true but no models still schedules T5.
        mock_components.ml_strategy = None
        mock_components.ml_enabled = False
        sched = TradingScheduler(mock_components, timezone="Asia/Shanghai")
        job_ids = {j.id for j in sched._scheduler.get_jobs()}
        assert "t5_update_model" not in job_ids

    def test_t5_scheduled_when_ml_enabled_but_no_models(self, mock_components) -> None:
        # Bootstrap scenario: ml_enabled=True, but models haven't been
        # trained yet (ml_strategy is None). T5 must still be scheduled
        # so the monthly cron can produce the first set of models.
        mock_components.ml_strategy = None
        mock_components.ml_enabled = True
        sched = TradingScheduler(mock_components, timezone="Asia/Shanghai")
        job_ids = {j.id for j in sched._scheduler.get_jobs()}
        assert "t5_update_model" in job_ids

    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, scheduler: TradingScheduler) -> None:
        scheduler.start()
        assert scheduler._scheduler.running is True
        scheduler.shutdown()
        await asyncio.sleep(0.1)
        assert scheduler._scheduler.running is False


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
# Smoke tests for each wrapper — they just delegate to the underlying task
# ---------------------------------------------------------------------------


class TestTaskWrappers:
    @pytest.mark.asyncio
    async def test_t1_syncs_calendar_and_constituents(
        self,
        scheduler: TradingScheduler,
    ) -> None:
        await scheduler.sync_reference_data()
        scheduler._c.stock_pool.sync_trading_calendar.assert_called_once()
        scheduler._c.stock_pool.sync_index_constituents.assert_called_once()

    @pytest.mark.asyncio
    async def test_t2_polls_news_and_cleans_up(self, scheduler: TradingScheduler) -> None:
        await scheduler.poll_news()
        scheduler._c.news.get_latest_news.assert_called_once_with(limit=50)
        scheduler._c.db.cleanup_old_news.assert_called_once_with(30)
        scheduler._c.db.cleanup_old_task_runs.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_t3_syncs_bars_and_fundamentals(self, scheduler: TradingScheduler) -> None:
        await scheduler.sync_domestic_market()
        scheduler._c.market.get_daily_bars.assert_called()
        scheduler._c.fundamental.get_financial_indicator.assert_called()

    @pytest.mark.asyncio
    async def test_t4_calls_market_snapshot(self, scheduler: TradingScheduler) -> None:
        await scheduler.sync_global_market()
        scheduler._c.market.get_global_snapshot.assert_called_once()


# ---------------------------------------------------------------------------
# run_signals (manual T6 trigger)
# ---------------------------------------------------------------------------


class TestRunSignals:
    @pytest.mark.asyncio
    async def test_runs_t4_when_snapshot_missing(self, scheduler: TradingScheduler) -> None:
        scheduler._c.db.load_latest_global_snapshots.return_value = pd.DataFrame()
        scheduler.sync_global_market = AsyncMock()
        scheduler.generate_signals = AsyncMock()

        await scheduler.run_signals()

        scheduler.sync_global_market.assert_awaited_once()
        scheduler.generate_signals.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_t4_when_snapshot_fresh(self, scheduler: TradingScheduler) -> None:
        from pangu.tz import today_str

        scheduler._c.db.load_latest_global_snapshots.return_value = pd.DataFrame(
            {"date": [today_str()], "symbol": ["^GSPC"], "close": [4000.0]}
        )
        scheduler.sync_global_market = AsyncMock()
        scheduler.generate_signals = AsyncMock()

        await scheduler.run_signals()

        scheduler.sync_global_market.assert_not_awaited()
        scheduler.generate_signals.assert_awaited_once()
