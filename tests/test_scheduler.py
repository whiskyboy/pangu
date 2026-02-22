"""Tests for TradingScheduler (M5.2)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from pangu.scheduler import Components, TradingScheduler


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
    stock_pool.get_all_symbols.return_value = ["600519"]
    from pangu.models import StockMeta
    stock_pool.get_stock_metadata.return_value = {"600519": StockMeta(name="贵州茅台", sector="白酒")}
    stock_pool.sync_trading_calendar.return_value = 0
    stock_pool.sync_csi300_constituents = MagicMock(return_value=0)

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
        assert len(jobs) == 6

    def test_job_ids(self, scheduler: TradingScheduler) -> None:
        job_ids = {j.id for j in scheduler._scheduler.get_jobs()}
        expected = {
            "t1_sync_global_market",
            "t2_poll_news",
            "t3_sync_domestic_market",
            "t4_generate_signals",
            "t5_sync_reference_data",
            "t6_verify_signals",
        }
        assert job_ids == expected

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
# T1: Sync global market
# ---------------------------------------------------------------------------


class TestT1:
    @pytest.mark.asyncio
    async def test_fetches_snapshot(self, scheduler: TradingScheduler) -> None:
        await scheduler.sync_global_market()
        scheduler._c.market.get_global_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_computes_macro_factors(self, scheduler: TradingScheduler) -> None:
        await scheduler.sync_global_market()
        scheduler._c.macro_engine.compute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_failure(self, scheduler: TradingScheduler) -> None:
        scheduler._c.market.get_global_snapshot.side_effect = RuntimeError("network")
        await scheduler.sync_global_market()  # should not raise
        # macro_engine still called with empty DataFrame
        scheduler._c.macro_engine.compute.assert_called_once()


# ---------------------------------------------------------------------------
# T2: Poll news
# ---------------------------------------------------------------------------


class TestT2:
    @pytest.mark.asyncio
    async def test_fetches_telegraph(self, scheduler: TradingScheduler) -> None:
        await scheduler.poll_news()
        scheduler._c.news.get_latest_news.assert_called_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_cleans_old_news(self, scheduler: TradingScheduler) -> None:
        scheduler._c.db.cleanup_old_news.return_value = 5
        await scheduler.poll_news()
        scheduler._c.db.cleanup_old_news.assert_called_once_with(30)

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
        scheduler._c.stock_pool.get_all_symbols.assert_called_once()
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
        from pangu.models import Action, SignalStatus, TradeSignal
        from pangu.tz import now

        sig = TradeSignal(
            timestamp=now(), symbol="600519", name="贵州茅台",
            action=Action.HOLD, signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0, price=1850.0, confidence=0.6,
            source="llm_judge", reason="test",
        )
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[sig])
        scheduler._c.db.load_factor_pool_previous_day = MagicMock(
            return_value=pd.DataFrame(columns=["symbol", "score", "rank"]),
        )
        scheduler._c.db.load_recent_news = MagicMock(return_value=[])

        await scheduler.generate_signals()

        scheduler._c.stock_pool.get_all_symbols.assert_called()
        scheduler._c.db.save_trade_signal.assert_called_once()
        scheduler._c.factor_strategy.generate_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_reads_telegraph_from_db(self, scheduler: TradingScheduler) -> None:
        """T4 should read telegraph from DB (T2 accumulated), not API."""
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[])
        scheduler._c.db.load_factor_pool_previous_day = MagicMock(
            return_value=pd.DataFrame(columns=["symbol", "score", "rank"]),
        )
        scheduler._c.db.load_recent_news = MagicMock(return_value=[])

        await scheduler.generate_signals()

        scheduler._c.db.load_recent_news.assert_called_once_with(hours=24)

    @pytest.mark.asyncio
    async def test_uses_factor_signals_for_candidates(self, scheduler: TradingScheduler) -> None:
        """LLM candidates = factor_signals symbols + watchlist, not full pool."""
        from pangu.models import Action, SignalStatus, TradeSignal
        from pangu.tz import now

        # Factor strategy returns pool of 3 stocks but only 1 signal
        buy_sig = TradeSignal(
            timestamp=now(), symbol="000001", name="平安银行",
            action=Action.BUY, signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0, price=10.0, confidence=0.8,
            source="factor", reason="top",
        )
        scheduler._c.factor_strategy.generate_signals.return_value = (
            pd.DataFrame({
                "symbol": ["000001", "000002", "000003"],
                "score": [0.9, 0.7, 0.5],
                "rank": [1, 2, 3],
            }),
            [buy_sig],  # only 000001 has a factor signal
        )
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[])
        scheduler._c.db.load_factor_pool_previous_day = MagicMock(
            return_value=pd.DataFrame(columns=["symbol", "score", "rank"]),
        )
        scheduler._c.db.load_recent_news = MagicMock(return_value=[])

        await scheduler.generate_signals()

        # build_evidence_pool called with factor signal (000001) + watchlist (600519)
        call_args = scheduler._c.judge_engine.build_evidence_pool.call_args
        candidate_symbols = call_args[0][0]
        assert "000001" in candidate_symbols  # factor signal
        assert "600519" in candidate_symbols  # watchlist
        assert "000002" not in candidate_symbols  # not selected by factor strategy
        assert "000003" not in candidate_symbols  # not selected by factor strategy

    @pytest.mark.asyncio
    async def test_signal_status_new_entry(self, scheduler: TradingScheduler) -> None:
        """Stock in current pool but not in previous → NEW_ENTRY."""
        from pangu.models import Action, SignalStatus, TradeSignal
        from pangu.tz import now

        sig = TradeSignal(
            timestamp=now(), symbol="600519", name="贵州茅台",
            action=Action.BUY, signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0, price=1850.0, confidence=0.8,
            source="llm_judge", reason="test",
        )
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[sig])
        # Previous pool is empty → all current stocks are NEW_ENTRY
        scheduler._c.db.load_factor_pool_previous_day = MagicMock(
            return_value=pd.DataFrame(columns=["symbol", "score", "rank"]),
        )
        scheduler._c.db.load_recent_news = MagicMock(return_value=[])
        # Factor strategy returns 600519 in the pool
        scheduler._c.factor_strategy.generate_signals.return_value = (
            pd.DataFrame({"symbol": ["600519"], "score": [0.8], "rank": [1]}),
            [],
        )

        await scheduler.generate_signals()

        saved_signal = scheduler._c.db.save_trade_signal.call_args[0][0]
        assert saved_signal.signal_status == SignalStatus.NEW_ENTRY

    @pytest.mark.asyncio
    async def test_signal_status_sustained(self, scheduler: TradingScheduler) -> None:
        """Stock in both current and previous pool → SUSTAINED."""
        from pangu.models import Action, SignalStatus, TradeSignal
        from pangu.tz import now

        sig = TradeSignal(
            timestamp=now(), symbol="600519", name="贵州茅台",
            action=Action.HOLD, signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0, price=1850.0, confidence=0.7,
            source="llm_judge", reason="test",
        )
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[sig])
        # Previous pool had 600519
        scheduler._c.db.load_factor_pool_previous_day = MagicMock(
            return_value=pd.DataFrame({
                "symbol": ["600519"], "score": [0.75], "rank": [1],
            }),
        )
        scheduler._c.db.load_recent_news = MagicMock(return_value=[])
        scheduler._c.factor_strategy.generate_signals.return_value = (
            pd.DataFrame({"symbol": ["600519"], "score": [0.8], "rank": [1]}),
            [],
        )

        await scheduler.generate_signals()

        saved_signal = scheduler._c.db.save_trade_signal.call_args[0][0]
        assert saved_signal.signal_status == SignalStatus.SUSTAINED
        assert saved_signal.prev_factor_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# T5: Sync reference data
# ---------------------------------------------------------------------------


class TestT5:
    @pytest.mark.asyncio
    async def test_syncs_calendar_and_csi300(self, scheduler: TradingScheduler) -> None:
        scheduler._c.stock_pool.sync_csi300_constituents = MagicMock(return_value=300)
        await scheduler.sync_reference_data()
        scheduler._c.stock_pool.sync_trading_calendar.assert_called_once()
        scheduler._c.stock_pool.sync_csi300_constituents.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_calendar_failure(self, scheduler: TradingScheduler) -> None:
        scheduler._c.stock_pool.sync_trading_calendar.side_effect = RuntimeError("fail")
        scheduler._c.stock_pool.sync_csi300_constituents = MagicMock(return_value=300)
        await scheduler.sync_reference_data()  # should not raise
        # CSI300 sync should still be called
        scheduler._c.stock_pool.sync_csi300_constituents.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_csi300_failure(self, scheduler: TradingScheduler) -> None:
        scheduler._c.stock_pool.sync_csi300_constituents = MagicMock(side_effect=RuntimeError("fail"))
        await scheduler.sync_reference_data()  # should not raise


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_runs_all_tasks(self, scheduler: TradingScheduler) -> None:
        scheduler._c.judge_engine.judge_pool = AsyncMock(return_value=[])
        scheduler._c.db.load_factor_pool_previous_day = MagicMock(
            return_value=pd.DataFrame(columns=["symbol", "score", "rank"]),
        )
        scheduler._c.db.load_recent_news = MagicMock(return_value=[])

        await scheduler.run_once()

        # All 5 tasks should have been called
        scheduler._c.stock_pool.sync_trading_calendar.assert_called()  # T5
        scheduler._c.market.get_daily_bars.assert_called()  # T3
        scheduler._c.fundamental.get_valuation.assert_called()  # T3
        scheduler._c.market.get_global_snapshot.assert_called()  # T1 + T4
        scheduler._c.news.get_latest_news.assert_called()  # T2
        scheduler._c.factor_strategy.generate_signals.assert_called()  # T4
