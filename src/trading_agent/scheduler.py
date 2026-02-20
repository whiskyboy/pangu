"""TradingAgent scheduler — APScheduler-based task orchestration.

Registers 5 cron tasks:
  T1 sync_global_market   — 08:00 trading days
  T2 poll_news             — 07:00-20:00 hourly
  T3 sync_domestic_market  — 15:30 trading days
  T4 generate_signals      — 08:15 trading days
  T5 sync_reference_data   — 1st of each month
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from trading_agent.data.fundamental import AkShareFundamentalProvider
from trading_agent.data.market import AkShareMarketDataProvider
from trading_agent.data.news import AkShareNewsDataProvider
from trading_agent.data.stock_pool import StockPoolManager
from trading_agent.data.storage import Database
from trading_agent.factor.fundamental import FundamentalFactorEngine
from trading_agent.factor.macro import MacroFactorEngine
from trading_agent.factor.technical import PandasTAFactorEngine
from trading_agent.notification import NotificationManager
from trading_agent.strategy.factor_strategy import MultiFactorStrategy
from trading_agent.strategy.llm_engine import LLMJudgeEngineImpl
from trading_agent.tasks import (
    t1_global_market,
    t2_news,
    t3_domestic_market,
    t4_signals,
    t5_reference,
)
from trading_agent.tz import now as _now
from trading_agent.tz import today_str

logger = logging.getLogger(__name__)


@dataclass
class Components:
    """All injected components for task execution."""

    db: Database
    market: AkShareMarketDataProvider
    news: AkShareNewsDataProvider
    fundamental: AkShareFundamentalProvider
    stock_pool: StockPoolManager
    tech_engine: PandasTAFactorEngine
    fund_engine: FundamentalFactorEngine
    macro_engine: MacroFactorEngine
    factor_strategy: MultiFactorStrategy
    judge_engine: LLMJudgeEngineImpl
    notif_manager: NotificationManager
    watchlist_path: str = "config/watchlist.yaml"


class TradingScheduler:
    """APScheduler-based orchestrator for the 5 trading tasks."""

    def __init__(self, components: Components, timezone: str = "Asia/Shanghai") -> None:
        self._c = components
        self._tz = timezone
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._register_jobs()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler (non-blocking)."""
        self._scheduler.start()
        logger.info("TradingScheduler started (%d jobs)", len(self._scheduler.get_jobs()))

    def shutdown(self) -> None:
        """Gracefully shut down the scheduler."""
        self._scheduler.shutdown(wait=False)
        logger.info("TradingScheduler shut down")

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        """Register the 5 cron tasks."""
        # T1: sync global market — 08:00 on trading days
        self._scheduler.add_job(
            self._run_if_trading_day, "cron",
            args=[self.sync_global_market],
            hour=8, minute=0, id="t1_sync_global_market",
            name="T1 全球市场同步",
        )
        # T2: poll news — every hour 07:00–20:00
        self._scheduler.add_job(
            self.poll_news, "cron",
            hour="7-20", minute=0, id="t2_poll_news",
            name="T2 快讯采集",
        )
        # T3: sync domestic market — 15:30 on trading days
        self._scheduler.add_job(
            self._run_if_trading_day, "cron",
            args=[self.sync_domestic_market],
            hour=15, minute=30, id="t3_sync_domestic_market",
            name="T3 国内行情同步",
        )
        # T4: generate signals — 08:15 on trading days
        self._scheduler.add_job(
            self._run_if_trading_day, "cron",
            args=[self.generate_signals],
            hour=8, minute=15, id="t4_generate_signals",
            name="T4 信号生成",
        )
        # T5: sync reference data — 1st of each month at 06:00
        self._scheduler.add_job(
            self.sync_reference_data, "cron",
            day=1, hour=6, minute=0, id="t5_sync_reference_data",
            name="T5 基础数据同步",
        )

    # ------------------------------------------------------------------
    # Trading day gate
    # ------------------------------------------------------------------

    async def _run_if_trading_day(self, task_fn: Any) -> None:
        """Run *task_fn* only if today is a trading day."""
        today = today_str()
        if not self._c.db.is_trading_day(today):
            logger.info("Skipping %s — %s is not a trading day",
                        task_fn.__name__, today)
            return
        await task_fn()

    # ------------------------------------------------------------------
    # T1–T5: Thin wrappers delegating to task modules
    # ------------------------------------------------------------------

    async def sync_global_market(self) -> None:
        await t1_global_market.sync_global_market(self._c)

    async def poll_news(self) -> None:
        await t2_news.poll_news(self._c)

    async def sync_domestic_market(self) -> None:
        await t3_domestic_market.sync_domestic_market(self._c)

    async def generate_signals(self) -> None:
        await t4_signals.generate_signals(self._c)

    async def sync_reference_data(self) -> None:
        await t5_reference.sync_reference_data(self._c)

    # ------------------------------------------------------------------
    # Manual trigger (for CLI / first-run)
    # ------------------------------------------------------------------

    async def run_once(self) -> None:
        """Run all tasks sequentially (for manual / first-run use).

        Order follows data dependencies:
          T5 calendar → T3 domestic (K-lines + fundamentals) →
          T1 global → T2 news → T4 signals

        T5 is skipped if CSI300 constituents were synced within the last 7 days.
        """
        logger.info("=== run_once: executing all tasks ===")

        # Skip T5 if CSI300 recently synced
        skip_t5 = False
        try:
            rows = self._c.db.load_index_constituents("000300")
            if rows and isinstance(rows, list) and len(rows) > 0:
                import datetime as _dt_mod
                latest = max(r.get("updated_date", "") for r in rows)
                latest_date = _dt_mod.date.fromisoformat(latest)
                days_ago = (_now().date() - latest_date).days
                if days_ago <= 7:
                    logger.info("[T5] CSI300 synced %d day(s) ago, skipping", days_ago)
                    skip_t5 = True
        except Exception:  # noqa: BLE001
            pass
        if not skip_t5:
            await self.sync_reference_data()

        await self.sync_domestic_market()
        await self.sync_global_market()
        await self.poll_news()
        await self.generate_signals()
        logger.info("=== run_once complete ===")

