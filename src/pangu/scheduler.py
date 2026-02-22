"""PanGu scheduler — APScheduler-based task orchestration.

Registers 6 cron tasks (times configurable via [scheduler] in settings.toml):
  T1 sync_global_market   — trading days (default 08:00)
  T2 poll_news             — hourly (default 07:00-20:00)
  T3 sync_domestic_market  — trading days (default 15:30)
  T4 generate_signals      — trading days (default 08:15)
  T5 sync_reference_data   — monthly (default 1st at 06:00)
  T6 verify_signals        — trading days (default 16:00)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from pangu.data.fundamental import AkShareFundamentalProvider
from pangu.data.market import AkShareMarketDataProvider
from pangu.data.news import AkShareNewsDataProvider
from pangu.data.stock_pool import StockPoolManager
from pangu.data.storage import Database
from pangu.factor.fundamental import FundamentalFactorEngine
from pangu.factor.macro import MacroFactorEngine
from pangu.factor.technical import PandasTAFactorEngine
from pangu.notification import NotificationManager
from pangu.strategy.factor import MultiFactorStrategy
from pangu.strategy.llm import LLMJudgeEngineImpl
from pangu.tasks import (
    sync_global_market,
    poll_news,
    sync_domestic_market,
    generate_signals,
    sync_reference_data,
    verify_signals,
)
from pangu.tz import now as _now
from pangu.tz import today_str

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> tuple[int, int]:
    """Parse 'HH:MM' → (hour, minute)."""
    parts = s.split(":")
    return int(parts[0]), int(parts[1])


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

    async def alert(self, msg: str) -> None:
        """Send a plain-text alert via notification channels."""
        try:
            await self.notif_manager.notify_text(f"⚠️ {msg}")
        except Exception:  # noqa: BLE001
            _logger = logging.getLogger(__name__)
            _logger.warning("Failed to send alert: %s", msg, exc_info=True)


class TradingScheduler:
    """APScheduler-based orchestrator for the 5 trading tasks."""

    def __init__(
        self,
        components: Components,
        timezone: str = "Asia/Shanghai",
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._c = components
        self._tz = timezone
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._sched_cfg = scheduler_cfg or {}
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
        """Register the 5 cron tasks from config (with sensible defaults)."""
        cfg = self._sched_cfg

        # T1: sync global market
        t1_h, t1_m = _parse_time(cfg.get("global_market_sync_time", "08:00"))
        self._scheduler.add_job(
            self._run_if_trading_day, "cron",
            args=[self.sync_global_market],
            hour=t1_h, minute=t1_m, id="t1_sync_global_market",
            name="T1 全球市场同步",
        )

        # T2: poll news
        t2_start, _ = _parse_time(cfg.get("news_poll_start_time", "07:00"))
        t2_end, _ = _parse_time(cfg.get("news_poll_end_time", "20:00"))
        self._scheduler.add_job(
            self.poll_news, "cron",
            hour=f"{t2_start}-{t2_end}", minute=0, id="t2_poll_news",
            name="T2 快讯采集",
        )

        # T3: sync domestic market
        t3_h, t3_m = _parse_time(cfg.get("domestic_market_sync_time", "15:30"))
        self._scheduler.add_job(
            self._run_if_trading_day, "cron",
            args=[self.sync_domestic_market],
            hour=t3_h, minute=t3_m, id="t3_sync_domestic_market",
            name="T3 国内行情同步",
        )

        # T4: generate signals
        t4_h, t4_m = _parse_time(cfg.get("signal_generate_time", "08:15"))
        self._scheduler.add_job(
            self._run_if_trading_day, "cron",
            args=[self.generate_signals],
            hour=t4_h, minute=t4_m, id="t4_generate_signals",
            name="T4 信号生成",
        )

        # T5: sync reference data — 1st of each month at 06:00
        cal_day = cfg.get("calendar_sync_day", 1)
        self._scheduler.add_job(
            self.sync_reference_data, "cron",
            day=cal_day, hour=6, minute=0, id="t5_sync_reference_data",
            name="T5 基础数据同步",
        )

        # T6: verify signals — after T3 (default 16:00)
        t6_h, t6_m = _parse_time(cfg.get("signal_verify_time", "16:00"))
        self._scheduler.add_job(
            self._run_if_trading_day, "cron",
            args=[self.verify_signals],
            hour=t6_h, minute=t6_m, id="t6_verify_signals",
            name="T6 信号验证",
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
        await sync_global_market(self._c)

    async def poll_news(self) -> None:
        await poll_news(self._c)

    async def sync_domestic_market(self) -> None:
        await sync_domestic_market(self._c)

    async def generate_signals(self) -> None:
        await generate_signals(self._c)

    async def sync_reference_data(self) -> None:
        await sync_reference_data(self._c)

    async def verify_signals(self) -> None:
        await verify_signals(self._c)

    # ------------------------------------------------------------------
    # Manual trigger (for CLI / first-run)
    # ------------------------------------------------------------------

    async def run_once(self) -> None:
        """Run all tasks sequentially (for manual / first-run use).

        Order follows data dependencies:
          T5 calendar → T3 domestic (K-lines + fundamentals) →
          T1 global → T2 news → T4 signals

        T5 is skipped if index constituents were synced within the last 7 days.
        """
        logger.info("=== run_once: executing all tasks ===")

        # Skip T5 if index constituents recently synced
        skip_t5 = False
        try:
            rows = self._c.db.load_all_index_constituents()
            if rows and isinstance(rows, list) and len(rows) > 0:
                import datetime as _dt_mod
                latest = max(r.get("updated_date", "") for r in rows)
                latest_date = _dt_mod.date.fromisoformat(latest)
                days_ago = (_now().date() - latest_date).days
                if days_ago <= 7:
                    logger.info("[T5] Index constituents synced %d day(s) ago, skipping", days_ago)
                    skip_t5 = True
        except Exception:  # noqa: BLE001
            pass
        if not skip_t5:
            await self.sync_reference_data()

        await self.sync_domestic_market()
        await self.sync_global_market()
        await self.poll_news()
        await self.generate_signals()
        await self.verify_signals()
        logger.info("=== run_once complete ===")

