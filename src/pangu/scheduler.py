"""PanGu scheduler — APScheduler-based task orchestration.

Task numbering follows logical dependency (data → model → signals), NOT
scheduling order. Actual daily run order (assuming month start):
  T2 (hourly)  → T5 (02:00) → T1 (06:00, monthly) → T4 (07:00) →
  T6 (08:15)   → T3 (18:00)

Registered cron tasks (times configurable via [scheduler] in settings.toml):
  T1 sync_reference_data    — monthly (default 1st at 06:00)
  T2 poll_news              — hourly (default 00:00-23:00, 24x/day)
  T3 sync_domestic_market   — trading days (default 18:00)
  T4 sync_global_market     — trading days (default 07:00)
  T5 update_model           — monthly (when ml.enabled, default 1st at 02:00)
  T6 generate_signals       — trading days (default 08:15)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from pangu.data.fundamental import CompositeFundamentalProvider
from pangu.data.market import CompositeMarketDataProvider
from pangu.data.news import AkShareNewsDataProvider
from pangu.data.stock_pool import IndexStockPool
from pangu.data.storage import Database
from pangu.factor.fundamental import FundamentalFactorEngine
from pangu.factor.technical import PandasTAFactorEngine
from pangu.notification import NotificationManager
from pangu.strategy.llm import LLMJudgeEngineImpl
from pangu.tasks import (
    generate_signals,
    poll_news,
    sync_domestic_market,
    sync_global_market,
    sync_reference_data,
    update_model,
)
from pangu.tz import today_str

if TYPE_CHECKING:
    from pangu.portfolio import PortfolioState
    from pangu.strategy.ml.ml_strategy import MLScoringStrategy

logger = logging.getLogger(__name__)


def _parse_time(s: str) -> tuple[int, int]:
    """Parse 'HH:MM' → (hour, minute)."""
    parts = s.split(":")
    return int(parts[0]), int(parts[1])


@dataclass
class Components:
    """All injected components for task execution."""

    db: Database
    market: CompositeMarketDataProvider
    news: AkShareNewsDataProvider
    fundamental: CompositeFundamentalProvider
    stock_pool: IndexStockPool
    tech_engine: PandasTAFactorEngine
    fund_engine: FundamentalFactorEngine
    judge_engine: LLMJudgeEngineImpl
    notif_manager: NotificationManager
    ml_strategy: MLScoringStrategy | None = None
    ml_enabled: bool = False
    portfolio_state: PortfolioState | None = field(default=None)
    initial_capital: float = 100_000.0  # for T6 share-size suggestions

    async def alert(self, msg: str) -> None:
        """Send a plain-text alert via notification channels."""
        try:
            await self.notif_manager.notify_text(f"⚠️ {msg}")
        except Exception:  # noqa: BLE001
            _logger = logging.getLogger(__name__)
            _logger.warning("Failed to send alert: %s", msg, exc_info=True)


class TradingScheduler:
    """APScheduler-based orchestrator for the 7 trading tasks."""

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
        """Register the 6 cron tasks from config (with sensible defaults).

        Numbering reflects logical dependency (data → model → signals), not
        scheduling order. Source filenames are kept (named by function).
        """
        cfg = self._sched_cfg

        # T1: sync reference data (calendar + index constituents) — monthly
        # misfire_grace_time=86400: if the monthly trigger is missed (server
        # restart, network outage, etc.), allow catchup within 24h instead of
        # waiting a full month. Avoids cross-year trading_calendar gaps that
        # would freeze T3/T4/T6 via the is_trading_day gate.
        t1_h, t1_m = _parse_time(cfg.get("reference_data_sync_time", "06:00"))
        t1_day = cfg.get("reference_data_sync_day", 1)
        self._scheduler.add_job(
            self.sync_reference_data,
            "cron",
            day=t1_day,
            hour=t1_h,
            minute=t1_m,
            id="t1_sync_reference_data",
            name="T1 参考数据同步",
            misfire_grace_time=86400,
        )

        # T2: poll news — hourly cycle by default (00:00-23:00)
        t2_start, _ = _parse_time(cfg.get("news_poll_start_time", "00:00"))
        t2_end, _ = _parse_time(cfg.get("news_poll_end_time", "23:00"))
        self._scheduler.add_job(
            self.poll_news,
            "cron",
            hour=f"{t2_start}-{t2_end}",
            minute=0,
            id="t2_poll_news",
            name="T2 快讯采集",
        )

        # T3: sync domestic market (K-lines + fundamentals) — trading days
        t3_h, t3_m = _parse_time(cfg.get("domestic_kline_sync_time", "18:00"))
        self._scheduler.add_job(
            self._run_if_trading_day,
            "cron",
            args=[self.sync_domestic_market],
            hour=t3_h,
            minute=t3_m,
            id="t3_sync_domestic_market",
            name="T3 国内行情同步",
        )

        # T4: sync overnight global market — trading days
        t4_h, t4_m = _parse_time(cfg.get("international_data_sync_time", "07:00"))
        self._scheduler.add_job(
            self._run_if_trading_day,
            "cron",
            args=[self.sync_global_market],
            hour=t4_h,
            minute=t4_m,
            id="t4_sync_global_market",
            name="T4 全球行情同步",
        )

        # T5: monthly model update — gated by ml.enabled config so a fresh
        # deployment (ml.enabled=true but no model files yet) can still
        # auto-train on the next month boundary.
        if self._c.ml_enabled:
            t5_h, t5_m = _parse_time(cfg.get("model_training_time", "02:00"))
            t5_day = cfg.get("model_training_day", 1)
            self._scheduler.add_job(
                self.update_model,
                "cron",
                day=t5_day,
                hour=t5_h,
                minute=t5_m,
                id="t5_update_model",
                name="T5 月度模型训练",
                misfire_grace_time=86400,
            )

        # T6: generate signals — trading days
        t6_h, t6_m = _parse_time(cfg.get("signal_generation_time", "08:15"))
        self._scheduler.add_job(
            self._run_if_trading_day,
            "cron",
            args=[self.generate_signals],
            hour=t6_h,
            minute=t6_m,
            id="t6_generate_signals",
            name="T6 信号生成与调仓",
        )

    # ------------------------------------------------------------------
    # Trading day gate
    # ------------------------------------------------------------------

    async def _run_if_trading_day(self, task_fn: Any) -> None:
        """Run *task_fn* only if today is a trading day."""
        today = today_str()
        if not self._c.db.is_trading_day(today):
            logger.info("Skipping %s — %s is not a trading day", task_fn.__name__, today)
            return
        await task_fn()

    # ------------------------------------------------------------------
    # T1–T6: Thin wrappers delegating to task modules
    # ------------------------------------------------------------------

    async def sync_reference_data(self) -> None:
        await sync_reference_data(self._c)

    async def poll_news(self) -> None:
        await poll_news(self._c)

    async def sync_domestic_market(self) -> None:
        await sync_domestic_market(self._c)

    async def sync_global_market(self) -> None:
        await sync_global_market(self._c)

    async def update_model(self) -> None:
        await update_model(self._c)

    async def generate_signals(self) -> None:
        await generate_signals(self._c)

    # ------------------------------------------------------------------
    # Manual trigger
    # ------------------------------------------------------------------

    async def run_signals(self) -> None:
        """Manual trigger for T6 signal generation.

        Auto-runs T4 (international data) if today's global snapshot is
        missing. T3 K-line freshness is only warned about (08:15 cannot
        get same-day K-lines anyway — they're populated at T3's 18:00 cron).
        """
        from pangu.tz import today_str

        today = today_str()
        snaps = self._c.db.load_latest_global_snapshots()
        need_t4 = snaps.empty or (isinstance(snaps["date"].max(), str) and snaps["date"].max() < today)
        if need_t4:
            logger.info("[run signals] Today's global snapshot missing — running T4 first")
            await self.sync_global_market()
        else:
            logger.info("[run signals] Today's global snapshot already present — skipping T4")

        await self.generate_signals()
