"""TradingAgent scheduler — APScheduler-based task orchestration.

Registers 5 cron tasks:
  T1 sync_global_market   — 08:00 trading days
  T2 poll_news             — 07:00-20:00 hourly
  T3 sync_domestic_market  — 15:30 trading days
  T4 generate_signals      — 08:15 trading days
  T5 sync_calendar         — 1st of each month
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from trading_agent.data.fundamental import AkShareFundamentalProvider
from trading_agent.data.market import AkShareMarketDataProvider
from trading_agent.data.news import AkShareNewsDataProvider
from trading_agent.data.stock_pool import StockPoolManager
from trading_agent.data.storage import Database
from trading_agent.factor.fundamental import FundamentalFactorEngine
from trading_agent.factor.macro import MacroFactorEngine
from trading_agent.factor.technical import PandasTAFactorEngine
from trading_agent.models import Action, TradeSignal
from trading_agent.notification import NotificationManager
from trading_agent.strategy.factor_strategy import MultiFactorStrategy
from trading_agent.strategy.llm_engine import LLMJudgeEngineImpl
from trading_agent.tz import today_str

logger = logging.getLogger(__name__)

_ACTION_EMOJI = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}
_ACTION_ORDER = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}


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
    # Cached data shared between tasks (T1 → T4)
    _global_snapshot: pd.DataFrame = field(default_factory=pd.DataFrame)
    _macro_factors: dict[str, float] = field(default_factory=dict)


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

    @property
    def running(self) -> bool:
        return self._scheduler.running

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
        # T5: sync calendar — 1st of each month at 06:00
        self._scheduler.add_job(
            self.sync_calendar, "cron",
            day=1, hour=6, minute=0, id="t5_sync_calendar",
            name="T5 交易日历同步",
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
    # T1: Sync global market
    # ------------------------------------------------------------------

    async def sync_global_market(self) -> None:
        """Fetch global market snapshot + compute macro factors."""
        logger.info("[T1] Syncing global market...")
        try:
            snapshot = self._c.market.get_global_snapshot()
            self._c._global_snapshot = snapshot
            logger.info("[T1] Global snapshot: %d rows", len(snapshot))
        except Exception:  # noqa: BLE001
            logger.warning("[T1] Global snapshot failed", exc_info=True)
            self._c._global_snapshot = pd.DataFrame()

        self._c._macro_factors = self._c.macro_engine.compute(
            self._c._global_snapshot,
        )
        logger.info("[T1] Macro factors: %s",
                    {k: round(v, 4) for k, v in self._c._macro_factors.items()})
        logger.info("[T1] Done")

    # ------------------------------------------------------------------
    # T2: Poll news
    # ------------------------------------------------------------------

    async def poll_news(self) -> None:
        """Fetch latest telegraph news and store to DB."""
        logger.info("[T2] Polling news...")
        try:
            items = self._c.news.get_latest_news(limit=50)
            logger.info("[T2] Telegraph: %d items fetched", len(items))
        except Exception:  # noqa: BLE001
            logger.warning("[T2] News polling failed", exc_info=True)
        logger.info("[T2] Done")

    # ------------------------------------------------------------------
    # T3: Sync domestic market
    # ------------------------------------------------------------------

    async def sync_domestic_market(self) -> None:
        """Sync daily K-lines + fundamentals for watchlist stocks."""
        logger.info("[T3] Syncing domestic market...")
        c = self._c
        pool = c.stock_pool.get_active_pool()
        today = today_str()

        # Daily bars
        ok, fail = 0, 0
        for symbol in pool:
            try:
                bars = c.market.get_daily_bars(symbol, "2025-06-01", today)
                if bars is not None and not bars.empty:
                    ok += 1
                else:
                    fail += 1
            except Exception:  # noqa: BLE001
                fail += 1
                logger.warning("[T3] %s: daily bars failed", symbol, exc_info=True)
        logger.info("[T3] Daily bars: %d ok, %d failed", ok, fail)

        # Fundamentals
        ok, fail = 0, 0
        for symbol in pool:
            try:
                c.fundamental.get_valuation(symbol)
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
                logger.warning("[T3] %s: fundamentals failed", symbol, exc_info=True)
        logger.info("[T3] Fundamentals: %d ok, %d failed", ok, fail)
        logger.info("[T3] Done")

    # ------------------------------------------------------------------
    # T4: Generate signals
    # ------------------------------------------------------------------

    async def generate_signals(self) -> None:
        """Full signal pipeline: factors → evidence → LLM judge → push."""
        logger.info("[T4] Generating signals...")
        c = self._c
        today = today_str()

        # 1. Get stock pool
        watchlist = c.stock_pool.get_watchlist()
        pool = c.stock_pool.get_active_pool()
        logger.info("[T4] Pool: %d stocks, watchlist: %d", len(pool), len(watchlist))

        # 2. Fetch bars + compute technical factors
        tech_df: dict[str, pd.DataFrame] = {}
        for symbol in pool:
            try:
                bars = c.market.get_daily_bars(symbol, "2025-06-01", today)
                if bars is not None and not bars.empty:
                    tech_df[symbol] = c.tech_engine.compute(bars)
            except Exception:  # noqa: BLE001
                logger.warning("[T4] %s: tech failed", symbol, exc_info=True)
        logger.info("[T4] Tech factors: %d/%d", len(tech_df), len(pool))

        # 3. Fetch fundamentals + compute fundamental factors
        fund_rows: list[dict] = []
        for symbol in pool:
            try:
                val = c.fundamental.get_valuation(symbol)
                val["symbol"] = symbol
                fund_rows.append(val)
            except Exception:  # noqa: BLE001
                logger.warning("[T4] %s: fundamentals failed", symbol, exc_info=True)
        fund_raw = pd.DataFrame(fund_rows)
        if not fund_raw.empty and "symbol" in fund_raw.columns:
            fund_raw = fund_raw.set_index("symbol")
        fund_df = c.fund_engine.compute(fund_raw)

        # 4. Global snapshot + macro (use cached from T1, or fetch fresh)
        global_snapshot = self._c._global_snapshot
        macro_factors = self._c._macro_factors
        if global_snapshot.empty or not macro_factors:
            logger.info("[T4] No cached global data, fetching fresh...")
            try:
                global_snapshot = c.market.get_global_snapshot()
                macro_factors = c.macro_engine.compute(global_snapshot)
            except Exception:  # noqa: BLE001
                logger.warning("[T4] Global snapshot failed", exc_info=True)
                global_snapshot = pd.DataFrame()
                macro_factors = c.macro_engine.compute(global_snapshot)

        # 5. Fetch telegraph
        telegraph = c.news.get_latest_news(limit=50)
        logger.info("[T4] Telegraph: %d items", len(telegraph))

        # 6. Factor ranking
        prev_pool = c.db.load_factor_pool_latest()
        pool_df, _factor_signals = c.factor_strategy.generate_signals(
            tech_df, fund_df, macro_factors, prev_pool=prev_pool,
        )
        logger.info("[T4] Factor pool: %d stocks", len(pool_df))

        if not pool_df.empty:
            c.db.save_factor_pool(today, pool_df)

        # 7. Load name + sector map
        name_map, sector_map = _load_watchlist_maps(c.watchlist_path)

        # 8. Build factor matrix
        factor_matrix = c.factor_strategy._build_factor_matrix(
            pool, tech_df, fund_df, macro_factors,
            {s: sector_map.get(s, "") for s in pool},
        )

        # 9. Build LLM candidates (factor pool + watchlist, deduplicated)
        candidate_symbols = list(dict.fromkeys(
            list(pool_df["symbol"]) + watchlist if not pool_df.empty else watchlist
        ))

        candidates: list[dict] = []
        for sym in candidate_symbols:
            row = pool_df[pool_df["symbol"] == sym] if not pool_df.empty else pd.DataFrame()
            f_score = float(row["score"].iloc[0]) if not row.empty else 0.5
            f_rank = int(row["rank"].iloc[0]) if not row.empty else len(candidate_symbols)
            f_details = (
                factor_matrix.loc[sym].to_dict()
                if sym in factor_matrix.index else {}
            )

            try:
                s_news = c.news.get_stock_news(sym, limit=10)
            except Exception:  # noqa: BLE001
                s_news = []
            try:
                s_anns = c.news.get_announcements(sym, limit=5)
            except Exception:  # noqa: BLE001
                s_anns = []

            bars = tech_df.get(sym)
            price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else 0.0

            candidates.append({
                "symbol": sym,
                "name": name_map.get(sym, sym),
                "factor_score": f_score,
                "factor_rank": f_rank,
                "factor_details": f_details,
                "stock_news": s_news,
                "announcements": s_anns,
                "price": price,
            })

        logger.info("[T4] LLM candidates: %d stocks", len(candidates))

        # 10. LLM comprehensive judge
        signals: list[TradeSignal] = await c.judge_engine.judge_pool(
            candidates, telegraph=telegraph, global_market=global_snapshot,
        )
        logger.info("[T4] Signals: %d", len(signals))
        _print_signal_summary(signals)

        # 11. Save signals + push
        for signal in signals:
            c.db.save_trade_signal(signal)

        watchlist_set = set(watchlist)
        to_push = [s for s in signals if s.action != Action.HOLD or s.symbol in watchlist_set]
        to_push.sort(key=lambda s: (_ACTION_ORDER.get(s.action, 9), -(s.factor_score or 0)))

        for signal in to_push:
            try:
                result = await c.notif_manager.notify(signal)
                if result:
                    logger.info("[T4] Push %s: %s", signal.symbol, result)
            except Exception:  # noqa: BLE001
                logger.warning("[T4] Push failed for %s", signal.symbol, exc_info=True)

        logger.info("[T4] Done — %d signals, %d pushed", len(signals), len(to_push))

    # ------------------------------------------------------------------
    # T5: Sync calendar
    # ------------------------------------------------------------------

    async def sync_calendar(self) -> None:
        """Sync A-share trading calendar."""
        logger.info("[T5] Syncing trading calendar...")
        try:
            count = self._c.stock_pool.sync_trading_calendar()
            logger.info("[T5] %d new dates synced", count)
        except Exception:  # noqa: BLE001
            logger.warning("[T5] Calendar sync failed", exc_info=True)
        logger.info("[T5] Done")

    # ------------------------------------------------------------------
    # Manual trigger (for CLI / first-run)
    # ------------------------------------------------------------------

    async def run_once(self) -> None:
        """Run all tasks sequentially (for manual / first-run use).

        Order follows data dependencies:
          T5 calendar → T3 domestic (K-lines + fundamentals) →
          T1 global → T2 news → T4 signals
        """
        logger.info("=== run_once: executing all tasks ===")
        await self.sync_calendar()
        await self.sync_domestic_market()
        await self.sync_global_market()
        await self.poll_news()
        await self.generate_signals()
        logger.info("=== run_once complete ===")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_watchlist_maps(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Load name and sector maps from watchlist YAML."""
    name_map: dict[str, str] = {}
    sector_map: dict[str, str] = {}
    wl_path = Path(path)
    if wl_path.exists():
        wl_data = yaml.safe_load(wl_path.read_text()) or {}
        for item in wl_data.get("watchlist", []):
            sym = item["symbol"]
            name_map[sym] = item.get("name", sym)
            sector_map[sym] = item.get("sector", "")
    return name_map, sector_map


def _print_signal_summary(signals: list[TradeSignal]) -> None:
    """Print a human-readable summary of signals."""
    if not signals:
        logger.info("No signals generated")
        return
    print("\n" + "=" * 60)
    print("📊 Signal Summary")
    print("=" * 60)
    for sig in signals:
        emoji = _ACTION_EMOJI.get(sig.action, "?")
        print(
            f"  {emoji} {sig.action.value:4s} | {sig.name} ({sig.symbol}) "
            f"| ¥{sig.price:,.2f} | conf={sig.confidence:.2f} "
            f"| {sig.reason}"
        )
    print("=" * 60 + "\n")
