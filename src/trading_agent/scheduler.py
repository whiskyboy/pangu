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
from datetime import timedelta
from typing import Any

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from trading_agent.data.fundamental import AkShareFundamentalProvider
from trading_agent.data.market import AkShareMarketDataProvider
from trading_agent.data.news import AkShareNewsDataProvider
from trading_agent.data.stock_pool import StockPoolManager
from trading_agent.data.storage import Database
from trading_agent.factor.fundamental import FundamentalFactorEngine
from trading_agent.factor.macro import MacroFactorEngine
from trading_agent.factor.technical import PandasTAFactorEngine
from trading_agent.models import Action, SignalStatus, TradeSignal
from trading_agent.notification import NotificationManager
from trading_agent.strategy.factor_strategy import MultiFactorStrategy
from trading_agent.strategy.llm_engine import LLMJudgeEngineImpl
from trading_agent.tz import now as _now
from trading_agent.tz import today_str

logger = logging.getLogger(__name__)

_ACTION_EMOJI = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}
_ACTION_ORDER = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
# Lookback window for daily bars (~120 trading days)
_BARS_LOOKBACK_DAYS = 200


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
    # T1: Sync global market
    # ------------------------------------------------------------------

    async def sync_global_market(self) -> None:
        """Fetch global market snapshot + compute macro factors."""
        logger.info("[T1] Syncing global market...")
        try:
            snapshot = self._c.market.get_global_snapshot()
            logger.info("[T1] Global snapshot: %d rows", len(snapshot))
        except Exception:  # noqa: BLE001
            logger.warning("[T1] Global snapshot failed", exc_info=True)
            snapshot = pd.DataFrame()

        macro_factors = self._c.macro_engine.compute(snapshot)
        logger.info("[T1] Macro factors: %s",
                    {k: round(v, 4) for k, v in macro_factors.items()})
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
        try:
            deleted = self._c.db.cleanup_old_news(30)
            if deleted:
                logger.info("[T2] Cleaned up %d old news items", deleted)
        except Exception:  # noqa: BLE001
            logger.warning("[T2] News cleanup failed", exc_info=True)
        logger.info("[T2] Done")

    # ------------------------------------------------------------------
    # T3: Sync domestic market
    # ------------------------------------------------------------------

    async def sync_domestic_market(self) -> None:
        """Sync daily K-lines + fundamentals for watchlist + CSI300."""
        logger.info("[T3] Syncing domestic market...")
        c = self._c
        pool = c.stock_pool.get_factor_universe()
        today = today_str()
        start = (_now() - timedelta(days=_BARS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        total = len(pool)
        logger.info("[T3] Sync universe: %d stocks", total)

        # Daily bars
        ok, fail = 0, 0
        for i, symbol in enumerate(pool, 1):
            try:
                bars = c.market.get_daily_bars(symbol, start, today)
                if bars is not None and not bars.empty:
                    ok += 1
                else:
                    fail += 1
            except Exception:  # noqa: BLE001
                fail += 1
                logger.warning("[T3] %s: daily bars failed", symbol, exc_info=True)
            if i % 50 == 0:
                logger.info("[T3] Daily bars: %d/%d processed", i, total)
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
        start = (_now() - timedelta(days=_BARS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        # 1. Get stock pools
        watchlist = c.stock_pool.get_watchlist()
        factor_universe = c.stock_pool.get_factor_universe()
        logger.info("[T4] Factor universe: %d stocks, watchlist: %d",
                    len(factor_universe), len(watchlist))

        # 2. Compute technical factors for full universe (from DB, T3 synced)
        tech_df: dict[str, pd.DataFrame] = {}
        for symbol in factor_universe:
            try:
                bars = c.market.get_daily_bars(symbol, start, today)
                if bars is not None and not bars.empty:
                    tech_df[symbol] = c.tech_engine.compute(bars)
            except Exception:  # noqa: BLE001
                logger.warning("[T4] %s: tech failed", symbol, exc_info=True)
        logger.info("[T4] Tech factors: %d/%d", len(tech_df), len(factor_universe))

        # 3. Compute fundamental factors for full universe
        fund_rows: list[dict] = []
        for symbol in factor_universe:
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

        # 4. Global snapshot + macro (Provider reads DB cache from T1)
        try:
            global_snapshot = c.market.get_global_snapshot()
        except Exception:  # noqa: BLE001
            logger.warning("[T4] Global snapshot failed", exc_info=True)
            global_snapshot = pd.DataFrame()
        macro_factors = c.macro_engine.compute(global_snapshot)

        # 5. Load telegraph from DB (T2 accumulated 24h news)
        telegraph = c.db.load_recent_news(hours=24)
        logger.info("[T4] Telegraph (from DB): %d items", len(telegraph))

        # 6. Factor ranking across full universe
        prev_pool = c.db.load_factor_pool_previous_day()
        pool_df, _factor_signals = c.factor_strategy.generate_signals(
            tech_df, fund_df, macro_factors, prev_pool=prev_pool,
        )
        logger.info("[T4] Factor pool: %d stocks (universe: %d)",
                    len(pool_df), len(factor_universe))

        if not pool_df.empty:
            c.db.save_factor_pool(today, pool_df)

        # 6b. Signal status tracking
        prev_symbols = set(prev_pool["symbol"]) if not prev_pool.empty else set()
        curr_symbols = set(pool_df["symbol"]) if not pool_df.empty else set()
        status_map: dict[str, tuple[SignalStatus, int, float | None]] = {}
        for sym in curr_symbols:
            prev_row = prev_pool[prev_pool["symbol"] == sym] if not prev_pool.empty else pd.DataFrame()
            prev_score = float(prev_row["score"].iloc[0]) if not prev_row.empty else None
            if sym in prev_symbols:
                status_map[sym] = (SignalStatus.SUSTAINED, 0, prev_score)
            else:
                status_map[sym] = (SignalStatus.NEW_ENTRY, 0, prev_score)
        for sym in prev_symbols - curr_symbols:
            prev_row = prev_pool[prev_pool["symbol"] == sym]
            prev_score = float(prev_row["score"].iloc[0]) if not prev_row.empty else None
            status_map[sym] = (SignalStatus.EXIT, 0, prev_score)

        # 7. Load name + sector map (unified from DB + watchlist YAML)
        name_map, sector_map = c.stock_pool.get_name_sector_maps()

        # 8. Build factor matrix
        factor_matrix = c.factor_strategy._build_factor_matrix(
            factor_universe, tech_df, fund_df, macro_factors,
            {s: sector_map.get(s, "") for s in factor_universe},
        )

        # 9. Build LLM candidates (factor-selected signals + watchlist + EXIT)
        factor_signal_symbols = [s.symbol for s in _factor_signals]
        candidate_symbols = list(dict.fromkeys(
            factor_signal_symbols + watchlist
        ))
        exit_symbols = [s for s in (prev_symbols - curr_symbols) if s not in candidate_symbols]
        candidate_symbols.extend(exit_symbols)
        logger.info("[T4] LLM candidates: %d (factor=%d, watchlist=%d, exit=%d)",
                    len(candidate_symbols), len(factor_signal_symbols),
                    len(watchlist), len(exit_symbols))

        # Build factor signal map for prompt context
        _buy_syms = {s.symbol for s in _factor_signals if s.action == Action.BUY}
        _exit_syms = set(exit_symbols)
        factor_signal_map: dict[str, str] = {}
        for sym in candidate_symbols:
            if sym in _buy_syms:
                factor_signal_map[sym] = "BUY"
            elif sym in _exit_syms:
                factor_signal_map[sym] = "EXIT"
            else:
                factor_signal_map[sym] = "WATCHLIST"

        # Default status for watchlist stocks outside any factor pool
        for sym in candidate_symbols:
            if sym not in status_map:
                status_map[sym] = (SignalStatus.NEW_ENTRY, 0, None)

        # 9b. Pre-fetch news for candidates (I/O in scheduler, pure data to engine)
        stock_news_map: dict[str, tuple[list, list]] = {}
        for sym in candidate_symbols:
            try:
                s_news = c.news.get_stock_news(sym, limit=10)
            except Exception:  # noqa: BLE001
                s_news = []
            try:
                s_anns = c.news.get_announcements(sym, limit=5)
            except Exception:  # noqa: BLE001
                s_anns = []
            stock_news_map[sym] = (s_news, s_anns)

        evidence_pool = c.judge_engine.build_evidence_pool(
            candidate_symbols, pool_df, factor_matrix,
            status_map, tech_df, name_map, stock_news_map,
            factor_signal_map=factor_signal_map,
        )
        logger.info("[T4] Evidence pool: %d stocks", len(evidence_pool))

        # 10. LLM comprehensive judge
        signals: list[TradeSignal] = await c.judge_engine.judge_pool(
            evidence_pool, telegraph=telegraph, global_market=global_snapshot,
            universe_size=len(pool_df),
        )

        # Inject signal status into TradeSignal objects
        for signal in signals:
            st_info = status_map.get(signal.symbol)
            if st_info:
                signal.signal_status = st_info[0]
                signal.days_in_top_n = st_info[1]
                signal.prev_factor_score = st_info[2]

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
    # T5: Sync reference data (calendar + CSI300 constituents)
    # ------------------------------------------------------------------

    async def sync_reference_data(self) -> None:
        """Sync trading calendar and CSI300 index constituents."""
        logger.info("[T5] Syncing reference data...")
        try:
            count = self._c.stock_pool.sync_trading_calendar()
            logger.info("[T5] Calendar: %d new dates synced", count)
        except Exception:  # noqa: BLE001
            logger.warning("[T5] Calendar sync failed", exc_info=True)
        try:
            count = self._c.stock_pool.sync_csi300_constituents()
            logger.info("[T5] CSI300: %d constituents synced", count)
        except Exception:  # noqa: BLE001
            logger.warning("[T5] CSI300 sync failed", exc_info=True)
        logger.info("[T5] Done")

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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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
