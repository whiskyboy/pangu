"""T4: Full signal pipeline — factors → evidence → LLM judge → push."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from trading_agent.models import Action, SignalStatus, TradeSignal
from trading_agent.tz import today_str
from trading_agent.utils import date_str

if TYPE_CHECKING:
    from trading_agent.scheduler import Components

logger = logging.getLogger(__name__)

_ACTION_EMOJI = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}
_ACTION_ORDER = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
_BARS_LOOKBACK_DAYS = 200
_LLM_FAIL_THRESHOLD = 0.5


async def generate_signals(c: Components) -> None:
    """Full signal pipeline: factors → evidence → LLM judge → push."""
    logger.info("[T4] Generating signals...")
    today = today_str()
    start = date_str(days_ago=_BARS_LOOKBACK_DAYS)

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

    if not signals and evidence_pool:
        await c.alert(f"[T4] LLM 判断全部失败，{len(evidence_pool)} 只候选股无信号生成")
    elif evidence_pool and len(signals) < len(evidence_pool) * (1 - _LLM_FAIL_THRESHOLD):
        failed = len(evidence_pool) - len(signals)
        await c.alert(
            f"[T4] LLM 判断部分失败: {failed}/{len(evidence_pool)} 只股票失败"
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

    actionable = [s for s in signals if s.action != Action.HOLD]
    if not actionable and signals:
        await c.alert("[T4] 今日无买卖信号，所有股票建议持有观望")

    watchlist_set = set(watchlist)
    to_push = [s for s in signals if s.action != Action.HOLD or s.symbol in watchlist_set]
    to_push.sort(key=lambda s: (_ACTION_ORDER.get(s.action, 9), -(s.factor_score or 0)))

    for signal in to_push:
        try:
            result = await c.notif_manager.notify_signal(signal)
            if result:
                logger.info("[T4] Push %s: %s", signal.symbol, result)
        except Exception:  # noqa: BLE001
            logger.warning("[T4] Push failed for %s", signal.symbol, exc_info=True)

    logger.info("[T4] Done — %d signals, %d pushed", len(signals), len(to_push))


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
