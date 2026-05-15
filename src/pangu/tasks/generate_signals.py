"""T4: Full signal pipeline — factors → evidence → LLM judge → push."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from pangu.models import Action, SignalStatus, TradeSignal
from pangu.tz import today_str
from pangu.utils import date_str

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)

_ACTION_EMOJI = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}
_ACTION_ORDER = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
_BARS_LOOKBACK_DAYS = 200
_LLM_FAIL_THRESHOLD = 0.5


async def generate_signals(c: Components) -> None:
    """Full signal pipeline: factors → evidence → LLM judge → push."""
    try:
        await _generate_signals_impl(c)
    except Exception:  # noqa: BLE001
        logger.error("[T4] Signal generation failed", exc_info=True)
        await c.alert("[T4] 信号生成任务异常，请检查日志")


async def _generate_signals_impl(c: Components) -> None:
    """Inner implementation of signal generation pipeline."""
    logger.info("[T4] Generating signals...")
    today = today_str()

    # 0. Data freshness checks
    await _check_data_freshness(c, today)

    # 1. Get stock pools
    watchlist = c.stock_pool.get_watchlist()
    factor_universe = c.stock_pool.get_all_symbols()
    logger.info("[T4] Factor universe: %d stocks, watchlist: %d",
                len(factor_universe), len(watchlist))

    # 2. Load previous pool for status tracking
    prev_pool = c.db.load_factor_pool_previous_day()

    # 3. Factor ranking — ML or classic z-score
    tech_df: dict[str, pd.DataFrame] = {}
    factor_matrix: pd.DataFrame
    if c.ml_strategy is not None:
        pool_df, _factor_signals = await _ml_scoring_path(
            c, today, factor_universe, prev_pool,
        )
        factor_matrix = pd.DataFrame()  # populated later from candidate-only computation
    else:
        pool_df, _factor_signals, factor_matrix, tech_df = _classic_scoring_path(
            c, today, factor_universe, prev_pool,
        )

    logger.info("[T4] Factor pool: %d stocks (universe: %d)",
                len(pool_df), len(factor_universe))

    if not pool_df.empty:
        c.db.save_factor_pool(today, pool_df)

    # 4. Signal status tracking
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

    # 5. Load stock metadata
    stock_meta = c.stock_pool.get_stock_metadata()

    # 6. Build LLM candidates (factor-selected signals + watchlist + EXIT)
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

    # 7. Pre-fetch news for candidates
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

    # 8. ML mode: compute tech/fund/factor_matrix for candidates only
    if c.ml_strategy is not None:
        tech_df, factor_matrix = _build_candidate_factors(c, today, candidate_symbols)

    # 9. Build evidence pool for LLM
    evidence_pool = c.judge_engine.build_evidence_pool(
        candidate_symbols, pool_df, factor_matrix,
        status_map, tech_df, stock_meta, stock_news_map,
        factor_signal_map=factor_signal_map,
    )
    logger.info("[T4] Evidence pool: %d stocks", len(evidence_pool))

    # 10. Load global snapshot + telegraph for LLM
    global_snapshot = c.db.load_latest_global_snapshots()
    telegraph = c.db.load_recent_news(hours=24)
    logger.info("[T4] Telegraph (from DB): %d items", len(telegraph))

    # 11. LLM comprehensive judge
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

    # 12. Save signals + push
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


# ---------------------------------------------------------------------------
# Scoring paths
# ---------------------------------------------------------------------------

async def _ml_scoring_path(
    c: Components,
    today: str,
    factor_universe: list[str],
    prev_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, list[TradeSignal]]:
    """ML scoring: Alpha158 + LightGBM → pool_df + signals.

    Note: factor_matrix is built later (for candidates only) by
    ``_build_candidate_factors``. ML mode does not compute tech indicators
    for the full 800-stock universe.
    """
    pool_df, signals = c.ml_strategy.generate_signals(
        today, list(factor_universe), prev_pool=prev_pool,
    )
    logger.info("[T4/ML] Scored %d stocks, %d signals (window=%d, seeds=%d)",
                len(pool_df), len(signals),
                c.ml_strategy._scorer.window_id,
                c.ml_strategy._scorer.n_models)
    return pool_df, signals


def _build_candidate_factors(
    c: Components,
    today: str,
    candidates: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute tech indicators + fund factors for candidate stocks only.

    Used by ML mode to give LLM the same rich factor context as classic mode,
    but at ~30 stocks instead of 800 — much cheaper.

    Returns
    -------
    (tech_df, factor_matrix) — same shapes as classic mode output.
    """
    if not candidates:
        return {}, pd.DataFrame()

    start = date_str(days_ago=_BARS_LOOKBACK_DAYS)

    # Technical indicators for candidates only
    tech_df: dict[str, pd.DataFrame] = {}
    for symbol in candidates:
        try:
            bars = c.db.load_daily_bars(symbol, start, today)
            if bars is not None and not bars.empty:
                tech_df[symbol] = c.tech_engine.compute(bars)
        except Exception:  # noqa: BLE001
            logger.warning("[T4/ML] %s: tech failed", symbol, exc_info=True)
    logger.info("[T4/ML] Tech factors for candidates: %d/%d",
                len(tech_df), len(candidates))

    # Fundamental factors for candidates
    fund_raw = c.db.load_latest_fundamentals(candidates)
    if not fund_raw.empty and "symbol" in fund_raw.columns:
        fund_raw = fund_raw.set_index("symbol")
    fund_df = c.fund_engine.compute(fund_raw)

    # Macro factors (cheap)
    global_snapshot = c.db.load_latest_global_snapshots()
    macro_factors = c.macro_engine.compute(global_snapshot)

    # Build factor matrix (LLM context)
    stock_meta = c.stock_pool.get_stock_metadata()
    sector_map = {s: stock_meta[s].sector if s in stock_meta else "" for s in candidates}
    factor_matrix = c.factor_strategy._build_factor_matrix(
        candidates, tech_df, fund_df, macro_factors, sector_map,
    )

    return tech_df, factor_matrix


def _classic_scoring_path(
    c: Components,
    today: str,
    factor_universe: list[str],
    prev_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, list[TradeSignal], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Classic z-score path: tech/fund/macro → MultiFactorStrategy."""
    start = date_str(days_ago=_BARS_LOOKBACK_DAYS)

    # Compute technical factors for full universe
    tech_df: dict[str, pd.DataFrame] = {}
    for symbol in factor_universe:
        try:
            bars = c.db.load_daily_bars(symbol, start, today)
            if bars is not None and not bars.empty:
                tech_df[symbol] = c.tech_engine.compute(bars)
        except Exception:  # noqa: BLE001
            logger.warning("[T4] %s: tech failed", symbol, exc_info=True)
    logger.info("[T4] Tech factors: %d/%d", len(tech_df), len(factor_universe))

    # Compute fundamental factors
    fund_raw = c.db.load_latest_fundamentals(list(factor_universe))
    if not fund_raw.empty and "symbol" in fund_raw.columns:
        fund_raw = fund_raw.set_index("symbol")
    fund_df = c.fund_engine.compute(fund_raw)

    # Global snapshot + macro
    global_snapshot = c.db.load_latest_global_snapshots()
    macro_factors = c.macro_engine.compute(global_snapshot)

    # Factor ranking
    pool_df, signals = c.factor_strategy.generate_signals(
        tech_df, fund_df, macro_factors, prev_pool=prev_pool,
    )

    # Build factor matrix for LLM
    stock_meta = c.stock_pool.get_stock_metadata()
    sector_map = {s: stock_meta[s].sector if s in stock_meta else "" for s in factor_universe}
    factor_matrix = c.factor_strategy._build_factor_matrix(
        list(factor_universe), tech_df, fund_df, macro_factors, sector_map,
    )

    return pool_df, signals, factor_matrix, tech_df


async def _check_data_freshness(c: Components, today: str) -> None:
    """Alert if key data sources are stale."""
    from datetime import timedelta

    from pangu.tz import now as tz_now

    # K-line freshness: warn if latest bar is >1 trading day behind
    latest_bar = c.db.get_latest_bar_date()
    if latest_bar is not None:
        missed = c.db.has_trading_day_between(latest_bar, today)
        if missed is True:
            await c.alert(f"[T4] K线数据可能过期: 最新日期 {latest_bar}，请检查 T3 是否正常")

    # News freshness: warn if no news in 24 hours
    latest_news = c.db.get_latest_news_timestamp()
    if latest_news is None:
        await c.alert("[T4] 数据库中无新闻数据，请检查 T2 是否正常")
    else:
        try:
            from datetime import datetime

            news_dt = datetime.fromisoformat(latest_news)
            if tz_now() - news_dt > timedelta(hours=24):
                await c.alert("[T4] 24小时内无新闻更新，请检查 T2 是否正常")
        except (ValueError, TypeError):
            pass


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
