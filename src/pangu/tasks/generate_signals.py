"""T4: Weekly LLM-TopkDropout rebalance pipeline.

Daily run:
  * Data freshness checks (always).
  * If not the first trading day of an ISO week → return (no trade signals).
  * Otherwise: ML scoring → SELL pool (held bottom) + BUY pool (non-held top)
    → LLM pool-level Bull/Bear/Judge → fallback to ML rank for under-fill
    → write target_portfolio.json → push BUY/SELL signals.

Classic z-score path (when ml.enabled is False) is kept for backward compat;
it does not implement weekly gating and runs the legacy daily flow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from pangu.models import Action, SignalStatus, TradeSignal
from pangu.portfolio import Portfolio
from pangu.tz import now as _tz_now
from pangu.tz import today_str
from pangu.utils import date_str, is_rebalance_day

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)

_ACTION_EMOJI = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}
_ACTION_ORDER = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
_BARS_LOOKBACK_DAYS = 200
_LLM_FAIL_THRESHOLD = 0.5

# Factor keys forwarded to LLM rebalance evidence packs (must intersect
# pangu.strategy.llm.judge._KNOWN_FACTORS).
_REBAL_FACTOR_KEYS = (
    "rsi_14", "macd_hist", "bias_20", "obv", "atr_14",
    "volume_ratio", "pe_ttm", "pb", "roe_ttm",
)


async def generate_signals(c: Components) -> None:
    """Full signal pipeline: data checks → (weekly) ML+LLM rebalance → push."""
    try:
        await _generate_signals_impl(c)
    except Exception:  # noqa: BLE001
        logger.error("[T4] Signal generation failed", exc_info=True)
        await c.alert("[T4] 信号生成任务异常，请检查日志")


async def _generate_signals_impl(c: Components) -> None:
    today = today_str()
    logger.info("[T4] Generating signals for %s ...", today)

    await _check_data_freshness(c, today)

    if c.ml_strategy is not None:
        if not is_rebalance_day(today, c.db):
            logger.info("[T4/ML] %s is not a rebalance day (not ISO week start); "
                        "skipping rebalance, freshness checks only", today)
            return
        await _ml_rebalance_path(c, today)
    else:
        # Classic z-score path retains the legacy daily flow.
        await _classic_path(c, today)


# ---------------------------------------------------------------------------
# ML-TopkDropout rebalance path (weekly)
# ---------------------------------------------------------------------------

async def _ml_rebalance_path(c: Components, today: str) -> None:
    ml = c.ml_strategy
    ps = c.portfolio_state
    if ps is None:
        raise RuntimeError("ML mode requires PortfolioState; check Components wiring.")

    # 1. Load previous virtual portfolio
    prev_portfolio = ps.load()
    holdings = list(prev_portfolio.symbols) if prev_portfolio else []
    prev_ranks = dict(prev_portfolio.ranks) if prev_portfolio else {}
    logger.info("[T4/ML] Prev portfolio: %d holdings (%s)",
                len(holdings), "cold start" if not holdings else f"date={prev_portfolio.date}")

    # 2. Score ML pool
    factor_universe = c.stock_pool.get_all_symbols()
    pool_df = ml.score_pool(today, list(factor_universe))
    if pool_df.empty:
        await c.alert(f"[T4/ML] ML score_pool returned empty for {today}; aborting rebalance.")
        return
    logger.info("[T4/ML] Scored %d stocks (window=%d, seeds=%d)",
                len(pool_df), ml._scorer.window_id, ml._scorer.n_models)

    # Persist pool snapshot to DB (existing behaviour for downstream tools)
    c.db.save_factor_pool(today, pool_df)

    universe_size = len(pool_df)

    # 3. Cold start: choose top-N as initial portfolio, no LLM call needed
    if not holdings:
        new_symbols = ml.cold_start_portfolio(pool_df)
        logger.info("[T4/ML] Cold start: initializing portfolio with %d symbols",
                    len(new_symbols))
        ps.save(_make_portfolio(today, new_symbols, pool_df))
        signals = _build_cold_start_signals(new_symbols, pool_df)
        await _persist_and_push(c, signals)
        return

    # 4. Build candidate pools
    sell_pool = ml.get_sell_candidate_pool(pool_df, holdings)
    buy_pool = ml.get_buy_candidate_pool(pool_df, holdings)
    logger.info("[T4/ML] Candidate pools: SELL=%d, BUY=%d", len(sell_pool), len(buy_pool))

    if not sell_pool and not buy_pool:
        logger.info("[T4/ML] No candidates on either side; portfolio unchanged.")
        ps.save(_make_portfolio(today, holdings, pool_df))
        return

    # 5. Build evidence packs for LLM
    all_syms = list(dict.fromkeys(sell_pool + buy_pool))
    tech_df, factor_matrix = _build_candidate_factors(c, today, all_syms)
    stock_meta = c.stock_pool.get_stock_metadata()
    stock_news_map = _prefetch_news(c, all_syms)

    sell_info = _build_pool_info(
        sell_pool, pool_df, factor_matrix,
        stock_meta, stock_news_map,
        prev_ranks=prev_ranks,
    )
    buy_info = _build_pool_info(
        buy_pool, pool_df, factor_matrix,
        stock_meta, stock_news_map,
        prev_ranks=None,
    )

    # 6. Global snapshot + telegraph
    global_snapshot = c.db.load_latest_global_snapshots()
    telegraph = c.db.load_recent_news(hours=24)

    # 7. LLM rebalance decision
    decision = await c.judge_engine.judge_rebalance(
        today=today,
        sell_candidates=sell_info,
        buy_candidates=buy_info,
        telegraph=telegraph,
        global_market=global_snapshot,
        top_n=ml.top_n,
        n_drop=ml.n_drop,
        universe_size=universe_size,
    )
    logger.info("[T4/ML] LLM decision (source=%s): sells=%d, buys=%d",
                decision.source, len(decision.sells), len(decision.buys))

    if decision.source == "llm_failed":
        await c.alert("[T4/ML] LLM 调仓决策失败，使用 ML 排名兜底 (经典 TopkDropout 等价)")

    # 8. Apply LLM choices + ML fallback to reach n_drop
    n_drop = ml.n_drop
    llm_sell_picks = {p.symbol: p for p in decision.sells}
    llm_buy_picks = {p.symbol: p for p in decision.buys}

    llm_sell_syms = list(llm_sell_picks.keys())[:n_drop]
    llm_buy_syms = list(llm_buy_picks.keys())[:n_drop]

    fb_sells = ml.fallback_sells(sell_pool, set(llm_sell_syms), pool_df,
                                 n_drop - len(llm_sell_syms))
    fb_buys = ml.fallback_buys(buy_pool, set(llm_buy_syms), pool_df,
                               n_drop - len(llm_buy_syms))

    final_sells = llm_sell_syms + fb_sells
    final_buys = llm_buy_syms + fb_buys

    # Balance turnover: SELL and BUY counts must match
    n_turn = min(len(final_sells), len(final_buys))
    final_sells = final_sells[:n_turn]
    final_buys = final_buys[:n_turn]

    logger.info("[T4/ML] Final rebalance: SELL %d (llm=%d, fallback=%d), "
                "BUY %d (llm=%d, fallback=%d)",
                len(final_sells),
                sum(1 for s in final_sells if s in llm_sell_picks),
                sum(1 for s in final_sells if s not in llm_sell_picks),
                len(final_buys),
                sum(1 for s in final_buys if s in llm_buy_picks),
                sum(1 for s in final_buys if s not in llm_buy_picks))

    # 9. Update portfolio
    held_set = set(holdings)
    held_set.difference_update(final_sells)
    held_set.update(final_buys)
    new_symbols = sorted(held_set)
    ps.save(_make_portfolio(today, new_symbols, pool_df))

    # 10. Build trade signals (BUY/SELL only) and push
    signals = _build_rebalance_signals(
        final_sells=final_sells,
        final_buys=final_buys,
        llm_sell_picks=llm_sell_picks,
        llm_buy_picks=llm_buy_picks,
        decision=decision,
        pool_df=pool_df,
        stock_meta=stock_meta,
    )
    await _persist_and_push(c, signals)


# ---------------------------------------------------------------------------
# Pool info builder for LLM
# ---------------------------------------------------------------------------

def _build_pool_info(
    symbols: list[str],
    pool_df: pd.DataFrame,
    factor_matrix: pd.DataFrame,
    stock_meta: dict,
    stock_news_map: dict[str, tuple[list, list]],
    *,
    prev_ranks: dict[str, int] | None,
) -> list[dict[str, Any]]:
    """Build per-symbol evidence dicts consumed by judge_rebalance."""
    if not symbols:
        return []
    score_map = (
        dict(zip(pool_df["symbol"], pool_df["score"], strict=False))
        if not pool_df.empty else {}
    )
    rank_map = (
        dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
        if not pool_df.empty else {}
    )

    out: list[dict[str, Any]] = []
    for sym in symbols:
        ml_score = float(score_map.get(sym, 0.0))
        ml_rank = int(rank_map.get(sym, 0))
        prev_rank = None
        rank_delta = None
        if prev_ranks is not None:
            prev_rank = prev_ranks.get(sym)
            if prev_rank is not None and ml_rank > 0:
                rank_delta = int(ml_rank - prev_rank)

        factor_details: dict[str, float] = {}
        if not factor_matrix.empty and sym in factor_matrix.index:
            row = factor_matrix.loc[sym]
            for key in _REBAL_FACTOR_KEYS:
                if key in row.index:
                    val = row[key]
                    if pd.notna(val):
                        factor_details[key] = float(val)

        news, anns = stock_news_map.get(sym, ([], []))
        out.append({
            "symbol": sym,
            "name": stock_meta[sym].name if sym in stock_meta else sym,
            "ml_score": ml_score,
            "ml_rank": ml_rank,
            "prev_ml_rank": prev_rank,
            "rank_delta": rank_delta,
            "factor_details": factor_details,
            "stock_news": news,
            "announcements": anns,
        })
    return out


# ---------------------------------------------------------------------------
# Candidate-only factor computation (reuse from previous T4)
# ---------------------------------------------------------------------------

def _build_candidate_factors(
    c: Components,
    today: str,
    candidates: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute tech + fund + macro factors for candidate stocks only."""
    if not candidates:
        return {}, pd.DataFrame()

    start = date_str(days_ago=_BARS_LOOKBACK_DAYS)
    tech_df: dict[str, pd.DataFrame] = {}
    for symbol in candidates:
        try:
            bars = c.db.load_daily_bars(symbol, start, today)
            if bars is not None and not bars.empty:
                tech_df[symbol] = c.tech_engine.compute(bars)
        except Exception:  # noqa: BLE001
            logger.warning("[T4/ML] %s: tech failed", symbol, exc_info=True)
    logger.info("[T4/ML] Tech factors for candidates: %d/%d", len(tech_df), len(candidates))

    fund_raw = c.db.load_latest_fundamentals(candidates)
    if not fund_raw.empty and "symbol" in fund_raw.columns:
        fund_raw = fund_raw.set_index("symbol")
    fund_df = c.fund_engine.compute(fund_raw)

    global_snapshot = c.db.load_latest_global_snapshots()
    macro_factors = c.macro_engine.compute(global_snapshot)

    stock_meta = c.stock_pool.get_stock_metadata()
    sector_map = {s: stock_meta[s].sector if s in stock_meta else "" for s in candidates}
    factor_matrix = c.factor_strategy._build_factor_matrix(
        candidates, tech_df, fund_df, macro_factors, sector_map,
    )
    return tech_df, factor_matrix


def _prefetch_news(c: Components, symbols: list[str]) -> dict[str, tuple[list, list]]:
    out: dict[str, tuple[list, list]] = {}
    for sym in symbols:
        try:
            s_news = c.news.get_stock_news(sym, limit=10)
        except Exception:  # noqa: BLE001
            s_news = []
        try:
            s_anns = c.news.get_announcements(sym, limit=5)
        except Exception:  # noqa: BLE001
            s_anns = []
        out[sym] = (s_news, s_anns)
    return out


# ---------------------------------------------------------------------------
# Portfolio + signal builders
# ---------------------------------------------------------------------------

def _make_portfolio(date: str, symbols: list[str], pool_df: pd.DataFrame) -> Portfolio:
    score_map = (
        dict(zip(pool_df["symbol"], pool_df["score"], strict=False))
        if not pool_df.empty else {}
    )
    rank_map = (
        dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
        if not pool_df.empty else {}
    )
    return Portfolio(
        date=date,
        symbols=sorted(symbols),
        scores={s: float(score_map.get(s, 0.0)) for s in symbols},
        ranks={s: int(rank_map.get(s, 0)) for s in symbols},
    )


def _build_cold_start_signals(
    symbols: list[str], pool_df: pd.DataFrame,
) -> list[TradeSignal]:
    now = _tz_now()
    score_map = dict(zip(pool_df["symbol"], pool_df["score"], strict=False))
    rank_map = dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
    signals: list[TradeSignal] = []
    for sym in sorted(symbols):
        score = float(score_map.get(sym, 0.0))
        rank = int(rank_map.get(sym, 0))
        signals.append(TradeSignal(
            timestamp=now,
            symbol=sym, name=sym,
            action=Action.BUY,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=0.0,
            confidence=score,
            source="ml_cold_start",
            reason=f"冷启动 - ML rank {rank}",
            factor_score=score,
            metadata={"cold_start": True, "ml_rank": rank, "ml_score": score},
        ))
    return signals


def _build_rebalance_signals(
    *,
    final_sells: list[str],
    final_buys: list[str],
    llm_sell_picks: dict,
    llm_buy_picks: dict,
    decision,
    pool_df: pd.DataFrame,
    stock_meta: dict,
) -> list[TradeSignal]:
    now = _tz_now()
    score_map = dict(zip(pool_df["symbol"], pool_df["score"], strict=False))
    rank_map = dict(zip(pool_df["symbol"], pool_df["rank"], strict=False))
    signals: list[TradeSignal] = []

    for sym in final_sells:
        pick = llm_sell_picks.get(sym)
        score = float(score_map.get(sym, 0.0))
        rank = int(rank_map.get(sym, 0))
        if pick is not None:
            reason = pick.reason
            origin = "llm"
            evidence = pick.evidence
        else:
            reason = f"ML 排名兜底卖出 (rank={rank})"
            origin = "fallback"
            evidence = ""
        signals.append(TradeSignal(
            timestamp=now,
            symbol=sym,
            name=stock_meta[sym].name if sym in stock_meta else sym,
            action=Action.SELL,
            signal_status=SignalStatus.EXIT,
            days_in_top_n=0,
            price=0.0,
            confidence=1.0 - score,
            source=f"llm_topkdrop:{origin}",
            reason=reason,
            factor_score=score,
            metadata={
                "origin": origin,
                "ml_rank": rank,
                "ml_score": score,
                "evidence": evidence,
                "sell_debate_bull": decision.sell_debate.bull,
                "sell_debate_bear": decision.sell_debate.bear,
                "n_drop": len(final_sells),
            },
        ))

    for sym in final_buys:
        pick = llm_buy_picks.get(sym)
        score = float(score_map.get(sym, 0.0))
        rank = int(rank_map.get(sym, 0))
        if pick is not None:
            reason = pick.reason
            origin = "llm"
            evidence = pick.evidence
        else:
            reason = f"ML 排名兜底买入 (rank={rank})"
            origin = "fallback"
            evidence = ""
        signals.append(TradeSignal(
            timestamp=now,
            symbol=sym,
            name=stock_meta[sym].name if sym in stock_meta else sym,
            action=Action.BUY,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=0.0,
            confidence=score,
            source=f"llm_topkdrop:{origin}",
            reason=reason,
            factor_score=score,
            metadata={
                "origin": origin,
                "ml_rank": rank,
                "ml_score": score,
                "evidence": evidence,
                "buy_debate_bull": decision.buy_debate.bull,
                "buy_debate_bear": decision.buy_debate.bear,
                "n_drop": len(final_buys),
            },
        ))
    return signals


# ---------------------------------------------------------------------------
# Persist + push
# ---------------------------------------------------------------------------

async def _persist_and_push(c: Components, signals: list[TradeSignal]) -> None:
    if not signals:
        logger.info("[T4] No signals to push.")
        return
    for sig in signals:
        c.db.save_trade_signal(sig)
    _print_signal_summary(signals)

    signals.sort(key=lambda s: (_ACTION_ORDER.get(s.action, 9), -(s.factor_score or 0)))
    pushed = 0
    for sig in signals:
        try:
            result = await c.notif_manager.notify_signal(sig)
            if result:
                pushed += 1
                logger.info("[T4] Pushed %s: %s", sig.symbol, result)
        except Exception:  # noqa: BLE001
            logger.warning("[T4] Push failed for %s", sig.symbol, exc_info=True)
    logger.info("[T4] Done — %d signals saved, %d pushed", len(signals), pushed)


# ---------------------------------------------------------------------------
# Classic z-score path (legacy, kept for ml.enabled = false)
# ---------------------------------------------------------------------------

async def _classic_path(c: Components, today: str) -> None:
    """Legacy daily z-score + per-stock LLM judge pipeline.

    Kept verbatim from the pre-LLM-TopkDropout implementation for users who
    explicitly opt out of ML mode by setting ``ml.enabled = false``.
    """
    factor_universe = c.stock_pool.get_all_symbols()
    watchlist = c.stock_pool.get_watchlist()
    logger.info("[T4/classic] Factor universe: %d stocks, watchlist: %d",
                len(factor_universe), len(watchlist))

    prev_pool = c.db.load_factor_pool_previous_day()
    pool_df, factor_signals, factor_matrix, tech_df = _classic_scoring_path(
        c, today, list(factor_universe), prev_pool,
    )

    if not pool_df.empty:
        c.db.save_factor_pool(today, pool_df)

    # Signal status tracking
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

    stock_meta = c.stock_pool.get_stock_metadata()
    factor_signal_symbols = [s.symbol for s in factor_signals]
    candidate_symbols = list(dict.fromkeys(factor_signal_symbols + watchlist))
    exit_symbols = [s for s in (prev_symbols - curr_symbols) if s not in candidate_symbols]
    candidate_symbols.extend(exit_symbols)

    buy_syms = {s.symbol for s in factor_signals if s.action == Action.BUY}
    exit_set = set(exit_symbols)
    factor_signal_map: dict[str, str] = {}
    for sym in candidate_symbols:
        if sym in buy_syms:
            factor_signal_map[sym] = "BUY"
        elif sym in exit_set:
            factor_signal_map[sym] = "EXIT"
        else:
            factor_signal_map[sym] = "WATCHLIST"

    for sym in candidate_symbols:
        if sym not in status_map:
            status_map[sym] = (SignalStatus.NEW_ENTRY, 0, None)

    stock_news_map = _prefetch_news(c, candidate_symbols)

    evidence_pool = c.judge_engine.build_evidence_pool(
        candidate_symbols, pool_df, factor_matrix,
        status_map, tech_df, stock_meta, stock_news_map,
        factor_signal_map=factor_signal_map,
    )
    global_snapshot = c.db.load_latest_global_snapshots()
    telegraph = c.db.load_recent_news(hours=24)

    signals = await c.judge_engine.judge_pool(
        evidence_pool, telegraph=telegraph, global_market=global_snapshot,
        universe_size=len(pool_df),
    )

    if not signals and evidence_pool:
        await c.alert(f"[T4] LLM 判断全部失败，{len(evidence_pool)} 只候选股无信号生成")
    elif evidence_pool and len(signals) < len(evidence_pool) * (1 - _LLM_FAIL_THRESHOLD):
        failed = len(evidence_pool) - len(signals)
        await c.alert(f"[T4] LLM 判断部分失败: {failed}/{len(evidence_pool)} 只股票失败")

    for sig in signals:
        st_info = status_map.get(sig.symbol)
        if st_info:
            sig.signal_status = st_info[0]
            sig.days_in_top_n = st_info[1]
            sig.prev_factor_score = st_info[2]

    logger.info("[T4/classic] Signals: %d", len(signals))
    _print_signal_summary(signals)

    for sig in signals:
        c.db.save_trade_signal(sig)

    actionable = [s for s in signals if s.action != Action.HOLD]
    if not actionable and signals:
        await c.alert("[T4] 今日无买卖信号，所有股票建议持有观望")

    watchlist_set = set(watchlist)
    to_push = [s for s in signals if s.action != Action.HOLD or s.symbol in watchlist_set]
    to_push.sort(key=lambda s: (_ACTION_ORDER.get(s.action, 9), -(s.factor_score or 0)))
    for sig in to_push:
        try:
            result = await c.notif_manager.notify_signal(sig)
            if result:
                logger.info("[T4/classic] Pushed %s: %s", sig.symbol, result)
        except Exception:  # noqa: BLE001
            logger.warning("[T4/classic] Push failed for %s", sig.symbol, exc_info=True)


def _classic_scoring_path(
    c: Components,
    today: str,
    factor_universe: list[str],
    prev_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, list[TradeSignal], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Classic z-score path: tech/fund/macro → MultiFactorStrategy."""
    start = date_str(days_ago=_BARS_LOOKBACK_DAYS)
    tech_df: dict[str, pd.DataFrame] = {}
    for symbol in factor_universe:
        try:
            bars = c.db.load_daily_bars(symbol, start, today)
            if bars is not None and not bars.empty:
                tech_df[symbol] = c.tech_engine.compute(bars)
        except Exception:  # noqa: BLE001
            logger.warning("[T4] %s: tech failed", symbol, exc_info=True)
    logger.info("[T4] Tech factors: %d/%d", len(tech_df), len(factor_universe))

    fund_raw = c.db.load_latest_fundamentals(list(factor_universe))
    if not fund_raw.empty and "symbol" in fund_raw.columns:
        fund_raw = fund_raw.set_index("symbol")
    fund_df = c.fund_engine.compute(fund_raw)

    global_snapshot = c.db.load_latest_global_snapshots()
    macro_factors = c.macro_engine.compute(global_snapshot)

    pool_df, signals = c.factor_strategy.generate_signals(
        tech_df, fund_df, macro_factors, prev_pool=prev_pool,
    )

    stock_meta = c.stock_pool.get_stock_metadata()
    sector_map = {s: stock_meta[s].sector if s in stock_meta else "" for s in factor_universe}
    factor_matrix = c.factor_strategy._build_factor_matrix(
        list(factor_universe), tech_df, fund_df, macro_factors, sector_map,
    )
    return pool_df, signals, factor_matrix, tech_df


# ---------------------------------------------------------------------------
# Data freshness + display helpers
# ---------------------------------------------------------------------------

async def _check_data_freshness(c: Components, today: str) -> None:
    """Alert if key data sources are stale."""
    from datetime import datetime, timedelta

    latest_bar = c.db.get_latest_bar_date()
    if latest_bar is not None:
        missed = c.db.has_trading_day_between(latest_bar, today)
        if missed is True:
            await c.alert(f"[T4] K线数据可能过期: 最新日期 {latest_bar}，请检查 T3 是否正常")

    latest_news = c.db.get_latest_news_timestamp()
    if latest_news is None:
        await c.alert("[T4] 数据库中无新闻数据，请检查 T2 是否正常")
    else:
        try:
            news_dt = datetime.fromisoformat(latest_news)
            if _tz_now() - news_dt > timedelta(hours=24):
                await c.alert("[T4] 24小时内无新闻更新，请检查 T2 是否正常")
        except (ValueError, TypeError):
            pass


def _print_signal_summary(signals: list[TradeSignal]) -> None:
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
