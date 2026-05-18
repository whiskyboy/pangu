"""T6: Weekly LLM-TopkDropout rebalance pipeline.

Daily run:
  * Data freshness checks (always).
  * If not the first trading day of an ISO week → return (no trade signals).
  * Otherwise: ML scoring → SELL pool (held bottom) + BUY pool (non-held top)
    → LLM pool-level Bull/Bear/Judge → fallback to ML rank for under-fill
    → write target_portfolio.json → push BUY/SELL signals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from pangu.factor.matrix import build_factor_matrix
from pangu.models import Action, SignalStatus, TradeSignal
from pangu.portfolio import Portfolio
from pangu.strategy.ml.ml_strategy import pool_score_rank_maps
from pangu.tasks._base import scheduled_task
from pangu.tz import now as _tz_now
from pangu.tz import today_str
from pangu.utils import date_str, is_rebalance_day

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)

_ACTION_EMOJI = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}
_ACTION_ORDER = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
_BARS_LOOKBACK_DAYS = 200

# Factor keys forwarded to LLM rebalance evidence packs (must intersect
# pangu.strategy.llm.judge._KNOWN_FACTORS).
_REBAL_FACTOR_KEYS = (
    "rsi_14",
    "macd_hist",
    "bias_20",
    "obv",
    "atr_14",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "roe_ttm",
)


@scheduled_task("T6", "信号生成与调仓")
async def generate_signals(c: Components) -> None:
    """Full signal pipeline: data checks → (weekly) ML+LLM rebalance → push."""
    await _generate_signals_impl(c)


async def _generate_signals_impl(c: Components) -> None:
    today = today_str()
    logger.info("[T6] Generating signals for %s ...", today)

    await _check_data_freshness(c, today)

    if c.ml_strategy is None:
        await c.alert("[T6] ml_strategy 未装配，无法生成信号；请检查 [ml].enabled 配置")
        return
    if not is_rebalance_day(today, c.db):
        logger.info(
            "[T6/ML] %s is not a rebalance day (not ISO week start); skipping rebalance, freshness checks only", today
        )
        return
    await _ml_rebalance_path(c, today)


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
    logger.info(
        "[T6/ML] Prev portfolio: %d holdings (%s)",
        len(holdings),
        "cold start" if not holdings else f"date={prev_portfolio.date}",
    )

    # 2. Score ML pool
    #
    # T6 runs at 08:15 — today's K-line bars don't exist yet (T3 ingests at
    # 18:00). Use the most recent available bar date (usually T-1) as the
    # factor reference date, while keeping ``today`` as the rebalance /
    # snapshot date.
    factor_date = c.db.get_latest_bar_date()
    if factor_date is None:
        await c.alert("[T6/ML] 无 K 线数据，无法生成信号；请检查 T3 是否正常")
        return
    factor_universe = c.stock_pool.get_all_symbols()
    pool_df = ml.score_pool(factor_date, list(factor_universe))
    if pool_df.empty:
        await c.alert(f"[T6/ML] ML score_pool returned empty for factor_date={factor_date}; aborting rebalance.")
        return
    logger.info(
        "[T6/ML] Scored %d stocks (factor_date=%s, window=%d, seeds=%d)",
        len(pool_df),
        factor_date,
        ml._scorer.window_id,
        ml._scorer.n_models,
    )

    # Persist pool snapshot to DB (existing behaviour for downstream tools)
    c.db.save_factor_pool(today, pool_df)

    universe_size = len(pool_df)

    # 3. Cold start: choose top-N as initial portfolio, no LLM call needed
    if not holdings:
        new_symbols = ml.cold_start_portfolio(pool_df)
        logger.info("[T6/ML] Cold start: initializing portfolio with %d symbols", len(new_symbols))
        ps.save(_make_portfolio(today, new_symbols, pool_df))
        c.db.save_portfolio_snapshot(today, sorted(new_symbols), is_rebalance=True)
        per_stock = c.initial_capital / max(1, ml.top_n)
        signals = _build_cold_start_signals(new_symbols, pool_df, c=c, per_stock_capital=per_stock)
        await _persist_and_push(c, signals)
        return

    # 4. Build candidate pools
    sell_pool = ml.get_sell_candidate_pool(pool_df, holdings)
    buy_pool = ml.get_buy_candidate_pool(pool_df, holdings)
    logger.info("[T6/ML] Candidate pools: SELL=%d, BUY=%d", len(sell_pool), len(buy_pool))

    if not sell_pool and not buy_pool:
        logger.info("[T6/ML] No candidates on either side; portfolio unchanged.")
        ps.save(_make_portfolio(today, holdings, pool_df))
        c.db.save_portfolio_snapshot(today, sorted(holdings), is_rebalance=True)
        return

    # 5. Build evidence packs for LLM
    all_syms = list(dict.fromkeys(sell_pool + buy_pool))
    tech_df, factor_matrix = _build_candidate_factors(c, today, all_syms)
    stock_meta = c.stock_pool.get_stock_metadata()
    stock_news_map = _prefetch_news(c, all_syms)

    sell_info = _build_pool_info(
        sell_pool,
        pool_df,
        factor_matrix,
        stock_meta,
        stock_news_map,
        prev_ranks=prev_ranks,
    )
    buy_info = _build_pool_info(
        buy_pool,
        pool_df,
        factor_matrix,
        stock_meta,
        stock_news_map,
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
        stock_meta=stock_meta,
    )
    logger.info(
        "[T6/ML] LLM decision (source=%s): sells=%d, buys=%d", decision.source, len(decision.sells), len(decision.buys)
    )

    if decision.source == "llm_failed":
        await c.alert("[T6/ML] LLM 调仓决策失败，使用 ML 排名兜底 (经典 TopkDropout 等价)")

    # 8. Apply LLM choices + ML fallback to reach n_drop
    n_drop = ml.n_drop
    llm_sell_picks = {p.symbol: p for p in decision.sells}
    llm_buy_picks = {p.symbol: p for p in decision.buys}

    llm_sell_syms = list(llm_sell_picks.keys())[:n_drop]
    llm_buy_syms = list(llm_buy_picks.keys())[:n_drop]

    fb_sells = ml.fallback_sells(sell_pool, set(llm_sell_syms), pool_df, n_drop - len(llm_sell_syms))
    fb_buys = ml.fallback_buys(buy_pool, set(llm_buy_syms), pool_df, n_drop - len(llm_buy_syms))

    final_sells = llm_sell_syms + fb_sells
    final_buys = llm_buy_syms + fb_buys

    # Balance turnover: SELL and BUY counts must match
    n_turn = min(len(final_sells), len(final_buys))
    final_sells = final_sells[:n_turn]
    final_buys = final_buys[:n_turn]

    logger.info(
        "[T6/ML] Final rebalance: SELL %d (llm=%d, fallback=%d), BUY %d (llm=%d, fallback=%d)",
        len(final_sells),
        sum(1 for s in final_sells if s in llm_sell_picks),
        sum(1 for s in final_sells if s not in llm_sell_picks),
        len(final_buys),
        sum(1 for s in final_buys if s in llm_buy_picks),
        sum(1 for s in final_buys if s not in llm_buy_picks),
    )

    # 9. Update portfolio
    held_set = set(holdings)
    held_set.difference_update(final_sells)
    held_set.update(final_buys)
    new_symbols = sorted(held_set)
    ps.save(_make_portfolio(today, new_symbols, pool_df))
    c.db.save_portfolio_snapshot(today, new_symbols, is_rebalance=True)

    # 10. Build trade signals (BUY/SELL only) and push
    per_stock = c.initial_capital / max(1, ml.top_n)
    signals = _build_rebalance_signals(
        final_sells=final_sells,
        final_buys=final_buys,
        llm_sell_picks=llm_sell_picks,
        llm_buy_picks=llm_buy_picks,
        decision=decision,
        pool_df=pool_df,
        stock_meta=stock_meta,
        c=c,
        per_stock_capital=per_stock,
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
    score_map, rank_map = pool_score_rank_maps(pool_df)

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
        out.append(
            {
                "symbol": sym,
                "name": stock_meta[sym].name if sym in stock_meta else sym,
                "ml_score": ml_score,
                "ml_rank": ml_rank,
                "prev_ml_rank": prev_rank,
                "rank_delta": rank_delta,
                "factor_details": factor_details,
                "stock_news": news,
                "announcements": anns,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Candidate-only factor computation (reuse from previous version of this task)
# ---------------------------------------------------------------------------


def _build_candidate_factors(
    c: Components,
    today: str,
    candidates: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute tech + fund factors for candidate stocks only.

    Returns ``(tech_df_by_symbol, factor_matrix)`` where ``factor_matrix`` is a
    cross-sectional table indexed by symbol — fed to ``_build_pool_info`` to
    extract ``_REBAL_FACTOR_KEYS`` for the LLM evidence packs.
    """
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
            logger.warning("[T6/ML] %s: tech failed", symbol, exc_info=True)
    logger.info("[T6/ML] Tech factors for candidates: %d/%d", len(tech_df), len(candidates))

    fund_raw = c.db.load_latest_fundamentals(candidates)
    if not fund_raw.empty and "symbol" in fund_raw.columns:
        fund_raw = fund_raw.set_index("symbol")
    fund_df = c.fund_engine.compute(fund_raw)

    factor_matrix = build_factor_matrix(candidates, tech_df, fund_df)
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
    score_map, rank_map = pool_score_rank_maps(pool_df)
    return Portfolio(
        date=date,
        symbols=sorted(symbols),
        scores={s: float(score_map.get(s, 0.0)) for s in symbols},
        ranks={s: int(rank_map.get(s, 0)) for s in symbols},
    )


def _trade_context(
    c: Components,
    sym: str,
    *,
    per_stock_capital: float,
) -> dict:
    """Compute reference price, suggested shares, and limit warning for *sym*.

    Returns a dict shaped for `TradeSignal.metadata` consumption.
    Empty dict if we couldn't get T-1 close.
    """
    from datetime import datetime, timedelta

    today = today_str()
    start = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=15)).strftime("%Y-%m-%d")
    bars = c.db.load_daily_bars(sym, start, today)
    if bars is None or bars.empty:
        return {}

    bars = bars.sort_values("date")
    ref_price = float(bars["close"].iloc[-1])
    if not (ref_price > 0):
        return {}

    if sym.startswith("688") or sym.startswith("689"):
        lot = 200
    else:
        lot = 100
    shares = int(per_stock_capital // (ref_price * lot)) * lot
    estimated_amount = shares * ref_price

    ctx: dict[str, Any] = {
        "ref_price": ref_price,
        "shares": shares,
        "estimated_amount": estimated_amount,
        "lot": lot,
    }

    if len(bars) >= 2:
        prev_close = float(bars["close"].iloc[-2])
        if prev_close > 0:
            pct = (ref_price - prev_close) / prev_close
            if pct >= 0.098:
                ctx["limit_warning"] = "⚠️ 昨日触及涨停"
            elif pct <= -0.098:
                ctx["limit_warning"] = "⚠️ 昨日触及跌停"

    return ctx


def _build_cold_start_signals(
    symbols: list[str],
    pool_df: pd.DataFrame,
    *,
    c: Components,
    per_stock_capital: float,
) -> list[TradeSignal]:
    now = _tz_now()
    score_map, rank_map = pool_score_rank_maps(pool_df)
    signals: list[TradeSignal] = []
    for sym in sorted(symbols):
        score = float(score_map.get(sym, 0.0))
        rank = int(rank_map.get(sym, 0))
        ctx = _trade_context(c, sym, per_stock_capital=per_stock_capital)
        ref_price = float(ctx.get("ref_price", 0.0))
        metadata = {
            "cold_start": True,
            "ml_rank": rank,
            "ml_score": score,
            **ctx,
        }
        signals.append(
            TradeSignal(
                timestamp=now,
                symbol=sym,
                name=sym,
                action=Action.BUY,
                signal_status=SignalStatus.NEW_ENTRY,
                days_in_top_n=0,
                price=ref_price,
                confidence=score,
                source="ml_cold_start",
                reason=f"冷启动 - ML rank {rank}",
                factor_score=score,
                metadata=metadata,
            )
        )
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
    c: Components,
    per_stock_capital: float,
) -> list[TradeSignal]:
    now = _tz_now()
    score_map, rank_map = pool_score_rank_maps(pool_df)
    signals: list[TradeSignal] = []

    for sym in final_sells:
        pick = llm_sell_picks.get(sym)
        score = float(score_map.get(sym, 0.0))
        rank = int(rank_map.get(sym, 0))
        # Reference price only — share count is not meaningful for SELL
        # (we don't know the user's actual held quantity at this layer).
        ctx = _trade_context(c, sym, per_stock_capital=per_stock_capital)
        ref_price = float(ctx.get("ref_price", 0.0))
        sell_ctx = {k: v for k, v in ctx.items() if k in {"ref_price", "limit_warning"}}
        if pick is not None:
            reason = pick.reason
            origin = "llm"
            evidence = pick.evidence
        else:
            reason = f"ML 排名兜底卖出 (rank={rank})"
            origin = "fallback"
            evidence = ""
        signals.append(
            TradeSignal(
                timestamp=now,
                symbol=sym,
                name=stock_meta[sym].name if sym in stock_meta else sym,
                action=Action.SELL,
                signal_status=SignalStatus.EXIT,
                days_in_top_n=0,
                price=ref_price,
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
                    **sell_ctx,
                },
            )
        )

    for sym in final_buys:
        pick = llm_buy_picks.get(sym)
        score = float(score_map.get(sym, 0.0))
        rank = int(rank_map.get(sym, 0))
        ctx = _trade_context(c, sym, per_stock_capital=per_stock_capital)
        ref_price = float(ctx.get("ref_price", 0.0))
        if pick is not None:
            reason = pick.reason
            origin = "llm"
            evidence = pick.evidence
        else:
            reason = f"ML 排名兜底买入 (rank={rank})"
            origin = "fallback"
            evidence = ""
        signals.append(
            TradeSignal(
                timestamp=now,
                symbol=sym,
                name=stock_meta[sym].name if sym in stock_meta else sym,
                action=Action.BUY,
                signal_status=SignalStatus.NEW_ENTRY,
                days_in_top_n=0,
                price=ref_price,
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
                    **ctx,
                },
            )
        )
    return signals


# ---------------------------------------------------------------------------
# Persist + push
# ---------------------------------------------------------------------------


async def _persist_and_push(c: Components, signals: list[TradeSignal]) -> None:
    if not signals:
        logger.info("[T6] No signals to push.")
        return
    for sig in signals:
        c.db.save_trade_signal(sig)
    _print_signal_summary(signals)

    signals.sort(key=lambda s: (_ACTION_ORDER.get(s.action, 9), -(s.factor_score or 0)))
    title, content = _build_rebalance_card(signals)
    try:
        result = await c.notif_manager.notify_markdown(title, content)
        logger.info("[T6] Done — %d signals saved, push result=%s", len(signals), result)
    except Exception:  # noqa: BLE001
        logger.warning("[T6] Push failed", exc_info=True)


def _build_rebalance_card(signals: list[TradeSignal]) -> tuple[str, str]:
    """Build a single consolidated markdown card for the rebalance batch."""
    today = signals[0].timestamp.strftime("%Y-%m-%d") if signals else today_str()
    sells = [s for s in signals if s.action is Action.SELL]
    buys = [s for s in signals if s.action is Action.BUY]
    title = f"📋 调仓信号 | {today} | 卖 {len(sells)} / 买 {len(buys)}"

    sections: list[str] = []
    for label, group in (("🔴 卖出", sells), ("🟢 买入", buys)):
        if not group:
            continue
        sections.append(f"## {label} ({len(group)})")
        for sig in group:
            meta = sig.metadata or {}
            rank = meta.get("ml_rank")
            score = meta.get("ml_score")
            rank_str = f"rank={rank}" if rank is not None else ""
            score_str = f"score={score:.3f}" if isinstance(score, (int, float)) else ""
            meta_parts = [p for p in (rank_str, score_str) if p]
            meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""

            warn = meta.get("limit_warning")
            warn_suffix = f" {warn}" if warn else ""
            sections.append(f"- **{sig.name} ({sig.symbol})**{meta_suffix}{warn_suffix}")

            ref_price = meta.get("ref_price")
            shares = meta.get("shares")
            amt = meta.get("estimated_amount")
            if isinstance(ref_price, (int, float)) and ref_price > 0:
                if isinstance(shares, int) and shares > 0 and isinstance(amt, (int, float)):
                    sections.append(f"  - 参考价 ¥{ref_price:,.2f} × {shares} 股 ≈ ¥{amt:,.0f}")
                elif sig.action is Action.SELL:
                    sections.append(f"  - 参考价 ¥{ref_price:,.2f}（按实际持仓数量卖出）")
                else:
                    sections.append(f"  - 参考价 ¥{ref_price:,.2f}（资金不足以买入 1 手）")
            if sig.reason:
                sections.append(f"  - {sig.reason}")
        sections.append("")

    content = "\n".join(sections).rstrip()
    return title, content


# ---------------------------------------------------------------------------
# Data freshness + display helpers
# ---------------------------------------------------------------------------


async def _check_data_freshness(c: Components, today: str) -> None:
    """Alert if key data sources are stale."""
    from datetime import datetime, timedelta

    latest_bar = c.db.get_latest_bar_date()
    if latest_bar is None:
        await c.alert("[T6] 数据库中无 K 线数据，请检查 T3 是否正常")
    else:
        # T6 runs at 08:15 before T3's 18:00 sync, so today's bar legitimately
        # doesn't exist yet. Compare ``latest_bar`` against the *previous*
        # trading day instead of ``today`` to avoid a daily false positive.
        prev_td = c.db.get_trading_day_offset(today, 1)
        if prev_td is not None and latest_bar < prev_td:
            await c.alert(f"[T6] K线数据可能过期: 最新日期 {latest_bar}，期望 {prev_td}，请检查 T3 是否正常")

    latest_news = c.db.get_latest_news_timestamp()
    if latest_news is None:
        await c.alert("[T6] 数据库中无新闻数据，请检查 T2 是否正常")
    else:
        try:
            news_dt = datetime.fromisoformat(latest_news)
            if _tz_now() - news_dt > timedelta(hours=24):
                await c.alert("[T6] 24小时内无新闻更新，请检查 T2 是否正常")
        except (ValueError, TypeError):
            pass

    snapshots = c.db.load_latest_global_snapshots()
    if snapshots.empty:
        await c.alert("[T6] 数据库中无全球行情快照，请检查 T4 是否正常")
    else:
        latest_snap = snapshots["date"].max()
        if isinstance(latest_snap, str) and latest_snap < today:
            await c.alert(
                f"[T6] 全球行情快照不是今日数据 (latest={latest_snap})，T4 可能尚未完成或失败，LLM 将使用上次快照"
            )


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
