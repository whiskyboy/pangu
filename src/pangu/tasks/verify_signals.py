"""T6: Post-verify historical signals against actual price changes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pangu.tz import today_str

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)

_LOOKBACKS = (1, 3, 5)
_ACTION_LABEL = {"BUY": "买入", "SELL": "卖出"}


async def verify_signals(c: Components) -> None:
    """Verify 1/3/5 trading-day old signals against actual closing prices."""
    today = today_str()
    total_verified = 0
    # {lookback: [{"name", "symbol", "action", "return_pct", "correct", "signal_date"}, ...]}
    grouped: dict[int, list[dict]] = {}

    for lb in _LOOKBACKS:
        signal_date = c.db.get_trading_day_offset(today, lb)
        if signal_date is None:
            logger.info("[T6] Not enough calendar data for %dd lookback", lb)
            continue

        signals = c.db.load_unverified_signals(signal_date, lb)
        if not signals:
            logger.info("[T6] No unverified %dd signals for %s", lb, signal_date)
            continue

        logger.info("[T6] Verifying %d signals from %s (%dd)", len(signals), signal_date, lb)
        items: list[dict] = []

        for sig in signals:
            close = _get_latest_close(c, sig["symbol"], today)
            if close is None or sig["price"] is None or sig["price"] == 0:
                logger.warning("[T6] No price data for %s, skipping", sig["symbol"])
                continue

            return_pct = (close - sig["price"]) / sig["price"] * 100
            c.db.update_signal_return(sig["id"], lb, round(return_pct, 2))

            is_correct = (
                (sig["action"] == "BUY" and return_pct > 0)
                or (sig["action"] == "SELL" and return_pct < 0)
            )
            items.append({
                "name": sig["name"],
                "symbol": sig["symbol"],
                "action": sig["action"],
                "return_pct": round(return_pct, 2),
                "correct": is_correct,
                "signal_date": signal_date,
            })

        if items:
            grouped[lb] = items
            total_verified += len(items)
            correct = sum(1 for i in items if i["correct"])
            logger.info("[T6] %dd: %d/%d correct", lb, correct, len(items))

    # Push summary to Feishu
    if grouped and c.notif_manager is not None:
        returns = c.db.get_signal_returns(30)
        markdown = _build_report_markdown(grouped, returns)
        await c.notif_manager.notify_markdown(
            f"📋 信号验证报告 | {today}", markdown,
        )
        logger.info("[T6] Verification report pushed to Feishu")

    logger.info("[T6] Done — %d signals verified", total_verified)


def _build_report_markdown(
    grouped: dict[int, list[dict]],
    returns: dict[str, dict],
) -> str:
    """Build markdown content for the verification card."""
    sections: list[str] = []

    for lb in _LOOKBACKS:
        items = grouped.get(lb)
        if not items:
            continue

        signal_date = items[0]["signal_date"]
        # Compute strategy return for this batch
        strat_returns = [
            it["return_pct"] if it["action"] == "BUY" else -it["return_pct"]
            for it in items
        ]
        avg_ret = sum(strat_returns) / len(strat_returns)
        emoji_avg = "📈" if avg_ret >= 0 else "📉"

        lines = [f"**📅 {lb}日验证** (信号日期: {signal_date}，策略收益: {emoji_avg} {avg_ret:+.2f}%)"]
        for it in items:
            emoji = "✅" if it["correct"] else "❌"
            label = _ACTION_LABEL.get(it["action"], it["action"])
            lines.append(f"{emoji} {it['name']}({it['symbol']}) {label} → {it['return_pct']:+.2f}%")
        sections.append("\n".join(lines))

    # Rolling returns stats
    stats: list[str] = []
    for lb in _LOOKBACKS:
        key = f"{lb}d"
        if key in returns and returns[key]["count"] > 0:
            r = returns[key]
            emoji = "📈" if r["avg_return"] >= 0 else "📉"
            stats.append(f"**{key.upper()}**: {emoji} {r['avg_return']:+.2f}% ({r['count']}笔)")
    if stats:
        sections.append("📊 **30日滚动策略收益**\n" + " · ".join(stats))

    return "\n\n".join(sections)


def _get_latest_close(c: Components, symbol: str, today: str) -> float | None:
    """Get the latest closing price for *symbol* from daily_bars."""
    df = c.db.load_daily_bars(symbol, "2020-01-01", today)
    if df.empty:
        return None
    val = df["close"].iloc[-1]
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return None
    return float(val)
