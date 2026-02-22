"""T3: Sync daily K-lines + fundamentals for the full stock pool."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pangu.tz import today_str
from pangu.utils import date_str

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)

_BARS_LOOKBACK_DAYS = 200
_BARS_FAIL_THRESHOLD = 0.5


async def sync_domestic_market(c: Components) -> None:
    """Sync daily K-lines + fundamentals for the full stock pool."""
    try:
        await _sync_domestic_market_impl(c)
    except Exception:  # noqa: BLE001
        logger.error("[T3] Domestic market sync failed", exc_info=True)
        await c.alert("[T3] 国内行情同步任务异常，请检查日志")


async def _sync_domestic_market_impl(c: Components) -> None:
    """Inner implementation of domestic market sync."""
    logger.info("[T3] Syncing domestic market...")
    pool = c.stock_pool.get_all_symbols()
    today = today_str()
    start = date_str(days_ago=_BARS_LOOKBACK_DAYS)
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
    if total > 0 and fail > total * _BARS_FAIL_THRESHOLD:
        await c.alert(f"[T3] 行情同步大面积失败: {fail}/{total} 股票获取失败")

    # Fundamentals — valuation (daily)
    ok, fail = 0, 0
    for symbol in pool:
        try:
            c.fundamental.get_valuation(symbol)
            ok += 1
        except Exception:  # noqa: BLE001
            fail += 1
            logger.warning("[T3] %s: fundamentals failed", symbol, exc_info=True)
    logger.info("[T3] Fundamentals (valuation): %d ok, %d failed", ok, fail)

    # Fundamentals — financial indicators (monthly)
    _sync_financial_indicators(c, pool)

    logger.info("[T3] Done")


_FINANCIAL_SYNC_INTERVAL_DAYS = 30


def _sync_financial_indicators(c: Components, pool: list[str]) -> None:
    """Sync financial indicators for stocks not updated within 30 days."""
    from datetime import datetime, timedelta

    from pangu.tz import now as tz_now

    cutoff = (tz_now() - timedelta(days=_FINANCIAL_SYNC_INTERVAL_DAYS)).strftime(
        "%Y-%m-%d"
    )
    ok, skip, fail = 0, 0, 0
    for symbol in pool:
        last = c.db.get_last_sync_date(symbol, "financial_indicator")
        if last is not None and last >= cutoff:
            skip += 1
            continue
        try:
            result = c.fundamental.get_financial_indicator(symbol)
            if result is not None and not result.empty:
                c.db.update_sync_log(
                    symbol, "financial_indicator", "ok", "akshare",
                    last_date=tz_now().strftime("%Y-%m-%d"),
                )
                ok += 1
            else:
                fail += 1
        except Exception:  # noqa: BLE001
            fail += 1
            logger.warning("[T3] %s: financial indicator failed", symbol, exc_info=True)
    logger.info(
        "[T3] Financial indicators: %d synced, %d skipped (recent), %d failed",
        ok, skip, fail,
    )
