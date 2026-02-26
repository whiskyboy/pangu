"""T3: Sync daily K-lines + fundamentals for the full stock pool."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pangu.tz import today_str
from pangu.utils import date_str

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)

_BARS_LOOKBACK_DAYS = 1200
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

    # CSI300 index daily bars (for benchmark/label calculation)
    try:
        c.market.get_index_daily_bars("000300", start, today)
        logger.info("[T3] CSI300 index bars synced")
    except Exception:  # noqa: BLE001
        logger.warning("[T3] CSI300 index sync failed", exc_info=True)

    # Fundamentals — financial indicators (monthly)
    fi_ok, fi_fail = 0, 0
    for symbol in pool:
        try:
            result = c.fundamental.get_financial_indicator(symbol)
            if result is not None and not result.empty:
                fi_ok += 1
            else:
                fi_fail += 1
        except Exception:  # noqa: BLE001
            fi_fail += 1
            logger.warning("[T3] %s: financial indicator failed", symbol, exc_info=True)
    logger.info("[T3] Financial indicators: %d synced, %d failed", fi_ok, fi_fail)

    logger.info("[T3] Done")
