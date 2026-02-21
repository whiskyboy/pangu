"""T3: Sync daily K-lines + fundamentals for watchlist + CSI300."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trading_agent.tz import today_str
from trading_agent.utils import date_str

if TYPE_CHECKING:
    from trading_agent.scheduler import Components

logger = logging.getLogger(__name__)

_BARS_LOOKBACK_DAYS = 200
_BARS_FAIL_THRESHOLD = 0.5


async def sync_domestic_market(c: Components) -> None:
    """Sync daily K-lines + fundamentals for watchlist + CSI300."""
    logger.info("[T3] Syncing domestic market...")
    pool = c.stock_pool.get_factor_universe()
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
