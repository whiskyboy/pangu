"""T5: Sync trading calendar and index constituents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


async def sync_reference_data(c: Components) -> None:
    """Sync trading calendar and index constituents."""
    logger.info("[T5] Syncing reference data...")
    try:
        count = c.stock_pool.sync_trading_calendar()
        logger.info("[T5] Calendar: %d new dates synced", count)
    except Exception:  # noqa: BLE001
        logger.warning("[T5] Calendar sync failed", exc_info=True)
        await c.alert("[T5] 交易日历同步失败")
    try:
        count = c.stock_pool.sync_index_constituents()
        logger.info("[T5] Index constituents: %d synced", count)
    except Exception:  # noqa: BLE001
        logger.warning("[T5] Index constituents sync failed", exc_info=True)
        await c.alert("[T5] 指数成分股同步失败")
    logger.info("[T5] Done")
