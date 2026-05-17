"""T1: Sync trading calendar and index constituents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pangu.tasks._base import scheduled_task

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


@scheduled_task("T1", "参考数据同步")
async def sync_reference_data(c: Components) -> None:
    """Sync trading calendar and index constituents."""
    try:
        count = c.stock_pool.sync_trading_calendar()
        logger.info("[T1] Calendar: %d new dates synced", count)
    except Exception:  # noqa: BLE001
        logger.warning("[T1] Calendar sync failed", exc_info=True)
        await c.alert("[T1] 交易日历同步失败")
    try:
        count = c.stock_pool.sync_index_constituents()
        logger.info("[T1] Index constituents: %d synced", count)
    except Exception:  # noqa: BLE001
        logger.warning("[T1] Index constituents sync failed", exc_info=True)
        await c.alert("[T1] 指数成分股同步失败")
