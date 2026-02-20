"""T5: Sync trading calendar and CSI300 index constituents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_agent.scheduler import Components

logger = logging.getLogger(__name__)


async def sync_reference_data(c: Components) -> None:
    """Sync trading calendar and CSI300 index constituents."""
    logger.info("[T5] Syncing reference data...")
    try:
        count = c.stock_pool.sync_trading_calendar()
        logger.info("[T5] Calendar: %d new dates synced", count)
    except Exception:  # noqa: BLE001
        logger.warning("[T5] Calendar sync failed", exc_info=True)
    try:
        count = c.stock_pool.sync_csi300_constituents()
        logger.info("[T5] CSI300: %d constituents synced", count)
    except Exception:  # noqa: BLE001
        logger.warning("[T5] CSI300 sync failed", exc_info=True)
    logger.info("[T5] Done")
