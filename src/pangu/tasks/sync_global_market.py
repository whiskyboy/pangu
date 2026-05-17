"""T4: Sync overnight global market snapshot to DB.

The persisted snapshot is consumed by T6 as ``global_market`` LLM context
(via ``Database.load_latest_global_snapshots()``) — purely market data, no
derived macro factors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from pangu.tasks._base import scheduled_task

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


@scheduled_task("T4", "全球行情同步")
async def sync_global_market(c: Components) -> None:
    """Fetch overnight global market snapshot and persist via the provider."""
    try:
        snapshot = c.market.get_global_snapshot()
        logger.info("[T4] Global snapshot: %d rows", len(snapshot))
    except Exception:  # noqa: BLE001
        logger.warning("[T4] Global snapshot failed", exc_info=True)
        snapshot = pd.DataFrame()
        await c.alert("[T4] 全球行情获取失败，T6 调仓将使用上一次缓存")
    logger.info("[T4] Done (%d snapshot rows)", len(snapshot))
