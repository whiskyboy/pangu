"""T1: Sync global market snapshot + compute macro factors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


async def sync_global_market(c: Components) -> None:
    """Fetch global market snapshot + compute macro factors."""
    try:
        await _sync_global_market_impl(c)
    except Exception:  # noqa: BLE001
        logger.error("[T1] Global market sync failed", exc_info=True)
        await c.alert("[T1] 全球市场同步任务异常，请检查日志")


async def _sync_global_market_impl(c: Components) -> None:
    """Inner implementation of global market sync."""
    logger.info("[T1] Syncing global market...")
    try:
        snapshot = c.market.get_global_snapshot()
        logger.info("[T1] Global snapshot: %d rows", len(snapshot))
    except Exception:  # noqa: BLE001
        logger.warning("[T1] Global snapshot failed", exc_info=True)
        snapshot = pd.DataFrame()
        await c.alert("[T1] 全球行情获取失败，宏观因子将使用默认值")

    macro_factors = c.macro_engine.compute(snapshot)
    logger.info("[T1] Macro factors: %s",
                {k: round(v, 4) for k, v in macro_factors.items()})
    logger.info("[T1] Done")
