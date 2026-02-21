"""T2: Poll latest telegraph news and store to DB."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


async def poll_news(c: Components) -> None:
    """Fetch latest telegraph news and store to DB."""
    logger.info("[T2] Polling news...")
    try:
        items = c.news.get_latest_news(limit=50)
        logger.info("[T2] Telegraph: %d items fetched", len(items))
    except Exception:  # noqa: BLE001
        logger.warning("[T2] News polling failed", exc_info=True)
    try:
        deleted = c.db.cleanup_old_news(30)
        if deleted:
            logger.info("[T2] Cleaned up %d old news items", deleted)
    except Exception:  # noqa: BLE001
        logger.warning("[T2] News cleanup failed", exc_info=True)
    logger.info("[T2] Done")
