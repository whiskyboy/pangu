"""T2: Poll latest telegraph news and store to DB."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pangu.tasks._base import scheduled_task

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


@scheduled_task("T2", "新闻轮询")
async def poll_news(c: Components) -> None:
    """Fetch latest telegraph news and store to DB."""
    items = c.news.get_latest_news(limit=50)
    logger.info("[T2] Telegraph: %d items fetched", len(items))
    deleted = c.db.cleanup_old_news(30)
    if deleted:
        logger.info("[T2] Cleaned up %d old news items", deleted)
    try:
        cleaned = c.db.cleanup_old_task_runs(30)
        if cleaned:
            logger.info("[T2] Cleaned up %d old task_run rows", cleaned)
    except Exception:  # noqa: BLE001
        logger.warning("[T2] task_runs cleanup failed", exc_info=True)
