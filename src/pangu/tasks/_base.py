"""Shared scheduled-task plumbing: log → alert → record_task_run."""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pangu.tz import now as _tz_now

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


def scheduled_task(
    task_id: str,
    name: str,
) -> Callable[[Callable[["Components"], Awaitable[None]]], Callable[["Components"], Awaitable[None]]]:
    """Wrap an async task with uniform logging, alerting, and run-history.

    Every wrapped task:

    1. Logs ``[<task_id>] <name> — start`` at the top.
    2. Catches any exception, logs full traceback, and pushes a Chinese alert
       via ``c.alert``.
    3. Persists a row into ``task_runs`` (success/failed + duration_ms) so
       ``pangu status`` can show recent task history.

    Inner functions should focus on business logic only — no outer
    try/except boilerplate, no alert calls for "task failed". Use
    ``c.alert`` inside the inner function only for data-quality / business
    issues (e.g. "K-line stale", "training under-fit"), not generic
    "task threw an exception".
    """

    def deco(
        async_fn: Callable[["Components"], Awaitable[None]],
    ) -> Callable[["Components"], Awaitable[None]]:
        @functools.wraps(async_fn)
        async def wrapper(c: "Components") -> None:
            started_dt = _tz_now()
            started_at = started_dt.strftime("%Y-%m-%d %H:%M:%S")
            logger.info("[%s] %s — start", task_id, name)
            ok, err = True, ""
            try:
                await async_fn(c)
            except Exception as exc:  # noqa: BLE001
                ok, err = False, repr(exc)
                logger.error("[%s] %s — failed", task_id, name, exc_info=True)
                try:
                    await c.alert(f"[{task_id}] {name}异常: {exc}")
                except Exception:  # noqa: BLE001
                    logger.exception("[%s] alert push failed", task_id)
            finally:
                completed_dt = _tz_now()
                completed_at = completed_dt.strftime("%Y-%m-%d %H:%M:%S")
                duration_ms = int(
                    (completed_dt - started_dt).total_seconds() * 1000,
                )
                try:
                    c.db.record_task_run(
                        task_id=task_id,
                        name=name,
                        started_at=started_at,
                        completed_at=completed_at,
                        status="success" if ok else "failed",
                        error_msg=err if err else None,
                        duration_ms=duration_ms,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("[%s] record_task_run failed", task_id)
                logger.info(
                    "[%s] %s — %s in %dms",
                    task_id,
                    name,
                    "done" if ok else "failed",
                    duration_ms,
                )

        return wrapper

    return deco
