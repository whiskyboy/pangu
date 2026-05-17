"""Tests for the @scheduled_task decorator (src/pangu/tasks/_base.py).

The decorator owns four cross-cutting concerns:

1. Log start banner
2. Catch any exception → log + alert
3. Record outcome (success / failed + duration) in ``task_runs``
4. Re-raise from ``c.alert`` failures are swallowed (not propagated)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pangu.tasks._base import scheduled_task


def _make_components():
    c = MagicMock()
    c.alert = AsyncMock()
    c.db = MagicMock()
    c.db.record_task_run = MagicMock()
    return c


class TestSuccessPath:
    @pytest.mark.asyncio
    async def test_inner_function_invoked_with_components(self):
        c = _make_components()
        seen: list = []

        @scheduled_task("T4", "全球行情同步")
        async def inner(comp):
            seen.append(comp)

        await inner(c)

        assert seen == [c]

    @pytest.mark.asyncio
    async def test_records_success_run(self):
        c = _make_components()

        @scheduled_task("T4", "全球行情同步")
        async def inner(_c):
            return

        await inner(c)

        c.db.record_task_run.assert_called_once()
        kwargs = c.db.record_task_run.call_args.kwargs
        assert kwargs["task_id"] == "T4"
        assert kwargs["name"] == "全球行情同步"
        assert kwargs["status"] == "success"
        assert kwargs["error_msg"] is None
        assert kwargs["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_no_alert_on_success(self):
        c = _make_components()

        @scheduled_task("T4", "全球行情同步")
        async def inner(_c):
            return

        await inner(c)
        c.alert.assert_not_awaited()


class TestFailurePath:
    @pytest.mark.asyncio
    async def test_records_failed_run_with_error_msg(self):
        c = _make_components()

        @scheduled_task("T4", "全球行情同步")
        async def inner(_c):
            raise RuntimeError("network down")

        # Decorator catches the exception — should not raise
        await inner(c)

        c.db.record_task_run.assert_called_once()
        kwargs = c.db.record_task_run.call_args.kwargs
        assert kwargs["status"] == "failed"
        assert "RuntimeError" in kwargs["error_msg"]
        assert "network down" in kwargs["error_msg"]

    @pytest.mark.asyncio
    async def test_alerts_on_failure(self):
        c = _make_components()

        @scheduled_task("T3", "国内行情/基本面同步")
        async def inner(_c):
            raise ValueError("bad data")

        await inner(c)

        c.alert.assert_awaited_once()
        msg = c.alert.await_args.args[0]
        assert "T3" in msg
        assert "国内行情/基本面同步" in msg
        assert "bad data" in msg

    @pytest.mark.asyncio
    async def test_alert_failure_is_swallowed(self):
        c = _make_components()
        c.alert = AsyncMock(side_effect=Exception("alert push failed"))

        @scheduled_task("T4", "test")
        async def inner(_c):
            raise RuntimeError("boom")

        # Both errors are swallowed — record_task_run still called
        await inner(c)
        c.db.record_task_run.assert_called_once()
        kwargs = c.db.record_task_run.call_args.kwargs
        assert kwargs["status"] == "failed"

    @pytest.mark.asyncio
    async def test_db_record_failure_is_swallowed(self):
        c = _make_components()
        c.db.record_task_run = MagicMock(side_effect=Exception("db down"))

        @scheduled_task("T4", "test")
        async def inner(_c):
            return

        # Should not raise even when DB write fails
        await inner(c)


class TestPreservesMetadata:
    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        @scheduled_task("T4", "test")
        async def my_task(_c):
            return

        # functools.wraps preserves __name__
        assert my_task.__name__ == "my_task"
