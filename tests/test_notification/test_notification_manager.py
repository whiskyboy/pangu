"""Tests for NotificationManager (text + markdown fan-out)."""

from __future__ import annotations

import asyncio

from pangu.notification import NotificationManager


class _SuccessChannel:
    async def send_text(self, text: str) -> bool:
        return True

    async def send_markdown(self, title: str, content: str) -> bool:
        return True


class _FailChannel:
    async def send_text(self, text: str) -> bool:
        return False

    async def send_markdown(self, title: str, content: str) -> bool:
        return False


class _ErrorChannel:
    async def send_text(self, text: str) -> bool:
        raise RuntimeError("boom")

    async def send_markdown(self, title: str, content: str) -> bool:
        raise RuntimeError("boom")


class TestNotificationManager:
    def test_no_channels(self) -> None:
        mgr = NotificationManager()
        assert asyncio.run(mgr.notify_text("hello")) == {}
        assert asyncio.run(mgr.notify_markdown("title", "body")) == {}

    def test_single_success_channel_text(self) -> None:
        mgr = NotificationManager([_SuccessChannel()])
        result = asyncio.run(mgr.notify_text("hi"))
        assert result == {"_SuccessChannel": True}

    def test_single_success_channel_markdown(self) -> None:
        mgr = NotificationManager([_SuccessChannel()])
        result = asyncio.run(mgr.notify_markdown("title", "body"))
        assert result == {"_SuccessChannel": True}

    def test_single_fail_channel(self) -> None:
        mgr = NotificationManager([_FailChannel()])
        result = asyncio.run(mgr.notify_text("hi"))
        assert result == {"_FailChannel": False}

    def test_multiple_channels(self) -> None:
        mgr = NotificationManager([_SuccessChannel(), _FailChannel()])
        result = asyncio.run(mgr.notify_text("hi"))
        assert result["_SuccessChannel"] is True
        assert result["_FailChannel"] is False

    def test_exception_channel_returns_false(self) -> None:
        mgr = NotificationManager([_ErrorChannel()])
        result = asyncio.run(mgr.notify_text("hi"))
        assert result["_ErrorChannel"] is False

    def test_add_channel(self) -> None:
        mgr = NotificationManager()
        mgr.add_channel(_SuccessChannel())
        result = asyncio.run(mgr.notify_markdown("title", "body"))
        assert result == {"_SuccessChannel": True}

    def test_mixed_success_and_error(self) -> None:
        mgr = NotificationManager([_SuccessChannel(), _ErrorChannel()])
        result = asyncio.run(mgr.notify_markdown("title", "body"))
        assert result["_SuccessChannel"] is True
        assert result["_ErrorChannel"] is False
