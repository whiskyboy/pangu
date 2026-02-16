"""Tests for NotificationManager (M1.6)."""

from __future__ import annotations

import asyncio
from datetime import datetime

from trading_agent.models import Action, SignalStatus, TradeSignal
from trading_agent.notification import NotificationManager


def _signal() -> TradeSignal:
    return TradeSignal(
        timestamp=datetime(2026, 2, 16, 10, 0),
        symbol="600519",
        name="贵州茅台",
        action=Action.BUY,
        signal_status=SignalStatus.NEW_ENTRY,
        days_in_top_n=0,
        price=1800.0,
        confidence=0.8,
        source="merged",
        reason="test",
    )


class _SuccessChannel:
    async def send(self, signal: TradeSignal) -> bool:
        return True


class _FailChannel:
    async def send(self, signal: TradeSignal) -> bool:
        return False


class _ErrorChannel:
    async def send(self, signal: TradeSignal) -> bool:
        raise RuntimeError("boom")


class TestNotificationManager:
    def test_no_channels(self) -> None:
        mgr = NotificationManager()
        result = asyncio.run(mgr.notify(_signal()))
        assert result == {}

    def test_single_success_channel(self) -> None:
        mgr = NotificationManager([_SuccessChannel()])
        result = asyncio.run(mgr.notify(_signal()))
        assert result == {"_SuccessChannel": True}

    def test_single_fail_channel(self) -> None:
        mgr = NotificationManager([_FailChannel()])
        result = asyncio.run(mgr.notify(_signal()))
        assert result == {"_FailChannel": False}

    def test_multiple_channels(self) -> None:
        mgr = NotificationManager([_SuccessChannel(), _FailChannel()])
        result = asyncio.run(mgr.notify(_signal()))
        assert result["_SuccessChannel"] is True
        assert result["_FailChannel"] is False

    def test_exception_channel_returns_false(self) -> None:
        mgr = NotificationManager([_ErrorChannel()])
        result = asyncio.run(mgr.notify(_signal()))
        assert result["_ErrorChannel"] is False

    def test_add_channel(self) -> None:
        mgr = NotificationManager()
        mgr.add_channel(_SuccessChannel())
        result = asyncio.run(mgr.notify(_signal()))
        assert result == {"_SuccessChannel": True}

    def test_mixed_success_and_error(self) -> None:
        mgr = NotificationManager([_SuccessChannel(), _ErrorChannel()])
        result = asyncio.run(mgr.notify(_signal()))
        assert result["_SuccessChannel"] is True
        assert result["_ErrorChannel"] is False
