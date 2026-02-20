"""Notification provider protocol."""

from __future__ import annotations

from typing import Protocol

from trading_agent.models import TradeSignal


class NotificationProvider(Protocol):
    """Interface for sending trade signal notifications."""

    async def send(self, signal: TradeSignal) -> bool:
        """Send a trade signal notification. Return True on success."""
        ...
