"""Notification provider protocol."""

from __future__ import annotations

from typing import Protocol

from pangu.models import TradeSignal


class NotificationProvider(Protocol):
    """Interface for sending notifications."""

    async def send_signal(self, signal: TradeSignal) -> bool:
        """Send a trade signal notification. Return True on success."""
        ...

    async def send_text(self, text: str) -> bool:
        """Send a plain text message. Return True on success."""
        ...

    async def send_markdown(self, title: str, content: str) -> bool:
        """Send a card with markdown content. Return True on success."""
        ...
