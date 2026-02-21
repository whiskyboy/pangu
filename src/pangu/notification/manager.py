"""Notification manager — multi-channel concurrent dispatch."""

from __future__ import annotations

import asyncio
import logging

from pangu.models import TradeSignal
from pangu.notification.protocol import NotificationProvider

logger = logging.getLogger(__name__)


class NotificationManager:
    """Dispatch notifications to all registered channels."""

    def __init__(self, channels: list[NotificationProvider] | None = None) -> None:
        self._channels: list[NotificationProvider] = channels or []

    def add_channel(self, channel: NotificationProvider) -> None:
        self._channels.append(channel)

    async def _dispatch(self, method: str, *args: object) -> dict[str, bool]:
        """Call *method* on every channel concurrently."""
        if not self._channels:
            logger.warning("NotificationManager: no channels configured")
            return {}
        tasks = [getattr(ch, method)(*args) for ch in self._channels]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outcome: dict[str, bool] = {}
        for ch, result in zip(self._channels, results):
            name = type(ch).__name__
            if isinstance(result, BaseException):
                logger.error("Channel %s raised: %s", name, result)
                outcome[name] = False
            else:
                outcome[name] = bool(result)
        return outcome

    async def notify_signal(self, signal: TradeSignal) -> dict[str, bool]:
        """Send a trade signal to all channels."""
        return await self._dispatch("send_signal", signal)

    async def notify_text(self, text: str) -> dict[str, bool]:
        """Send plain text to all channels."""
        return await self._dispatch("send_text", text)

    async def notify_markdown(self, title: str, content: str) -> dict[str, bool]:
        """Send a markdown card to all channels."""
        return await self._dispatch("send_markdown", title, content)
