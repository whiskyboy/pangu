"""Notification subsystem — multi-channel concurrent dispatch."""

from __future__ import annotations

import asyncio
import logging

from trading_agent.models import TradeSignal
from trading_agent.notification.feishu import NotificationProvider

logger = logging.getLogger(__name__)


class NotificationManager:
    """Dispatch trade signals to all registered notification channels."""

    def __init__(self, channels: list[NotificationProvider] | None = None) -> None:
        self._channels: list[NotificationProvider] = channels or []

    def add_channel(self, channel: NotificationProvider) -> None:
        self._channels.append(channel)

    async def notify(self, signal: TradeSignal) -> dict[str, bool]:
        """Send *signal* to all channels concurrently. Return {channel_name: success}."""
        if not self._channels:
            logger.warning("NotificationManager: no channels configured")
            return {}

        tasks = [ch.send(signal) for ch in self._channels]
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
