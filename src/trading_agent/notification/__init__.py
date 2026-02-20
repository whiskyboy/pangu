"""Notification subsystem — multi-channel concurrent dispatch."""

from trading_agent.notification.protocol import NotificationProvider
from trading_agent.notification.manager import NotificationManager

__all__ = ["NotificationProvider", "NotificationManager"]
