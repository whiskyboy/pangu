"""Notification subsystem — multi-channel concurrent dispatch."""

from pangu.notification.manager import NotificationManager
from pangu.notification.protocol import NotificationProvider

__all__ = ["NotificationProvider", "NotificationManager"]
