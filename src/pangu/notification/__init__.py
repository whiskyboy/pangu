"""Notification subsystem — multi-channel concurrent dispatch."""

from pangu.notification.protocol import NotificationProvider
from pangu.notification.manager import NotificationManager

__all__ = ["NotificationProvider", "NotificationManager"]
