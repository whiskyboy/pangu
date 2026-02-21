"""NewsDataProvider Protocol — PRD §4.1.3 / §6."""

from __future__ import annotations

from typing import Protocol

from pangu.models import NewsItem


class NewsDataProvider(Protocol):
    """Interface for domestic and international financial news."""

    def get_latest_news(self, limit: int = 50) -> list[NewsItem]:
        """Return latest domestic financial news."""
        ...

    def get_stock_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Return news related to a specific stock."""
        ...

    def get_announcements(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Return recent announcements for a specific stock (巨潮)."""
        ...
