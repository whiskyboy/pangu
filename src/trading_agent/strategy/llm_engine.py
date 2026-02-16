"""LLMEventEngine Protocol — PRD §4.3.2 / §6."""

from __future__ import annotations

from typing import Protocol

from trading_agent.models import NewsItem, TradeSignal


class LLMEventEngine(Protocol):
    """Interface for LLM-powered event-driven signal generation."""

    async def analyze_news(
        self, news: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        """Analyze news items and produce event-driven signals."""
        ...

    async def analyze_announcement(
        self, announcements: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        """Analyze company announcements and produce event-driven signals."""
        ...
