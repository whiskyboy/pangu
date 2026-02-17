"""LLMEventEngine Protocol + FakeLLMEventEngine — PRD §4.3.2 / §6."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from trading_agent.models import Action, NewsItem, SignalStatus, TradeSignal


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


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------


class FakeLLMEventEngine:
    """Returns deterministic event signals for news containing known symbols."""

    async def analyze_news(
        self, news: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        from trading_agent.tz import now as _now
        now = _now()
        signals: list[TradeSignal] = []
        for item in news:
            for symbol in item.symbols:
                if symbol not in watchlist:
                    continue
                signals.append(TradeSignal(
                    timestamp=now,
                    symbol=symbol,
                    name=symbol,
                    action=Action.BUY,
                    signal_status=SignalStatus.NEW_ENTRY,
                    days_in_top_n=0,
                    price=100.0,
                    confidence=0.8,
                    source="llm_event",
                    reason=f"fake LLM: bullish on '{item.title}'",
                    metadata={"news_title": item.title},
                ))
        return signals

    async def analyze_announcement(
        self, announcements: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        return await self.analyze_news(announcements, watchlist)
