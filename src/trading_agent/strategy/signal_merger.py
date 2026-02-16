"""SignalMerger Protocol — PRD §4.3.4 / §6."""

from __future__ import annotations

from typing import Protocol

from trading_agent.models import TradeSignal


class SignalMerger(Protocol):
    """Interface for merging signals from multiple strategy engines."""

    def merge(
        self,
        factor_signals: list[TradeSignal],
        event_signals: list[TradeSignal],
        anomaly_signals: list[TradeSignal],
        news_availability: dict[str, bool],
    ) -> list[TradeSignal]:
        """Merge and arbitrate signals using weighted scoring."""
        ...
