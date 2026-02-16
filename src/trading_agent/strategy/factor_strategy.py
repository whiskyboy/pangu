"""Strategy Protocol — PRD §4.3.1 / §6."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from trading_agent.models import TradeSignal


class Strategy(Protocol):
    """Interface for factor-based signal generation."""

    def generate_signals(
        self, data: pd.DataFrame, pool: list[str]
    ) -> list[TradeSignal]:
        """Generate trade signals from factor data for the given stock pool."""
        ...
