"""AnomalyDetector Protocol — PRD §4.3.3 / §6."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from trading_agent.models import TradeSignal


class AnomalyDetector(Protocol):
    """Interface for detecting unusual market activity."""

    def detect(
        self,
        a_share_quotes: pd.DataFrame,
        global_quotes: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        """Detect anomalies: volume ratio >3, price change >5%, etc."""
        ...
