"""Strategy Protocol + FakeFactorStrategy — PRD §4.3.1 / §6."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from trading_agent.models import Action, SignalStatus, TradeSignal


class Strategy(Protocol):
    """Interface for factor-based signal generation."""

    def generate_signals(
        self, data: pd.DataFrame, pool: list[str]
    ) -> list[TradeSignal]:
        """Generate trade signals from factor data for the given stock pool."""
        ...


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------

_FAKE_SCORES: dict[str, float] = {
    "600519": 0.82,
    "000858": 0.65,
    "300750": 0.75,
    "601318": 0.25,
    "000001": 0.45,
}


class FakeFactorStrategy:
    """Generates deterministic signals based on hardcoded factor scores."""

    def generate_signals(
        self, data: pd.DataFrame, pool: list[str]
    ) -> list[TradeSignal]:
        from trading_agent.tz import now as _now
        now = _now()
        signals: list[TradeSignal] = []
        for symbol in pool:
            score = _FAKE_SCORES.get(symbol, 0.5)
            if score >= 0.7:
                action = Action.BUY
                confidence = score
            elif score <= 0.3:
                action = Action.SELL
                confidence = 1.0 - score
            else:
                continue  # no signal for HOLD-range scores
            signals.append(TradeSignal(
                timestamp=now,
                symbol=symbol,
                name=symbol,
                action=action,
                signal_status=SignalStatus.NEW_ENTRY,
                days_in_top_n=0,
                price=100.0,
                confidence=confidence,
                source="factor",
                reason=f"fake factor score={score:.2f}",
                factor_score=score,
            ))
        return signals
