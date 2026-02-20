"""Strategy Protocol — PRD §4.3.1 / §6."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from trading_agent.models import TradeSignal


class Strategy(Protocol):
    """Interface for factor-based signal generation."""

    def generate_signals(
        self,
        tech_df: dict[str, pd.DataFrame],
        fund_df: pd.DataFrame,
        macro_factors: dict[str, float],
        *,
        prev_pool: pd.DataFrame | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, list[TradeSignal]]:
        """Generate trade signals from pre-computed factor data.

        Returns:
            (factor_pool DataFrame, list of TradeSignal)
        """
        ...


# Backward-compatible re-exports
from trading_agent.strategy.multi_factor import (  # noqa: E402, F401
    MultiFactorStrategy,
    _DEFAULT_WEIGHTS,
    _minmax_normalize,
    _weighted_score,
    _zscore_normalize,
)


