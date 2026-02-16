"""FactorEngine Protocol — PRD §4.2 / §6."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class FactorEngine(Protocol):
    """Interface for computing technical / fundamental factor scores."""

    def compute(
        self, bars: pd.DataFrame, global_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Compute factor values; *global_data* is the international snapshot for macro factors."""
        ...

    def get_factor_names(self) -> list[str]:
        """Return the list of factor names this engine computes."""
        ...
