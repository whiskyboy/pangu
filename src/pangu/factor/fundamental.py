"""Fundamental factor engine — PRD §4.2.2."""

from __future__ import annotations

import pandas as pd

_FACTOR_NAMES: list[str] = [
    "pe_ttm",
    "pb",
    "roe_ttm",
    "revenue_yoy",
    "profit_yoy",
]


class FundamentalFactorEngine:
    """Cross-sectional fundamental factor computation.

    Pure function: accepts a DataFrame already loaded from SQLite,
    returns a validated DataFrame with factor columns.
    """

    def compute(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame (index=symbol, columns=factor_names).

        *fundamentals_df* must have a ``symbol`` column (or index) and
        any subset of the factor columns.  Missing columns are filled
        with NaN.
        """
        if fundamentals_df is None or fundamentals_df.empty:
            return pd.DataFrame(columns=_FACTOR_NAMES)

        df = fundamentals_df.copy()

        # Normalise index: if "symbol" is a column, set it as index
        if "symbol" in df.columns:
            df = df.set_index("symbol")
        if df.index.name != "symbol":
            df.index.name = "symbol"

        # Ensure all factor columns exist, fill missing with NaN
        for col in _FACTOR_NAMES:
            if col not in df.columns:
                df[col] = float("nan")

        return df[_FACTOR_NAMES].astype(float)

    def get_factor_names(self) -> list[str]:
        return list(_FACTOR_NAMES)
