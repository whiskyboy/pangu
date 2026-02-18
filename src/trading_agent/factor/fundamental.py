"""Fundamental factor engine — PRD §4.2.2."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from trading_agent.data.fundamental import FundamentalDataProvider

logger = logging.getLogger(__name__)

_FACTOR_NAMES: list[str] = [
    "pe_ttm",
    "pb",
    "roe_ttm",
    "revenue_yoy",
    "profit_yoy",
]


class FundamentalFactorEngine:
    """Cross-sectional fundamental factor computation.

    Calls *provider* per symbol and assembles a DataFrame
    where rows = symbols and columns = factor names.
    """

    def compute(
        self,
        symbols: list[str],
        provider: FundamentalDataProvider,
    ) -> pd.DataFrame:
        """Return a DataFrame (index=symbol, columns=factor_names)."""
        if not symbols:
            return pd.DataFrame(columns=_FACTOR_NAMES)

        rows: list[dict[str, object]] = []
        for sym in symbols:
            row = self._fetch_one(sym, provider)
            rows.append(row)

        df = pd.DataFrame(rows, index=symbols)
        df.index.name = "symbol"
        return df

    def get_factor_names(self) -> list[str]:
        return list(_FACTOR_NAMES)

    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_one(symbol: str, provider: FundamentalDataProvider) -> dict[str, object]:
        """Fetch valuation + financial indicators for a single symbol."""
        record: dict[str, object] = {k: float("nan") for k in _FACTOR_NAMES}

        # Valuation: pe_ttm, pb
        try:
            val = provider.get_valuation(symbol)
            if val is not None:
                for key in ("pe_ttm", "pb"):
                    if key in val and val[key] is not None:
                        record[key] = float(val[key])
        except Exception:  # noqa: BLE001
            logger.warning("Failed to get valuation for %s", symbol)

        # Financial indicators: roe_ttm, revenue_yoy, profit_yoy
        try:
            fin = provider.get_financial_indicator(symbol)
            if fin is not None and not fin.empty:
                latest = fin.iloc[-1]
                for key in ("roe_ttm", "revenue_yoy", "profit_yoy"):
                    if key in latest and latest[key] is not None:
                        record[key] = float(latest[key])
        except Exception:  # noqa: BLE001
            logger.warning("Failed to get financial indicator for %s", symbol)

        return record
