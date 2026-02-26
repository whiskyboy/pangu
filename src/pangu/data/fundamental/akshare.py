"""AkShare FundamentalDataProvider — PRD §4.1.2."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from pangu.utils import CircuitBreaker, ThrottleMixin, retry_call

logger = logging.getLogger(__name__)


class AkShareFundamentalProvider(ThrottleMixin):
    """Pure AkShare API provider for fundamental data.

    API mapping:
    - get_financial_indicator → stock_financial_analysis_indicator

    This provider is **stateless** — it performs API calls only and
    returns results.  Caching and persistence are handled by
    :class:`CompositeFundamentalProvider`.

    Parameters
    ----------
    request_interval : float
        Minimum seconds between API calls (default 0.5).
    """

    def __init__(self, request_interval: float = 0.5) -> None:
        import akshare  # lazy import for test mocking

        self._ak = akshare
        self.__init_throttle__(request_interval)
        self._circuit = CircuitBreaker()

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert to float, returning None for empty/NaN/invalid values."""
        if value is None or value == "":
            return None
        try:
            f = float(value)
            return None if math.isnan(f) else f
        except (ValueError, TypeError):
            return None

    def _fetch_financial_indicator(self, symbol: str, start_year: str | None = None) -> pd.DataFrame:
        """Fetch financial analysis indicators via stock_financial_analysis_indicator."""
        self._throttle()

        from pangu.tz import now as _now
        if start_year is None:
            start_year = str(_now().year - 1)
        df = retry_call(
            lambda: self._ak.stock_financial_analysis_indicator(
                symbol=symbol, start_year=start_year,
            ),
            circuit=self._circuit,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    # -- Protocol methods --

    def get_financial_indicator(
        self, symbol: str, start: str | None = None, end: str | None = None,
    ) -> pd.DataFrame:
        """Return standardized financial indicators DataFrame."""
        try:
            start_year = start[:4] if start else None
            fin = self._fetch_financial_indicator(symbol, start_year=start_year)
            if fin.empty:
                return pd.DataFrame()

            rows = []
            for _, raw in fin.iterrows():
                row: dict[str, Any] = {
                    "symbol": symbol,
                    "date": str(raw.get("日期", "")),
                }
                # ROE
                roe = self._safe_float(raw.get("净资产收益率(%)"))
                row["roe_ttm"] = roe / 100 if roe is not None else None

                # Revenue YoY growth
                rev_growth = self._safe_float(raw.get("主营业务收入增长率(%)"))
                row["revenue_yoy"] = (
                    rev_growth / 100 if rev_growth is not None else None
                )

                # Net profit YoY growth
                profit_growth = self._safe_float(raw.get("净利润增长率(%)"))
                row["profit_yoy"] = (
                    profit_growth / 100 if profit_growth is not None else None
                )

                rows.append(row)

            result = pd.DataFrame(rows)

            # Filter by date range if provided
            if end is not None and "date" in result.columns:
                result = result[result["date"] <= end]

            return result

        except Exception:  # noqa: BLE001
            logger.warning(
                "get_financial_indicator(%s) failed", symbol, exc_info=True,
            )
            return pd.DataFrame()
