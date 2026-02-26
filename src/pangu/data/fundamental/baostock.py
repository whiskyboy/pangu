"""BaoStock FundamentalDataProvider — quarterly financial indicators."""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class BaoStockFundamentalProvider:
    """Pure BaoStock API provider for quarterly financial indicators.

    API mapping:
    - get_financial_indicator → query_profit_data + query_growth_data

    This provider is **stateless** — it performs API calls only and
    returns DataFrames.  Caching and persistence are handled by
    :class:`CompositeFundamentalProvider`.
    """

    def __init__(self) -> None:
        import baostock as bs

        self._bs = bs
        self._logged_in = False
        self._login_lock = threading.Lock()

    def _ensure_login(self) -> None:
        with self._login_lock:
            if not self._logged_in:
                lg = self._bs.login()
                if lg.error_code != "0":
                    raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
                self._logged_in = True

    def close(self) -> None:
        with self._login_lock:
            if self._logged_in:
                self._bs.logout()
                self._logged_in = False

    def __del__(self) -> None:
        try:
            self.close()
        except AttributeError:
            pass

    @staticmethod
    def _to_bs_code(symbol: str) -> str:
        if symbol.startswith(("sh.", "sz.")):
            return symbol
        first = symbol[0]
        if first == "6":
            return f"sh.{symbol}"
        if first in ("0", "3"):
            return f"sz.{symbol}"
        raise ValueError(f"BaoStock does not support: {symbol}")

    def _query_to_df(self, rs) -> pd.DataFrame:
        rows: list[list] = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            f = float(value)
            return None if math.isnan(f) else f
        except (ValueError, TypeError):
            return None

    # -- Protocol methods --

    def get_financial_indicator(
        self, symbol: str, start: str | None = None, end: str | None = None,
    ) -> pd.DataFrame:
        """Return ROE, revenue_yoy, profit_yoy from BaoStock quarterly data.

        Parameters
        ----------
        start : str | None
            Start date (YYYY-MM-DD). Determines start_year for BaoStock query.
            Defaults to previous year.
        end : str | None
            End date (YYYY-MM-DD). Filters results. Defaults to current date.
        """
        try:
            return self._fetch_financial(symbol, start=start, end=end)
        except Exception:  # noqa: BLE001
            logger.warning(
                "BaoStock financial indicator failed for %s",
                symbol, exc_info=True,
            )
            return pd.DataFrame()

    def _fetch_financial(
        self, symbol: str, start: str | None = None, end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch quarterly financial data from BaoStock."""
        self._ensure_login()
        bs_code = self._to_bs_code(symbol)

        from pangu.tz import now as _now
        current_year = _now().year

        if start is not None:
            start_year = int(start[:4])
        else:
            start_year = current_year - 1

        rows: list[dict[str, Any]] = []
        profit_by_quarter: dict[str, dict] = {}
        for year in range(start_year, current_year + 1):
            for quarter in range(1, 5):
                try:
                    rs_p = self._bs.query_profit_data(bs_code, year=year, quarter=quarter)
                    df_p = self._query_to_df(rs_p)
                    if not df_p.empty:
                        r = df_p.iloc[0]
                        profit_by_quarter[f"{year}Q{quarter}"] = {
                            "stat_date": r.get("statDate", ""),
                            "roe": self._safe_float(r.get("roeAvg")),
                            "mb_revenue": self._safe_float(r.get("MBRevenue")),
                        }
                except Exception:  # noqa: BLE001
                    pass

                try:
                    rs_g = self._bs.query_growth_data(bs_code, year=year, quarter=quarter)
                    df_g = self._query_to_df(rs_g)
                    if not df_g.empty:
                        r = df_g.iloc[0]
                        stat_date = r.get("statDate", "")
                        roe = profit_by_quarter.get(f"{year}Q{quarter}", {}).get("roe")
                        mb_revenue = profit_by_quarter.get(f"{year}Q{quarter}", {}).get("mb_revenue")

                        prev_key = f"{year - 1}Q{quarter}"
                        prev_revenue = profit_by_quarter.get(prev_key, {}).get("mb_revenue")
                        revenue_yoy = None
                        if mb_revenue and prev_revenue and prev_revenue != 0:
                            revenue_yoy = round(mb_revenue / prev_revenue - 1, 6)

                        profit_yoy = self._safe_float(r.get("YOYNI"))

                        rows.append({
                            "symbol": symbol,
                            "date": stat_date,
                            "roe_ttm": roe,
                            "revenue_yoy": revenue_yoy,
                            "profit_yoy": profit_yoy,
                        })
                except Exception:  # noqa: BLE001
                    pass

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)

        # Filter by end date if provided
        if end is not None and "date" in result.columns:
            result = result[result["date"] <= end]

        return result
