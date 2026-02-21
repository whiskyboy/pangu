"""AkShare FundamentalDataProvider — PRD §4.1.2."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from pangu.utils import CircuitBreaker, ThrottleMixin, retry_call

logger = logging.getLogger(__name__)


class AkShareFundamentalProvider(ThrottleMixin):
    """Fundamental data backed by AkShare + optional SQLite persistence.

    API mapping:
    - get_valuation   → stock_individual_info_em (market cap)
                      + stock_financial_analysis_indicator (EPS/BPS → PE/PB)
    - get_financial_indicator → stock_financial_analysis_indicator

    Parameters
    ----------
    storage : Database | None
        If provided, fundamentals are cached in SQLite.
    request_interval : float
        Minimum seconds between API calls (default 0.5).
    """

    def __init__(self, storage=None, request_interval: float = 0.5) -> None:
        import akshare  # lazy import for test mocking

        self._ak = akshare
        self._storage = storage
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

    def _fetch_individual_info(self, symbol: str) -> dict[str, Any]:
        """Fetch market cap / basic info via stock_individual_info_em."""
        self._throttle()
        df = retry_call(
            lambda: self._ak.stock_individual_info_em(symbol=symbol),
            circuit=self._circuit,
        )
        if df is None or df.empty:
            return {}
        # DataFrame has columns: item, value
        info: dict[str, Any] = {}
        for _, row in df.iterrows():
            info[str(row["item"])] = row["value"]
        return info

    def _fetch_financial_indicator(self, symbol: str) -> pd.DataFrame:
        """Fetch financial analysis indicators via stock_financial_analysis_indicator."""
        self._throttle()
        import datetime

        from pangu.tz import now as _now
        current_year = str(_now().year - 1)
        df = retry_call(
            lambda: self._ak.stock_financial_analysis_indicator(
                symbol=symbol, start_year=current_year,
            ),
            circuit=self._circuit,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    # -- Protocol methods --

    def get_valuation(self, symbol: str) -> dict[str, Any]:
        """Return PE_TTM, PB, PS, market_cap for a single stock.

        Checks DB cache first — if today's data exists, returns it without
        calling AkShare API.
        """
        # DB cache check
        if self._storage is not None:
            cached = self._load_cached_valuation(symbol)
            if cached is not None:
                return cached

        result: dict[str, Any] = {
            "pe_ttm": None, "pb": None, "ps": None, "market_cap": None,
        }
        try:
            info = self._fetch_individual_info(symbol)
            market_cap = self._safe_float(info.get("总市值"))
            if market_cap is not None:
                result["market_cap"] = market_cap

            fin = self._fetch_financial_indicator(symbol)
            if fin.empty:
                return result

            latest = fin.iloc[-1]

            eps = self._safe_float(latest.get("摊薄每股收益(元)"))
            bps = self._safe_float(latest.get("每股净资产_调整前(元)"))
            total_shares = self._safe_float(info.get("总股本"))

            if market_cap and total_shares:
                if eps and eps != 0:
                    net_profit = eps * total_shares
                    result["pe_ttm"] = round(market_cap / net_profit, 2)

                if bps and bps != 0:
                    net_assets = bps * total_shares
                    result["pb"] = round(market_cap / net_assets, 2)

            revenue = self._safe_float(latest.get("主营业务利润(元)"))
            if market_cap and revenue and revenue != 0:
                result["ps"] = round(market_cap / revenue, 2)

        except Exception:  # noqa: BLE001
            logger.warning("get_valuation(%s) failed", symbol, exc_info=True)

        # Persist to SQLite
        if self._storage is not None:
            self._persist_valuation(symbol, result)

        return result

    def get_financial_indicator(self, symbol: str) -> pd.DataFrame:
        """Return standardized financial indicators DataFrame."""
        try:
            fin = self._fetch_financial_indicator(symbol)
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

            # Persist to SQLite
            if self._storage is not None and not result.empty:
                self._persist_financial(symbol, result)

            return result

        except Exception:  # noqa: BLE001
            logger.warning(
                "get_financial_indicator(%s) failed", symbol, exc_info=True,
            )
            return pd.DataFrame()

    # -- persistence helpers --

    def _load_cached_valuation(self, symbol: str) -> dict[str, Any] | None:
        """Return today's cached valuation from DB, or None if not found."""
        import datetime

        today = datetime.date.today().isoformat()
        try:
            df = self._storage.load_fundamentals(symbol, today, today)
            if df.empty:
                return None
            row = df.iloc[-1]
            return {
                "pe_ttm": row.get("pe_ttm"),
                "pb": row.get("pb"),
                "ps": None,
                "market_cap": row.get("market_cap"),
            }
        except Exception:  # noqa: BLE001
            return None

    def _persist_valuation(self, symbol: str, val: dict[str, Any]) -> None:
        """Save latest valuation snapshot to SQLite fundamentals table."""
        import datetime

        today = datetime.date.today().isoformat()
        df = pd.DataFrame([{
            "symbol": symbol,
            "date": today,
            "pe_ttm": val.get("pe_ttm"),
            "pb": val.get("pb"),
            "roe_ttm": None,
            "revenue_yoy": None,
            "profit_yoy": None,
            "market_cap": val.get("market_cap"),
        }])
        try:
            self._storage.save_fundamentals(symbol, df)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to persist valuation for %s", symbol)

    def _persist_financial(self, symbol: str, result: pd.DataFrame) -> None:
        """Save financial indicators to SQLite fundamentals table."""
        df = pd.DataFrame({
            "symbol": symbol,
            "date": result["date"],
            "pe_ttm": None,
            "pb": None,
            "roe_ttm": result.get("roe_ttm"),
            "revenue_yoy": result.get("revenue_yoy"),
            "profit_yoy": result.get("profit_yoy"),
            "market_cap": None,
        })
        try:
            self._storage.save_fundamentals(symbol, df)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to persist financials for %s", symbol)
