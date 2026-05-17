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
                symbol=symbol,
                start_year=start_year,
            ),
            circuit=self._circuit,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    # -- Protocol methods --

    # AkShare field name → DB column name (% fields are divided by 100)
    _FIELD_MAP_PCT = {
        "净资产收益率(%)": "roe_ttm",
        "主营业务收入增长率(%)": "revenue_yoy",
        "净利润增长率(%)": "profit_yoy",
        "销售净利率(%)": "net_profit_margin",
        "销售毛利率(%)": "gross_margin",
        "资产负债率(%)": "debt_ratio",
        "净资产增长率(%)": "equity_yoy",
        "总资产增长率(%)": "asset_yoy",
        "经营现金净流量与净利润的比率(%)": "cashflow_to_profit",
        # High-value additions
        "总资产利润率(%)": "roa",
        "营业利润率(%)": "operating_profit_ratio",
        "经营现金净流量对销售收入比率(%)": "ocf_to_revenue",
        # Medium-value additions
        "成本费用利润率(%)": "cost_profit_ratio",
        "股息发放率(%)": "dividend_payout_ratio",
        "现金比率(%)": "cash_ratio",
        "产权比率(%)": "equity_ratio",
        "股东权益比率(%)": "shareholder_equity_ratio",
    }
    _FIELD_MAP_RAW = {
        "总资产周转率(次)": "asset_turnover",
        "流动比率": "current_ratio",
        "每股经营性现金流(元)": "cashflow_per_share",
        # High-value additions
        "加权每股收益(元)": "eps_weighted",
        "速动比率": "quick_ratio",
        "应收账款周转率(次)": "receivables_turnover",
        "存货周转率(次)": "inventory_turnover",
        # Medium-value additions
        "每股未分配利润(元)": "undistributed_per_share",
        "每股资本公积金(元)": "capital_reserve_per_share",
    }

    def fetch_pub_dates_batch(self, quarter_date: str) -> dict[str, str]:
        """Fetch first-announcement dates for all stocks via cninfo disclosure schedule.

        Uses ``ak.stock_report_disclosure`` which returns the official
        预约披露 data from 巨潮资讯 (cninfo.com.cn).  The ``实际披露``
        field is the first disclosure date (not revision date).

        Parameters
        ----------
        quarter_date : str
            Quarter end date in ``YYYYMMDD`` format (e.g. ``"20240331"``).

        Returns
        -------
        dict[str, str]
            Mapping of ``symbol → pub_date`` (``YYYY-MM-DD``).
        """
        from pangu.utils import quarter_to_cninfo_period

        period = quarter_to_cninfo_period(quarter_date)
        self._throttle()
        df = retry_call(
            lambda: self._ak.stock_report_disclosure(market="沪深京", period=period),
            circuit=self._circuit,
        )
        if df is None or df.empty:
            return {}

        result: dict[str, str] = {}
        for _, row in df.iterrows():
            code = str(row.get("股票代码", "")).strip()
            pub = row.get("实际披露")
            if not code or pub is None or (hasattr(pub, "__class__") and str(pub) == "NaT"):
                continue
            pub_str = pub.strftime("%Y-%m-%d") if hasattr(pub, "strftime") else str(pub)[:10]
            result[code] = pub_str
        return result

    def fetch_gross_margin_batch(self, quarter_date: str) -> dict[str, float]:
        """Fetch gross margin for all stocks for a given quarter via ``stock_yjbb_em``.

        Parameters
        ----------
        quarter_date : str
            Quarter end date in ``YYYYMMDD`` format (e.g. ``"20240331"``).

        Returns
        -------
        dict[str, float]
            Mapping of symbol → gross_margin (as decimal ratio, already /100).
        """
        self._throttle()
        df = retry_call(
            lambda: self._ak.stock_yjbb_em(date=quarter_date),
            circuit=self._circuit,
        )
        if df is None or df.empty:
            return {}
        result: dict[str, float] = {}
        for _, row in df.iterrows():
            sym = str(row.get("股票代码", ""))
            val = self._safe_float(row.get("销售毛利率"))
            if sym and val is not None:
                result[sym] = val / 100
        return result

    def get_financial_indicator(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return standardized financial indicators DataFrame.

        Also stores the raw API response in ``_last_raw_response`` for
        ``CompositeFundamentalProvider`` to persist to ``fundamentals_raw``.
        """
        self._last_raw_response = pd.DataFrame()
        try:
            start_year = start[:4] if start else None
            fin = self._fetch_financial_indicator(symbol, start_year=start_year)
            if fin.empty:
                return pd.DataFrame()

            self._last_raw_response = fin

            rows = []
            for _, raw in fin.iterrows():
                row: dict[str, Any] = {
                    "symbol": symbol,
                    "date": str(raw.get("日期", "")),
                }
                for cn_name, db_col in self._FIELD_MAP_PCT.items():
                    val = self._safe_float(raw.get(cn_name))
                    row[db_col] = val / 100 if val is not None else None
                for cn_name, db_col in self._FIELD_MAP_RAW.items():
                    row[db_col] = self._safe_float(raw.get(cn_name))

                rows.append(row)

            result = pd.DataFrame(rows)

            if end is not None and "date" in result.columns:
                result = result[result["date"] <= end]

            return result

        except Exception:  # noqa: BLE001
            logger.warning(
                "get_financial_indicator(%s) failed",
                symbol,
                exc_info=True,
            )
            return pd.DataFrame()
