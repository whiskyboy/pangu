"""BaoStock MarketDataProvider — PRD §4.1.1 (M2.4)."""

from __future__ import annotations

import logging
import threading

import pandas as pd

logger = logging.getLogger(__name__)


def _to_bs_code(symbol: str) -> str:
    """Convert 6-digit A-share code to BaoStock format (e.g. '600519' → 'sh.600519')."""
    if symbol.startswith(("sh.", "sz.")):
        return symbol
    if not symbol or len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"Invalid A-share stock code: {symbol!r}")
    first = symbol[0]
    if first == "6":
        return f"sh.{symbol}"
    if first in ("0", "3"):
        return f"sz.{symbol}"
    # BaoStock doesn't support BJ exchange (8xxxxx) or B-shares (9/2xxxxx)
    raise ValueError(f"BaoStock does not support this stock code: {symbol}")


def _to_bs_index_code(symbol: str) -> str:
    """Convert index code to BaoStock format (e.g. '000300' → 'sh.000300')."""
    if symbol.startswith(("sh.", "sz.")):
        return symbol
    if symbol.startswith("399"):
        return f"sz.{symbol}"
    return f"sh.{symbol}"


class BaoStockMarketDataProvider:
    """Pure BaoStock API provider for A-share daily bars.

    Supports ``get_daily_bars`` (with PE/PB columns) and
    ``get_index_daily_bars``.  Does **not** support international market
    data (``get_global_snapshot`` raises ``NotImplementedError``).

    This provider is **stateless** — it performs API calls only and
    returns DataFrames.  Caching and persistence are handled by
    :class:`CompositeMarketDataProvider`.

    BaoStock requires a login/logout session.  This class manages the
    session lazily (login on first call) and exposes ``close()`` for
    explicit cleanup.
    """

    def __init__(self) -> None:
        import baostock as bs  # lazy import

        self._bs = bs
        self._logged_in = False
        self._login_lock = threading.Lock()

    # -- session management --

    def _ensure_login(self) -> None:
        with self._login_lock:
            if not self._logged_in:
                lg = self._bs.login()
                if lg.error_code != "0":
                    raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
                self._logged_in = True

    def close(self) -> None:
        """Logout from BaoStock session."""
        with self._login_lock:
            if self._logged_in:
                self._bs.logout()
                self._logged_in = False

    def __del__(self) -> None:
        try:
            self.close()
        except AttributeError:
            pass

    # -- helpers --

    def _query_to_df(self, rs) -> pd.DataFrame:
        """Convert BaoStock ResultData to DataFrame."""
        rows: list[list] = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()

    # -- Protocol methods --

    _DAILY_FIELDS = (
        "date,code,open,high,low,close,preclose,volume,amount,"
        "adjustflag,pctChg,peTTM,pbMRQ"
    )

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily K-line via bs.query_history_k_data_plus.

        Returns OHLCV DataFrame with peTTM/pbMRQ columns (for Composite
        to extract and persist valuation data).
        """
        self._ensure_login()

        bs_code = _to_bs_code(symbol)
        rs = self._bs.query_history_k_data_plus(
            bs_code,
            self._DAILY_FIELDS,
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",  # 前复权 = qfq
        )
        if rs.error_code != "0":
            raise RuntimeError(
                f"BaoStock query_history_k_data_plus failed for {bs_code}: {rs.error_msg}"
            )

        df = self._query_to_df(rs)
        if df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
            )

        # Cast numeric columns (BaoStock returns strings)
        for col in ("open", "high", "low", "close", "volume", "amount", "peTTM", "pbMRQ"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["adj_factor"] = 1.0
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        return df

    def get_index_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch index daily K-line (e.g. CSI300 '000300')."""
        self._ensure_login()

        bs_code = _to_bs_index_code(symbol)
        rs = self._bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",  # 指数无复权
        )
        if rs.error_code != "0":
            raise RuntimeError(
                f"BaoStock index query failed for {bs_code}: {rs.error_msg}"
            )

        df = self._query_to_df(rs)
        if df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
            )

        for col in ("open", "high", "low", "close", "volume", "amount"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["adj_factor"] = 1.0
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df[["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]]

    def get_global_snapshot(self) -> pd.DataFrame:
        """Not supported by BaoStock."""
        raise NotImplementedError("BaoStock does not support global market data")
