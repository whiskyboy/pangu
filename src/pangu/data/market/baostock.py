"""BaoStock MarketDataProvider — PRD §4.1.1 (M2.4)."""

from __future__ import annotations

import logging
import threading

import pandas as pd

from pangu.utils import CircuitBreaker, retry_call

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
        self._circuit = CircuitBreaker()

    # -- session management --

    def _ensure_login(self) -> None:
        with self._login_lock:
            if not self._logged_in:
                lg = self._bs.login()
                if lg.error_code != "0":
                    raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
                self._logged_in = True

    def _relogin(self) -> None:
        """Force a fresh login (e.g. after session expiry)."""
        with self._login_lock:
            try:
                self._bs.logout()
            except Exception:  # noqa: BLE001
                pass
            self._logged_in = False
        self._ensure_login()

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

    def _query_with_retry(self, query_fn):
        """Execute a BaoStock query with exponential backoff and session recovery."""
        def _checked():
            self._ensure_login()
            rs = query_fn()
            if rs.error_code != "0":
                if "未登录" in getattr(rs, "error_msg", ""):
                    self._relogin()
                raise RuntimeError(getattr(rs, "error_msg", "unknown error"))
            return rs
        return retry_call(_checked, circuit=self._circuit)

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily K-line via bs.query_history_k_data_plus.

        Returns OHLCV DataFrame with peTTM/pbMRQ columns (for Composite
        to extract and persist valuation data).
        """
        bs_code = _to_bs_code(symbol)
        rs = self._query_with_retry(
            lambda: self._bs.query_history_k_data_plus(
                bs_code, self._DAILY_FIELDS,
                start_date=start, end_date=end,
                frequency="d", adjustflag="2",
            )
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
        bs_code = _to_bs_index_code(symbol)
        rs = self._query_with_retry(
            lambda: self._bs.query_history_k_data_plus(
                bs_code, "date,code,open,high,low,close,volume,amount",
                start_date=start, end_date=end,
                frequency="d", adjustflag="3",
            )
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
