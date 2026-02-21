"""AkShare MarketDataProvider — PRD §4.1.1."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from trading_agent.data.market.baostock import BaoStockMarketDataProvider
from trading_agent.utils import CircuitBreaker, ThrottleMixin, retry_call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AkShare column mappings (Chinese → English)
# ---------------------------------------------------------------------------

_HIST_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


class AkShareMarketDataProvider(ThrottleMixin):
    """Real A-share market data backed by AkShare + SQLite cache.

    Parameters
    ----------
    storage : Database | None
        If provided, daily bars are cached in SQLite with incremental updates.
    request_interval : float
        Minimum seconds between AkShare API calls (default 0.5).
    fallback : BaoStockMarketDataProvider | None
        If provided, daily bars automatically fall back to BaoStock when
        AkShare fails (circuit breaker open or retries exhausted).
    """

    def __init__(
        self,
        storage=None,
        request_interval: float = 0.5,
        fallback: BaoStockMarketDataProvider | None = None,
    ) -> None:
        import akshare  # noqa: F811  — lazy import so tests can mock

        self._ak = akshare
        self._storage = storage
        self.__init_throttle__(request_interval)
        self._circuit = CircuitBreaker()
        self._fallback = fallback

    # -- helpers --

    @staticmethod
    def _clean_hist(df: pd.DataFrame) -> pd.DataFrame:
        """Rename Chinese columns, keep only OHLCV+amount, add adj_factor."""
        df = df.rename(columns=_HIST_COL_MAP)
        required = ("date", "open", "high", "low", "close", "volume", "amount")
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"AkShare daily bars missing critical columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df[["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]]

    # -- Protocol methods --

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily bars with SQLite cache + incremental update + BaoStock fallback."""
        # Try cache first
        if self._storage is not None:
            last = self._storage.get_last_sync_date(symbol, "daily_bars")
            if last is not None and last >= end:
                return self._storage.load_daily_bars(symbol, start, end)
            # Incremental: fetch from (last+1) to end
            if last:
                next_day = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                fetch_start = next_day
            else:
                fetch_start = start
        else:
            fetch_start = start

        # AkShare uses date format YYYYMMDD
        ak_start = fetch_start.replace("-", "")
        ak_end = end.replace("-", "")

        try:
            self._throttle()
            df = retry_call(
                lambda: self._ak.stock_zh_a_hist(
                    symbol=symbol, period="daily",
                    start_date=ak_start, end_date=ak_end, adjust="qfq",
                ),
                circuit=self._circuit,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "AkShare get_daily_bars failed for %s, trying fallback", symbol, exc_info=True,
            )
            df = None

        # Fallback to BaoStock only on AkShare failure (not empty results)
        if df is None and self._fallback is not None:
            logger.info("Falling back to BaoStock for %s daily bars", symbol)
            try:
                fallback_df = self._fallback.get_daily_bars(symbol, fetch_start, end)
                if fallback_df is not None and not fallback_df.empty:
                    df = fallback_df
            except Exception:  # noqa: BLE001
                logger.warning("BaoStock fallback also failed for %s", symbol, exc_info=True)

        # No storage — return whatever we got
        if self._storage is None:
            if df is None or df.empty:
                return pd.DataFrame(
                    columns=["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
                )
            df = self._clean_hist(df)
            return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

        # Persist new data if any
        if df is not None and not df.empty:
            df = self._clean_hist(df)
            self._storage.save_daily_bars(symbol, df)

        sync_date = df["date"].max() if (df is not None and not df.empty) else end
        self._storage.update_sync_log(
            symbol, "daily_bars", "ok", "akshare", last_date=sync_date,
        )
        return self._storage.load_daily_bars(symbol, start, end)

    # -- international market data — PRD §4.1.1 --

    # US index symbol → (AkShare param, display name)
    _US_INDEX_MAP: dict[str, tuple[str, str]] = {
        "SPX": (".INX", "S&P 500"),
        "DJI": (".DJI", "Dow Jones"),
        "IXIC": (".IXIC", "NASDAQ"),
    }

    # HK index code → display name (from stock_hk_index_spot_em)
    _HK_INDEX_CODES: dict[str, str] = {
        "HSI": "恒生指数",
        "HSTECH": "恒生科技指数",
        "VHSI": "恒指波幅指数",
    }

    def _fetch_us_index(self, symbol: str, ak_sym: str, name: str) -> dict | None:
        """Fetch latest US index bar via index_us_stock_sina."""
        try:
            self._throttle()
            df = retry_call(
                lambda: self._ak.index_us_stock_sina(symbol=ak_sym),
                circuit=self._circuit,
            )
            if df is None or len(df) < 2:
                logger.warning("index_us_stock_sina(%s) returned insufficient data", ak_sym)
                return None
            last = df.iloc[-1]
            prev = df.iloc[-2]
            prev_close = float(prev["close"])
            close = float(last["close"])
            change_pct = (
                round((close - prev_close) / prev_close * 100, 4)
                if pd.notna(prev_close) and prev_close != 0
                else None
            )
            return {
                "symbol": symbol,
                "name": name,
                "date": str(last["date"])[:10],
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": close,
                "volume": float(last["volume"]) if last.get("volume") else None,
                "change_pct": round(change_pct, 4) if change_pct is not None else None,
                "source": "us_index",
            }
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch US index %s", symbol, exc_info=True)
            return None

    def _get_us_indices(self) -> pd.DataFrame:
        """Fetch latest US major indices (S&P 500, DJIA, NASDAQ)."""
        rows: list[dict] = []
        for symbol, (ak_sym, name) in self._US_INDEX_MAP.items():
            row = self._fetch_us_index(symbol, ak_sym, name)
            if row:
                rows.append(row)
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "name", "date", "open", "high", "low",
                      "close", "volume", "change_pct", "source"]
        )
        if self._storage is not None and not df.empty:
            self._storage.save_global_snapshots(df)
        return df

    def _get_hk_indices(self) -> pd.DataFrame:
        """Fetch latest Hang Seng Index data via stock_hk_index_spot_sina (~1s)."""
        try:
            self._throttle()
            full = retry_call(
                lambda: self._ak.stock_hk_index_spot_sina(),
                circuit=self._circuit,
            )
            if full is None or full.empty:
                logger.warning("stock_hk_index_spot_sina returned empty")
                return pd.DataFrame(
                    columns=["symbol", "name", "date", "open", "high", "low",
                              "close", "volume", "change_pct", "source"]
                )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch HK indices", exc_info=True)
            return pd.DataFrame(
                columns=["symbol", "name", "date", "open", "high", "low",
                          "close", "volume", "change_pct", "source"]
            )

        from trading_agent.tz import today_str

        today = today_str()
        rows: list[dict] = []
        for code, name in self._HK_INDEX_CODES.items():
            match = full[full["代码"] == code]
            if match.empty:
                logger.warning("HK index %s not found in response", code)
                continue
            r = match.iloc[0]
            rows.append({
                "symbol": code,
                "name": name,
                "date": today,
                "open": float(r["今开"]) if pd.notna(r["今开"]) else None,
                "high": float(r["最高"]) if pd.notna(r["最高"]) else None,
                "low": float(r["最低"]) if pd.notna(r["最低"]) else None,
                "close": float(r["最新价"]) if pd.notna(r["最新价"]) else None,
                "volume": None,  # sina API doesn't provide volume
                "change_pct": float(r["涨跌幅"]) if pd.notna(r["涨跌幅"]) else None,
                "source": "hk_index",
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "name", "date", "open", "high", "low",
                      "close", "volume", "change_pct", "source"]
        )
        if self._storage is not None and not df.empty:
            self._storage.save_global_snapshots(df)
        return df

    # Commodity symbols for futures_foreign_commodity_realtime batch call
    _COMMODITY_SYMBOLS = "GC,SI,CL,HG,FEF,NG,CT"
    _COMMODITY_NAME_MAP: dict[str, tuple[str, str]] = {
        "COMEX黄金": ("GC", "COMEX黄金"),
        "COMEX白银": ("SI", "COMEX白银"),
        "NYMEX原油": ("CL", "WTI原油"),
        "COMEX铜": ("HG", "LME铜"),
        "新加坡铁矿石": ("FEF", "铁矿石"),
        "NYMEX天然气": ("NG", "NYMEX天然气"),
        "NYBOT-棉花": ("CT", "NYBOT棉花"),
    }

    def _get_commodity_futures(self) -> pd.DataFrame:
        """Fetch international commodity futures via futures_foreign_commodity_realtime (~1s)."""
        from trading_agent.tz import today_str

        try:
            self._throttle()
            raw = retry_call(
                lambda: self._ak.futures_foreign_commodity_realtime(
                    symbol=self._COMMODITY_SYMBOLS,
                ),
                circuit=self._circuit,
            )
            if raw is None or raw.empty:
                logger.warning("futures_foreign_commodity_realtime returned empty")
                return pd.DataFrame(
                    columns=["symbol", "name", "date", "open", "high", "low",
                              "close", "volume", "change_pct", "source"]
                )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to fetch commodity futures", exc_info=True)
            return pd.DataFrame(
                columns=["symbol", "name", "date", "open", "high", "low",
                          "close", "volume", "change_pct", "source"]
            )

        rows: list[dict] = []
        for _, r in raw.iterrows():
            cn_name = str(r["名称"])
            if cn_name not in self._COMMODITY_NAME_MAP:
                continue
            sym, display_name = self._COMMODITY_NAME_MAP[cn_name]
            rows.append({
                "symbol": sym,
                "name": display_name,
                "date": str(r.get("日期", today_str())),
                "open": float(r["开盘价"]) if pd.notna(r.get("开盘价")) else None,
                "high": float(r["最高价"]) if pd.notna(r.get("最高价")) else None,
                "low": float(r["最低价"]) if pd.notna(r.get("最低价")) else None,
                "close": float(r["最新价"]) if pd.notna(r.get("最新价")) else None,
                "volume": None,  # realtime API doesn't provide volume
                "change_pct": float(r["涨跌幅"]) if pd.notna(r.get("涨跌幅")) else None,
                "source": "commodity",
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["symbol", "name", "date", "open", "high", "low",
                      "close", "volume", "change_pct", "source"]
        )
        if self._storage is not None and not df.empty:
            self._storage.save_global_snapshots(df)
        return df

    def get_global_snapshot(self) -> pd.DataFrame:
        """Aggregate all international quotes, with SQLite cache for today."""
        # Check DB cache: if we already have snapshots saved today, return them
        if self._storage is not None:
            from trading_agent.tz import today_str
            today = today_str()
            cached = self._storage.load_latest_global_snapshots()
            if not cached.empty and (cached["date"] >= today).any():
                return cached

        frames = [self._get_us_indices(), self._get_hk_indices(), self._get_commodity_futures()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(
                columns=["symbol", "name", "date", "open", "high", "low",
                          "close", "volume", "change_pct", "source"]
            )
        return pd.concat(frames, ignore_index=True)
