"""MarketDataProvider Protocol — PRD §4.1.1 / §6."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketDataProvider(Protocol):
    """Unified interface for A-share and international market data."""

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return A-share daily OHLCV bars with adjustment factor."""
        ...

    def get_realtime_quote(self, symbols: list[str]) -> pd.DataFrame:
        """Return real-time quote snapshot for given symbols."""
        ...

    def get_stock_list(self) -> pd.DataFrame:
        """Return full A-share stock list (code, name, industry, listing_date)."""
        ...

    def get_us_indices(self) -> pd.DataFrame:
        """Return latest US major indices (S&P 500, DJIA, NASDAQ)."""
        ...

    def get_hk_indices(self) -> pd.DataFrame:
        """Return latest Hang Seng Index data."""
        ...

    def get_commodity_futures(self) -> pd.DataFrame:
        """Return international commodity futures (gold, silver, crude oil, copper, iron ore)."""
        ...

    def get_global_snapshot(self) -> pd.DataFrame:
        """Aggregate all international quotes into a single snapshot."""
        ...


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------

_STOCKS = {
    "600519": ("贵州茅台", "白酒"),
    "000858": ("五粮液", "白酒"),
    "300750": ("宁德时代", "新能源"),
    "601318": ("中国平安", "保险"),
    "000001": ("平安银行", "银行"),
}

_BASE_PRICES: dict[str, float] = {
    "600519": 1800.0,
    "000858": 150.0,
    "300750": 220.0,
    "601318": 48.0,
    "000001": 12.0,
}


class FakeMarketDataProvider:
    """Deterministic fake data for testing."""

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(hash(symbol) % 2**31)
        base = _BASE_PRICES.get(symbol, 100.0)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        if n == 0:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
            )
        close = base * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        open_ = close * rng.uniform(0.98, 1.02, n)
        high = np.maximum(open_, close) * rng.uniform(1.0, 1.03, n)
        low = np.minimum(open_, close) * rng.uniform(0.97, 1.0, n)
        volume = rng.integers(1_000_000, 10_000_000, n)
        return pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "adj_factor": 1.0,
        })

    def get_realtime_quote(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        for s in symbols:
            name, _ = _STOCKS.get(s, (s, "未知"))
            base = _BASE_PRICES.get(s, 100.0)
            rows.append({
                "symbol": s,
                "name": name,
                "price": base,
                "change_pct": 0.5,
                "volume": 5_000_000,
                "amount": base * 5_000_000,
                "volume_ratio": 1.2,
            })
        return pd.DataFrame(rows)

    def get_stock_list(self) -> pd.DataFrame:
        rows = [
            {"symbol": sym, "name": name, "industry": ind, "listing_date": "2000-01-01"}
            for sym, (name, ind) in _STOCKS.items()
        ]
        return pd.DataFrame(rows)

    def get_us_indices(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "SPX", "name": "S&P 500", "date": "2026-01-02", "open": 5180.0,
             "high": 5220.0, "low": 5170.0, "close": 5200.0, "volume": 3e9,
             "change_pct": 0.3, "source": "us_index"},
            {"symbol": "DJI", "name": "Dow Jones", "date": "2026-01-02", "open": 38900.0,
             "high": 39100.0, "low": 38800.0, "close": 39000.0, "volume": 2e9,
             "change_pct": 0.2, "source": "us_index"},
            {"symbol": "IXIC", "name": "NASDAQ", "date": "2026-01-02", "open": 16400.0,
             "high": 16600.0, "low": 16350.0, "close": 16500.0, "volume": 4e9,
             "change_pct": 0.5, "source": "us_index"},
        ])

    def get_hk_indices(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "HSI", "name": "恒生指数", "date": "2026-01-02", "open": 17600.0,
             "high": 17700.0, "low": 17400.0, "close": 17500.0, "volume": 1e9,
             "change_pct": -0.3, "source": "hk_index"},
        ])

    def get_commodity_futures(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"symbol": "GC", "name": "COMEX黄金", "date": "2026-01-02", "open": 2340.0,
             "high": 2360.0, "low": 2330.0, "close": 2350.0, "volume": 1e6,
             "change_pct": 0.1, "source": "commodity"},
            {"symbol": "SI", "name": "COMEX白银", "date": "2026-01-02", "open": 28.3,
             "high": 28.8, "low": 28.1, "close": 28.5, "volume": 5e5,
             "change_pct": -0.2, "source": "commodity"},
            {"symbol": "CL", "name": "WTI原油", "date": "2026-01-02", "open": 77.5,
             "high": 78.5, "low": 77.0, "close": 78.0, "volume": 8e5,
             "change_pct": 0.8, "source": "commodity"},
            {"symbol": "HG", "name": "LME铜", "date": "2026-01-02", "open": 4.15,
             "high": 4.25, "low": 4.1, "close": 4.2, "volume": 3e5,
             "change_pct": 0.4, "source": "commodity"},
            {"symbol": "FEF", "name": "铁矿石", "date": "2026-01-02", "open": 830.0,
             "high": 835.0, "low": 815.0, "close": 820.0, "volume": 2e5,
             "change_pct": -1.0, "source": "commodity"},
        ])

    def get_global_snapshot(self) -> pd.DataFrame:
        return pd.concat(
            [self.get_us_indices(), self.get_hk_indices(), self.get_commodity_futures()],
            ignore_index=True,
        )


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


# ---------------------------------------------------------------------------
# Retry / circuit-breaker infrastructure — PRD §4.1.1
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Simple consecutive-failure circuit breaker (thread-safe)."""

    def __init__(self, threshold: int = 5, cooldown: float = 300.0) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._consecutive_failures = 0
        self._open_until: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._consecutive_failures >= self._threshold:
                if time.monotonic() < self._open_until:
                    return True
                # Cooldown expired — reset so next failure doesn't re-open immediately
                self._consecutive_failures = 0
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._open_until = time.monotonic() + self._cooldown
                logger.warning(
                    "Circuit breaker OPEN — %d consecutive failures, cooling down %.0fs",
                    self._consecutive_failures,
                    self._cooldown,
                )


def _retry_call(fn, *, max_retries: int = 3, backoff_base: float = 2.0,
                circuit: CircuitBreaker | None = None):
    """Call *fn()* with exponential back-off and optional circuit breaker.

    Returns the result of *fn()* on success, raises the last exception on
    exhaustion.
    """
    if circuit and circuit.is_open:
        raise RuntimeError("Circuit breaker is open — skipping call")

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = fn()
            if circuit:
                circuit.record_success()
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if circuit:
                circuit.record_failure()
            if attempt < max_retries - 1:
                wait = backoff_base ** (attempt + 1)
                logger.warning("Retry %d/%d after %.1fs: %s", attempt + 1, max_retries, wait, exc)
                time.sleep(wait)
            else:
                logger.warning("Retry %d/%d exhausted: %s", attempt + 1, max_retries, exc)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AkShare real implementation — PRD §4.1.1
# ---------------------------------------------------------------------------


class AkShareMarketDataProvider:
    """Real A-share market data backed by AkShare + SQLite cache.

    Parameters
    ----------
    storage : Database | None
        If provided, daily bars are cached in SQLite with incremental updates.
    request_interval : float
        Minimum seconds between AkShare API calls (default 0.5).
    """

    def __init__(
        self,
        storage=None,
        request_interval: float = 0.5,
    ) -> None:
        import akshare  # noqa: F811  — lazy import so tests can mock

        self._ak = akshare
        self._storage = storage
        self._interval = request_interval
        self._last_call: float = 0.0
        self._circuit = CircuitBreaker()
        self._throttle_lock = threading.Lock()

    # -- rate limiting --

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.monotonic()

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
        """Fetch daily bars with SQLite cache + incremental update."""
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

        self._throttle()
        df = _retry_call(
            lambda: self._ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=ak_start, end_date=ak_end, adjust="qfq",
            ),
            circuit=self._circuit,
        )

        if df is None or df.empty:
            if self._storage is not None:
                return self._storage.load_daily_bars(symbol, start, end)
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
            )

        df = self._clean_hist(df)

        # Persist to SQLite
        if self._storage is not None:
            self._storage.save_daily_bars(symbol, df)
            max_date = df["date"].max()
            self._storage.update_sync_log(
                symbol, "daily_bars", "ok", "akshare", last_date=max_date,
            )
            return self._storage.load_daily_bars(symbol, start, end)

        # Filter to requested range if no storage
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    # -- bid/ask row mapping --

    _BID_ASK_MAP = {
        "最新": "price",
        "涨幅": "change_pct",
        "总手": "volume",
        "金额": "amount",
        "最高": "high",
        "最低": "low",
        "今开": "open",
        "量比": "volume_ratio",
    }

    def _fetch_single_quote(self, symbol: str) -> dict:
        """Fetch a single stock quote via stock_bid_ask_em (fast, ~0.3s)."""
        self._throttle()
        df = _retry_call(
            lambda: self._ak.stock_bid_ask_em(symbol=symbol),
            circuit=self._circuit,
        )
        if df is None or df.empty:
            logger.warning("stock_bid_ask_em returned empty for %s", symbol)
            return {"symbol": symbol}
        # df has columns: item, value — pivot to dict
        row = dict(zip(df["item"], df["value"]))
        result = {"symbol": symbol}
        for cn, en in self._BID_ASK_MAP.items():
            result[en] = row.get(cn)
        return result

    def get_realtime_quote(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch real-time quotes per symbol via stock_bid_ask_em (fast)."""
        rows = [self._fetch_single_quote(s) for s in symbols]
        return pd.DataFrame(rows)

    def get_stock_list(self) -> pd.DataFrame:
        """Fetch full A-share stock list from exchange info APIs (~6s).

        Combines SH (main + STAR), SZ (main + ChiNext), and BJ exchanges.
        """
        frames: list[pd.DataFrame] = []

        def _try_append(label: str, fn, col_map: dict[str, str]) -> None:
            try:
                self._throttle()
                df = _retry_call(fn, circuit=self._circuit)
                if df is not None and not df.empty:
                    frames.append(df.rename(columns=col_map)[["symbol", "name"]])
                else:
                    logger.warning("get_stock_list: %s returned empty", label)
            except Exception:  # noqa: BLE001
                logger.warning("get_stock_list: %s failed, skipping", label, exc_info=True)

        _try_append("SH主板",
                     lambda: self._ak.stock_info_sh_name_code(symbol="主板A股"),
                     {"证券代码": "symbol", "证券简称": "name"})
        _try_append("SH科创板",
                     lambda: self._ak.stock_info_sh_name_code(symbol="科创板"),
                     {"证券代码": "symbol", "证券简称": "name"})
        _try_append("SZ",
                     lambda: self._ak.stock_info_sz_name_code(symbol="A股列表"),
                     {"A股代码": "symbol", "A股简称": "name"})
        _try_append("BJ",
                     lambda: self._ak.stock_info_bj_name_code(),
                     {"证券代码": "symbol", "证券简称": "name"})

        if not frames:
            logger.error("get_stock_list: all exchange APIs failed")
            return pd.DataFrame(columns=["symbol", "name"])
        return pd.concat(frames, ignore_index=True)

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
    }

    def _fetch_us_index(self, symbol: str, ak_sym: str, name: str) -> dict | None:
        """Fetch latest US index bar via index_us_stock_sina."""
        try:
            self._throttle()
            df = _retry_call(
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
            change_pct = (close - prev_close) / prev_close * 100 if prev_close else None
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

    def get_us_indices(self) -> pd.DataFrame:
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

    def get_hk_indices(self) -> pd.DataFrame:
        """Fetch latest Hang Seng Index data via stock_hk_index_spot_sina (~1s)."""
        try:
            self._throttle()
            full = _retry_call(
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

        today = pd.Timestamp.now(tz="Asia/Hong_Kong").strftime("%Y-%m-%d")
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
    _COMMODITY_SYMBOLS = "GC,SI,CL,HG,FEF"
    _COMMODITY_NAME_MAP: dict[str, tuple[str, str]] = {
        "COMEX黄金": ("GC", "COMEX黄金"),
        "COMEX白银": ("SI", "COMEX白银"),
        "NYMEX原油": ("CL", "WTI原油"),
        "COMEX铜": ("HG", "LME铜"),
        "新加坡铁矿石": ("FEF", "铁矿石"),
    }

    def get_commodity_futures(self) -> pd.DataFrame:
        """Fetch international commodity futures via futures_foreign_commodity_realtime (~1s)."""
        try:
            self._throttle()
            raw = _retry_call(
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
                "date": str(r.get("日期", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))),
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
        """Aggregate all international quotes into a single snapshot."""
        frames = [self.get_us_indices(), self.get_hk_indices(), self.get_commodity_futures()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(
                columns=["symbol", "name", "date", "open", "high", "low",
                          "close", "volume", "change_pct", "source"]
            )
        return pd.concat(frames, ignore_index=True)

