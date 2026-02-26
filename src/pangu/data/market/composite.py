"""Composite MarketDataProvider — orchestrates provider chain + caching."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "adj_factor"]


class CompositeMarketDataProvider:
    """Orchestrates a priority-ordered list of market data providers.

    Handles:
    - Provider chain fallback (try each provider in order)
    - SQLite caching with incremental updates
    - PE/PB extraction from daily bars (written to fundamentals table)

    Parameters
    ----------
    storage : Database
        SQLite database for caching daily bars and sync log.
    providers : list
        Market data providers ordered by priority (first = highest).
    """

    def __init__(self, storage, providers: list) -> None:
        self._storage = storage
        self._providers = providers

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily bars with cache + provider chain fallback."""
        # 1. Cache check
        last = self._storage.get_last_sync_date(symbol, "daily_bars")
        if last is not None and last >= end:
            return self._storage.load_daily_bars(symbol, start, end)

        # 2. Incremental fetch
        if last:
            next_day = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            fetch_start = next_day
        else:
            fetch_start = start

        # 3. Try providers in priority order
        df, source = None, "unknown"
        for p in self._providers:
            try:
                df = p.get_daily_bars(symbol, fetch_start, end)
                if df is not None and not df.empty:
                    source = type(p).__name__
                    break
            except NotImplementedError:
                continue
            except Exception:  # noqa: BLE001
                logger.warning(
                    "%s.get_daily_bars failed for %s, trying next provider",
                    type(p).__name__, symbol, exc_info=True,
                )
                df = None
                continue

        # 4. No data from any provider
        if df is None or df.empty:
            return self._storage.load_daily_bars(symbol, start, end)

        # 5. Extract PE/PB before trimming columns
        self._persist_valuation(symbol, df)

        # 6. Trim to standard columns + persist
        result = df[[c for c in _STD_COLS if c in df.columns]].copy()
        if not result.empty:
            self._storage.save_daily_bars(symbol, result)
            self._storage.update_sync_log(
                symbol, "daily_bars", "ok", source,
                last_date=result["date"].max(),
            )

        return self._storage.load_daily_bars(symbol, start, end)

    def get_index_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch index daily bars with cache + provider chain fallback."""
        # Cache check
        last = self._storage.get_last_sync_date(symbol, "daily_bars")
        if last is not None and last >= end:
            return self._storage.load_daily_bars(symbol, start, end)

        if last:
            next_day = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            fetch_start = next_day
        else:
            fetch_start = start

        # Try providers in priority order
        df, source = None, "unknown"
        for p in self._providers:
            try:
                df = p.get_index_daily_bars(symbol, fetch_start, end)
                if df is not None and not df.empty:
                    source = type(p).__name__
                    break
            except NotImplementedError:
                continue
            except Exception:  # noqa: BLE001
                logger.warning(
                    "%s.get_index_daily_bars failed for %s, trying next provider",
                    type(p).__name__, symbol, exc_info=True,
                )
                df = None
                continue

        if df is None or df.empty:
            return self._storage.load_daily_bars(symbol, start, end)

        if not df.empty:
            self._storage.save_daily_bars(symbol, df)
            self._storage.update_sync_log(
                symbol, "daily_bars", "ok", source,
                last_date=df["date"].max(),
            )

        return self._storage.load_daily_bars(symbol, start, end)

    def get_global_snapshot(self) -> pd.DataFrame:
        """Fetch global market snapshot via provider chain."""
        for p in self._providers:
            try:
                result = p.get_global_snapshot()
                if result is not None and not result.empty:
                    return result
            except NotImplementedError:
                continue
            except Exception:  # noqa: BLE001
                logger.warning(
                    "%s.get_global_snapshot failed, trying next provider",
                    type(p).__name__, exc_info=True,
                )
                continue
        return pd.DataFrame(
            columns=["symbol", "name", "date", "open", "high", "low",
                      "close", "volume", "change_pct", "source"]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist_valuation(self, symbol: str, df: pd.DataFrame) -> None:
        """Extract peTTM/pbMRQ from daily bars and save to fundamentals table."""
        pe_col = df.get("peTTM")
        pb_col = df.get("pbMRQ")
        if pe_col is None and pb_col is None:
            return

        rows = []
        for _, row in df.iterrows():
            pe = pd.to_numeric(row.get("peTTM", ""), errors="coerce")
            pb = pd.to_numeric(row.get("pbMRQ", ""), errors="coerce")
            if pd.isna(pe) and pd.isna(pb):
                continue
            rows.append({
                "symbol": symbol,
                "date": row["date"],
                "pe_ttm": None if pd.isna(pe) or pe == 0 else round(float(pe), 4),
                "pb": None if pd.isna(pb) or pb == 0 else round(float(pb), 4),
                "roe_ttm": None,
                "revenue_yoy": None,
                "profit_yoy": None,
                "market_cap": None,
            })
        if rows:
            val_df = pd.DataFrame(rows)
            try:
                self._storage.save_fundamentals(symbol, val_df)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to persist valuation for %s", symbol)
