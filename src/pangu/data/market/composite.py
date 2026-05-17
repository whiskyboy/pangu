"""Composite MarketDataProvider — orchestrates provider chain + caching."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_STD_COLS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
    "turn",
    "preclose",
    "tradestatus",
    "is_st",
]

# Column rename: BaoStock field names → daily_bars DB column names
_COL_RENAME = {
    "isST": "is_st",
}


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

    def get_daily_bars(self, symbol: str, start: str, end: str, *, force: bool = False) -> pd.DataFrame:
        """Fetch daily bars with cache + provider chain fallback.

        If *force* is True, bypass sync_log and fetch the full [start, end] range.
        """
        # 1. Check sync_log once
        last = None if force else self._storage.get_last_sync_date(symbol, "daily_bars")

        # 2. Cache hit: already have data up to end
        if last is not None and last >= end:
            return self._storage.load_daily_bars(symbol, start, end)

        # 3. Determine fetch range (full range if force, incremental otherwise)
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
                    type(p).__name__,
                    symbol,
                    exc_info=True,
                )
                df = None
                continue

        # 4. No data from any provider
        if df is None or df.empty:
            return self._storage.load_daily_bars(symbol, start, end)

        # 5. Extract PE/PB + market_cap before trimming columns
        self._persist_valuation(symbol, df)

        # 6. Rename BaoStock columns to DB column names, trim to standard + persist
        df = df.rename(columns=_COL_RENAME)
        result = df[[c for c in _STD_COLS if c in df.columns]].copy()
        if not result.empty:
            self._storage.save_daily_bars(symbol, result)
            self._storage.update_sync_log(
                symbol,
                "daily_bars",
                "ok",
                source,
                last_date=result["date"].max(),
            )

        # 7. Refresh historical adj_factor if provider supports it
        self._refresh_adj_factor(symbol)

        return self._storage.load_daily_bars(symbol, start, end)

    def get_index_daily_bars(self, symbol: str, start: str, end: str, *, force: bool = False) -> pd.DataFrame:
        """Fetch index daily bars with cache + provider chain fallback."""
        last = None if force else self._storage.get_last_sync_date(symbol, "daily_bars")

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
                    type(p).__name__,
                    symbol,
                    exc_info=True,
                )
                df = None
                continue

        if df is None or df.empty:
            return self._storage.load_daily_bars(symbol, start, end)

        if not df.empty:
            self._storage.save_daily_bars(symbol, df)
            self._storage.update_sync_log(
                symbol,
                "daily_bars",
                "ok",
                source,
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
                    type(p).__name__,
                    exc_info=True,
                )
                continue
        return pd.DataFrame(
            columns=["symbol", "name", "date", "open", "high", "low", "close", "volume", "change_pct", "source"]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist_valuation(self, symbol: str, df: pd.DataFrame) -> None:
        """Extract peTTM/pbMRQ/market_cap from daily bars and save to fundamentals table.

        Market cap is derived from turn (turnover rate):
        circ_shares = volume / (turn / 100), circ_market_cap = close * circ_shares.
        This gives circulating market cap — the A-share industry standard for size factor.
        """
        pe_col = df.get("peTTM")
        pb_col = df.get("pbMRQ")
        turn_col = df.get("turn")
        if pe_col is None and pb_col is None and turn_col is None:
            return

        rows = []
        for _, row in df.iterrows():
            pe = pd.to_numeric(row.get("peTTM", ""), errors="coerce")
            pb = pd.to_numeric(row.get("pbMRQ", ""), errors="coerce")
            ps = pd.to_numeric(row.get("psTTM", ""), errors="coerce")
            pcf = pd.to_numeric(row.get("pcfNcfTTM", ""), errors="coerce")
            turn = pd.to_numeric(row.get("turn", ""), errors="coerce")
            close = pd.to_numeric(row.get("close", ""), errors="coerce")
            volume = pd.to_numeric(row.get("volume", ""), errors="coerce")

            # Compute circulating market cap from turnover rate
            mktcap = None
            if not pd.isna(turn) and turn > 0 and not pd.isna(close) and not pd.isna(volume) and volume > 0:
                circ_shares = volume / (turn / 100)
                mktcap = round(float(close * circ_shares), 2)

            if pd.isna(pe) and pd.isna(pb) and pd.isna(ps) and pd.isna(pcf) and mktcap is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "date": row["date"],
                    "pe_ttm": None if pd.isna(pe) or pe == 0 else round(float(pe), 4),
                    "pb": None if pd.isna(pb) or pb == 0 else round(float(pb), 4),
                    "ps_ttm": None if pd.isna(ps) or ps == 0 else round(float(ps), 4),
                    "pcf_ttm": None if pd.isna(pcf) or pcf == 0 else round(float(pcf), 4),
                    "roe_ttm": None,
                    "revenue_yoy": None,
                    "profit_yoy": None,
                    "market_cap": mktcap,
                }
            )
        if rows:
            val_df = pd.DataFrame(rows)
            try:
                self._storage.save_fundamentals(symbol, val_df)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to persist valuation for %s", symbol)

    def _refresh_adj_factor(self, symbol: str) -> None:
        """Refresh adj_factor for all historical rows of a stock.

        Called after new bars are saved. Fetches the latest adjustment events
        from the primary provider and batch-updates any rows whose factor
        has changed (e.g. after a new dividend event).
        """
        import bisect

        # Only BaoStock provides adj factors
        provider = None
        for p in self._providers:
            if hasattr(p, "fetch_adjust_factors"):
                provider = p
                break
        if provider is None:
            return

        try:
            events = provider.fetch_adjust_factors(symbol)
        except Exception:
            logger.debug("Failed to refresh adj_factor for %s", symbol)
            return
        if not events:
            return

        event_dates = [e[0] for e in events]
        event_factors = [e[1] for e in events]

        rows = self._storage.get_daily_bar_dates(symbol)

        updates = []
        for date, old_factor in rows:
            idx = bisect.bisect_right(event_dates, date) - 1
            new_factor = event_factors[idx] if idx >= 0 else event_factors[0]
            if abs(new_factor - (old_factor or 1.0)) > 1e-8:
                updates.append((new_factor, date))

        if updates:
            self._storage.update_adj_factors(symbol, updates)
            logger.debug("Updated %d adj_factor rows for %s", len(updates), symbol)
