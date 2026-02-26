"""Composite FundamentalDataProvider — orchestrates provider chain + caching."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_FINANCIAL_SYNC_INTERVAL_DAYS = 30


class CompositeFundamentalProvider:
    """Orchestrates a priority-ordered list of fundamental data providers.

    Handles:
    - Provider chain fallback (try each provider in order)
    - SQLite caching via data_sync_log
    - Persistence of financial indicator data

    Parameters
    ----------
    storage : Database
        SQLite database for caching fundamentals and sync log.
    providers : list
        Fundamental data providers ordered by priority (first = highest).
    """

    def __init__(self, storage, providers: list) -> None:
        self._storage = storage
        self._providers = providers

    def get_financial_indicator(
        self, symbol: str, start: str | None = None, end: str | None = None,
    ) -> pd.DataFrame:
        """Return financial indicators with cache + provider chain fallback."""
        from datetime import timedelta

        from pangu.tz import now as tz_now

        # Cache check: if synced within interval, read from DB
        last = self._storage.get_last_sync_date(symbol, "financial_indicator")
        if last is not None:
            cutoff = (tz_now() - timedelta(days=_FINANCIAL_SYNC_INTERVAL_DAYS)).strftime("%Y-%m-%d")
            if last >= cutoff:
                cached = self._load_financial_from_db(symbol, start, end)
                if not cached.empty:
                    return cached

        # Try providers in priority order
        df, source = None, "unknown"
        for p in self._providers:
            try:
                df = p.get_financial_indicator(symbol, start=start, end=end)
                if df is not None and not df.empty:
                    source = type(p).__name__
                    break
            except NotImplementedError:
                continue
            except Exception:  # noqa: BLE001
                logger.warning(
                    "%s.get_financial_indicator failed for %s, trying next provider",
                    type(p).__name__, symbol, exc_info=True,
                )
                df = None
                continue

        if df is None or df.empty:
            return self._load_financial_from_db(symbol, start, end)

        # Persist + update sync log
        self._persist_financial(symbol, df)
        self._storage.update_sync_log(
            symbol, "financial_indicator", "ok", source,
            last_date=tz_now().strftime("%Y-%m-%d"),
        )

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_financial_from_db(
        self, symbol: str, start: str | None, end: str | None,
    ) -> pd.DataFrame:
        """Read financial indicators from fundamentals table."""
        try:
            s = start or "2000-01-01"
            e = end or "2099-12-31"
            df = self._storage.load_fundamentals(symbol, s, e)
            # Filter to rows with financial data (not just valuation rows)
            if not df.empty:
                mask = df["roe_ttm"].notna() | df["revenue_yoy"].notna() | df["profit_yoy"].notna()
                return df[mask].reset_index(drop=True)
            return df
        except Exception:  # noqa: BLE001
            logger.debug("DB read failed for financial indicator %s", symbol)
            return pd.DataFrame()

    def _persist_financial(self, symbol: str, result: pd.DataFrame) -> None:
        """Save financial indicators to fundamentals table."""
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
