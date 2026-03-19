"""Composite FundamentalDataProvider — orchestrates provider chain + caching."""

from __future__ import annotations

import logging

import pandas as pd

from pangu.utils import quarter_dates

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
        *, force: bool = False,
    ) -> pd.DataFrame:
        """Return financial indicators with cache + provider chain fallback.

        If *force* is True, bypass sync interval check and always fetch from providers.
        """
        from datetime import timedelta

        from pangu.tz import now as tz_now

        # Cache check: if synced within interval, read from DB (skip when force)
        if not force:
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
        # Save raw API response if available (for future field expansion)
        raw = getattr(p, "_last_raw_response", None) if df is not None else None
        if raw is not None and not raw.empty:
            try:
                self._storage.save_fundamentals_raw(symbol, raw)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to save raw fundamentals for %s", symbol)
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
        """Save financial indicators to fundamentals table.

        Passes through all columns from the provider result. Columns not in
        ``_FUND_COLS`` are silently ignored by ``save_fundamentals()``.
        """
        if result.empty:
            return
        df = result.copy()
        df["symbol"] = symbol
        # Ensure columns that come from other sources are not overwritten
        for col in ("pe_ttm", "pb", "market_cap", "ps_ttm", "pcf_ttm", "pub_date"):
            if col not in df.columns:
                df[col] = None
        try:
            self._storage.save_fundamentals(symbol, df)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to persist financials for %s", symbol)

    # ------------------------------------------------------------------
    # Gross margin backfill via stock_yjbb_em
    # ------------------------------------------------------------------

    def refresh_gross_margin(
        self, start: str, end: str, *, incremental: bool = False,
    ) -> tuple[int, int]:
        """Backfill gross_margin from ``stock_yjbb_em`` API (per-quarter batch).

        Parameters
        ----------
        start, end : str
            Date range in ``YYYY-MM-DD`` format.
        incremental : bool
            If True, only fetch the last 2 quarters (for routine use).

        Returns
        -------
        (ok_quarters, fail_quarters)
        """
        provider = None
        for p in self._providers:
            if hasattr(p, "fetch_gross_margin_batch"):
                provider = p
                break
        if provider is None:
            logger.warning("No provider supports fetch_gross_margin_batch")
            return 0, 0

        quarters = quarter_dates(start, end)
        if incremental:
            quarters = quarters[-2:]

        ok, fail = 0, 0
        for q in quarters:
            try:
                data = provider.fetch_gross_margin_batch(q)
                if data:
                    db_date = f"{q[:4]}-{q[4:6]}-{q[6:]}"
                    affected = self._storage.update_gross_margin_batch(db_date, data)
                    ok += 1
                    logger.info("gross_margin %s: %d/%d stocks updated", q, affected, len(data))
                else:
                    fail += 1
                    logger.warning("gross_margin %s: empty response", q)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to fetch gross_margin for %s", q, exc_info=True)
                fail += 1
        return ok, fail

    # ------------------------------------------------------------------
    # Publication date backfill via BaoStock
    # ------------------------------------------------------------------

    def refresh_pub_dates(
        self, symbols: list[str], start: str,
    ) -> tuple[int, int]:
        """Backfill ``pub_date`` (first announcement date) for quarterly rows.

        Uses BaoStock ``query_profit_data`` which returns ``pubDate``.
        Discovers the provider via ``hasattr`` (same pattern as
        ``refresh_gross_margin``).

        Parameters
        ----------
        symbols : list[str]
            Stock symbols to process.
        start : str
            Start date in ``YYYY-MM-DD`` format (determines start_year).

        Returns
        -------
        (ok_stocks, fail_stocks)
        """
        provider = None
        for p in self._providers:
            if hasattr(p, "fetch_pub_dates"):
                provider = p
                break
        if provider is None:
            logger.warning("No provider supports fetch_pub_dates")
            return 0, 0

        start_year = int(start[:4])
        ok, fail = 0, 0
        total = len(symbols)
        for i, sym in enumerate(symbols, 1):
            try:
                data = provider.fetch_pub_dates(sym, start_year=start_year)
                if data:
                    self._storage.update_pub_date_batch(sym, data)
                    ok += 1
                else:
                    fail += 1
            except Exception:  # noqa: BLE001
                logger.debug("Failed to fetch pub_dates for %s", sym, exc_info=True)
                fail += 1
            if i % 10 == 0 or i == total:
                logger.info("pub_dates [%d/%d] ok=%d fail=%d", i, total, ok, fail)
        return ok, fail
