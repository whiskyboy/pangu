"""Index-constituent-based StockPool.

Tracks the configurable A-share index universes (defaults to CSI300 + CSI500)
from the ``index_constituents`` DB table. No manual watchlist concept —
universe is entirely driven by exchange constituents history.

Stock metadata (company name, industry, listing date, main business, etc.)
is sourced from cninfo ``stock_profile_cninfo`` and stored in the
``stock_profiles`` DB table. ``index_constituents.sector`` is dual-written
with the coarse-grained cninfo industry classification so backtest
``--max-per-sector`` and historical PIT snapshots stay self-contained.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pangu.data.storage import Database
    from pangu.models import StockMeta
    from pangu.utils import CircuitBreaker

logger = logging.getLogger(__name__)


# cninfo wide-format column → stock_profiles field
_CNINFO_FIELD_MAP: dict[str, str] = {
    "A股简称": "name",
    "公司名称": "full_name",
    "所属行业": "sector",
    "上市日期": "list_date",
    "主营业务": "main_business",
    "注册地址": "registered_area",
}


def _fetch_profile_cninfo(symbol: str, circuit: CircuitBreaker) -> dict[str, str] | None:
    """Fetch a single symbol's profile via ``ak.stock_profile_cninfo``.

    Returns a dict with keys from ``_CNINFO_FIELD_MAP.values()``, or ``None``
    if the upstream returned empty (typical for delisted stocks) or all
    retries were exhausted. All values are stripped strings, missing values
    become empty strings.
    """
    import akshare as ak

    from pangu.utils import retry_call

    try:
        df = retry_call(
            lambda s=symbol: ak.stock_profile_cninfo(symbol=s),
            circuit=circuit,
        )
    except Exception:  # noqa: BLE001
        logger.warning("cninfo lookup failed for %s", symbol, exc_info=False)
        return None
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    profile: dict[str, str] = {dest: "" for dest in _CNINFO_FIELD_MAP.values()}
    for col, dest in _CNINFO_FIELD_MAP.items():
        if col in row.index:
            val = row[col]
            if val is None:
                profile[dest] = ""
            else:
                try:
                    if val != val:  # NaN check
                        profile[dest] = ""
                        continue
                except Exception:  # noqa: BLE001
                    pass
                profile[dest] = str(val).strip()
    return profile


class IndexStockPool:
    """Stock pool sourced from ``index_constituents`` DB table.

    Provides the universe (`get_all_symbols`) and metadata (`get_stock_metadata`)
    used by T3 / T6 / backtest / training, plus the sync entry points used by
    T1 (`sync_index_constituents`, `sync_trading_calendar`) and the backfill CLI
    (`sync_historical_constituents`, `backfill_sectors`).

    Parameters
    ----------
    storage : Database
        SQLite storage layer.
    indices : list[str] | None
        Index codes whose constituents form the pool (default ``["000300"]``).
    """

    def __init__(
        self,
        storage: Database,
        *,
        indices: list[str] | None = None,
    ) -> None:
        self._storage = storage
        self._indices = indices or ["000300"]

    # -- Protocol methods --

    def get_all_symbols(self) -> list[str]:
        """Return constituent symbols of all configured indices, deduplicated."""
        return self._get_index_stocks()

    def get_stock_metadata(self) -> dict[str, StockMeta]:
        """Return symbol → StockMeta.

        Primary source: ``stock_profiles`` (cninfo full profile). Falls back
        to ``index_constituents`` (name + sector only) when the profile
        record is missing (cold start / cninfo failed). The universe is the
        latest ``index_constituents`` snapshot — profiles for non-constituent
        symbols are ignored.
        """
        from pangu.models import StockMeta

        profiles = self._storage.load_all_stock_profiles()

        meta: dict[str, StockMeta] = {}
        for row in self._storage.load_all_index_constituents():
            sym = row["symbol"]
            if sym in meta:
                continue
            prof = profiles.get(sym)
            if prof is not None:
                meta[sym] = StockMeta(
                    name=prof["name"] or row.get("name") or "",
                    sector=prof["sector"] or row.get("sector") or "",
                    full_name=prof["full_name"],
                    list_date=prof["list_date"],
                    main_business=prof["main_business"],
                    registered_area=prof["registered_area"],
                )
            else:
                meta[sym] = StockMeta(
                    name=row.get("name") or "",
                    sector=row.get("sector") or "",
                )
        return meta

    # -- Trading calendar --

    def sync_trading_calendar(self) -> int:
        """Pull trading calendar from AkShare and save to SQLite.

        Returns the number of new dates inserted.
        """
        import akshare as ak

        from pangu.utils import CircuitBreaker, retry_call

        df = retry_call(
            lambda: ak.tool_trade_date_hist_sina(),
            circuit=CircuitBreaker(),
        )
        if df is None or df.empty:
            logger.warning("sync_trading_calendar: no data returned")
            return 0

        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in df["trade_date"]]
        count = self._storage.save_trading_calendar(dates)
        logger.info("sync_trading_calendar: %d new dates saved (%d total)", count, len(dates))
        return count

    # -- Index constituents --

    def sync_index_constituents(self) -> int:
        """Fetch constituents for all configured indices from AkShare.

        For each index, calls ``index_stock_cons(symbol=code)`` to get the
        current constituent list, then enriches each stock via cninfo
        ``stock_profile_cninfo`` and dual-writes:

          * ``index_constituents`` — (date, index_code, symbol, name, sector)
            PIT snapshot with coarse cninfo industry classification.
          * ``stock_profiles`` — (symbol, name, full_name, sector, list_date,
            main_business, registered_area) latest-value table consumed by
            the LLM judge prompt.

        Symbols whose cninfo lookup fails get sector="" in the constituents
        snapshot and no row in ``stock_profiles``.

        Returns the total number of ``index_constituents`` rows saved.
        """
        import akshare as ak

        from pangu.tz import today_str
        from pangu.utils import CircuitBreaker, retry_call

        circuit = CircuitBreaker()
        today = today_str()
        all_rows: list[dict[str, str]] = []
        profiles: list[dict[str, Any]] = []
        cninfo_failed = 0

        for index_code in self._indices:
            df = retry_call(
                lambda code=index_code: ak.index_stock_cons(symbol=code),
                circuit=circuit,
            )
            if df is None or df.empty:
                logger.warning("sync_index(%s): no constituents returned", index_code)
                continue

            total = len(df)
            for idx, (_, row) in enumerate(df.iterrows(), 1):
                symbol = str(row["品种代码"])
                name = str(row["品种名称"])
                profile = _fetch_profile_cninfo(symbol, circuit)
                if profile is None:
                    sector = ""
                    cninfo_failed += 1
                else:
                    sector = profile["sector"]
                    profiles.append({"symbol": symbol, **profile})
                all_rows.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "index_code": index_code,
                        "sector": sector,
                        "date": today,
                    }
                )
                if idx % 50 == 0:
                    logger.info(
                        "sync_index(%s): %d/%d stocks processed (%d profiles ok)",
                        index_code,
                        idx,
                        total,
                        len(profiles),
                    )

        count = self._storage.save_index_constituents(all_rows)
        profile_count = self._storage.save_stock_profiles_batch(profiles)
        removed = self._storage.delete_stale_index_constituents(self._indices)
        if removed:
            logger.info("sync_index: removed %d constituents from unconfigured indices", removed)
        logger.info(
            "sync_index: %d constituents saved across %d indices (%d profiles written, %d cninfo failures)",
            count,
            len(self._indices),
            profile_count,
            cninfo_failed,
        )
        return count

    def sync_historical_constituents(
        self,
        start: str = "2019-01-01",
        end: str | None = None,
    ) -> tuple[int, set[str]]:
        """Fetch historical index constituents from BaoStock, sampled semi-annually.

        Uses ``bs.query_hs300_stocks(date=)`` and ``bs.query_zz500_stocks(date=)``
        to get constituents at each sampling date. A-share indices adjust every
        June and December, so semi-annual sampling matches the real cadence.

        Returns (rows_saved, unique_symbols).
        """
        import baostock as bs
        import pandas as pd

        from pangu.tz import today_str
        from pangu.utils import CircuitBreaker, retry_call

        end = end or today_str()
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

        circuit = CircuitBreaker()

        dates = pd.date_range(start, end, freq="6MS").strftime("%Y-%m-%d").tolist()
        if end not in dates:
            dates.append(end)

        _INDEX_QUERIES = {
            "000300": bs.query_hs300_stocks,
            "000905": bs.query_zz500_stocks,
        }

        all_rows: list[dict] = []
        all_symbols: set[str] = set()

        for d in dates:
            for idx_code in self._indices:
                query_fn = _INDEX_QUERIES.get(idx_code)
                if query_fn is None:
                    continue

                def _query(fn=query_fn, dt=d):
                    rs = fn(date=dt)
                    if rs.error_code != "0":
                        if "未登录" in getattr(rs, "error_msg", ""):
                            bs.logout()
                            lg = bs.login()
                            if lg.error_code != "0":
                                raise RuntimeError(f"BaoStock re-login failed: {lg.error_msg}")
                        raise RuntimeError(getattr(rs, "error_msg", "unknown"))
                    return rs

                try:
                    rs = retry_call(_query, circuit=circuit)
                except Exception:  # noqa: BLE001
                    logger.warning("sync_historical(%s, %s): query failed", idx_code, d)
                    continue

                df = rs.get_data()
                if df.empty:
                    continue

                for _, r in df.iterrows():
                    code = r["code"]
                    if not code.startswith(("sh.", "sz.")):
                        continue
                    symbol = code[3:]
                    all_rows.append(
                        {
                            "date": d,
                            "index_code": idx_code,
                            "symbol": symbol,
                            "name": r.get("code_name", ""),
                        }
                    )
                    all_symbols.add(symbol)

            logger.info("sync_historical: %s — %d rows so far", d, len(all_rows))

        bs.logout()

        count = self._storage.save_index_constituents(all_rows)
        logger.info(
            "sync_historical: %d records saved, %d unique stocks, %d dates",
            count,
            len(all_symbols),
            len(dates),
        )
        return count, all_symbols

    def backfill_sectors(self, symbols: set[str] | None = None) -> int:
        """Backfill sector classification for historical constituents via cninfo.

        For each symbol missing a sector in ``index_constituents``, calls
        ``stock_profile_cninfo`` once. The fetched profile is used for two
        writes:

          * ``index_constituents.sector`` (UPDATE on all historical rows of
            this symbol — broadcasts the current-snapshot sector across
            history; this is a documented "current-value broadcast fill",
            not true historical retrieval).
          * ``stock_profiles`` (full upsert with all 6 fields).

        Dual-writing here keeps the CLI workflow self-sufficient: a user
        who runs ``pangu backfill constituents --with-sector`` after a
        clean-and-rebackfill gets ``stock_profiles`` populated as a side
        effect, without needing to also run T1 / ``pangu run init``.

        Parameters
        ----------
        symbols : set[str] or None
            Symbols to backfill. If None, queries DB for all symbols missing sector.

        Returns the number of ``index_constituents`` rows updated.
        """
        from pangu.utils import CircuitBreaker

        circuit = CircuitBreaker()

        if symbols is None:
            with self._storage._lock:
                rows = self._storage._conn.execute(
                    "SELECT DISTINCT symbol FROM index_constituents WHERE sector IS NULL OR sector = ''"
                ).fetchall()
            symbols = {r[0] for r in rows}

        if not symbols:
            logger.info("backfill_sectors: all symbols already have sector data")
            return 0

        logger.info("backfill_sectors: %d symbols to process via cninfo", len(symbols))
        sector_map: dict[str, str] = {}
        profiles: list[dict[str, Any]] = []
        failed = 0

        for i, symbol in enumerate(sorted(symbols), 1):
            profile = _fetch_profile_cninfo(symbol, circuit)
            if profile is None:
                failed += 1
            else:
                if profile["sector"]:
                    sector_map[symbol] = profile["sector"]
                profiles.append({"symbol": symbol, **profile})

            if i % 50 == 0:
                logger.info(
                    "backfill_sectors: %d/%d symbols queried (%d found, %d failed)",
                    i,
                    len(symbols),
                    len(sector_map),
                    failed,
                )

        updated = self._storage.update_constituent_sectors(sector_map)
        profile_count = self._storage.save_stock_profiles_batch(profiles)
        logger.info(
            "backfill_sectors: %d symbols queried, %d sectors found, "
            "%d DB rows updated, %d profiles written, %d failed",
            len(symbols),
            len(sector_map),
            updated,
            profile_count,
            failed,
        )
        return updated

    # -- Internal --

    def _get_index_stocks(self) -> list[str]:
        """Return constituent symbols from all configured indices (deduplicated)."""
        rows = self._storage.load_all_index_constituents()
        seen: set[str] = set()
        result: list[str] = []
        for r in rows:
            sym = r["symbol"]
            if sym not in seen:
                seen.add(sym)
                result.append(sym)
        return result
