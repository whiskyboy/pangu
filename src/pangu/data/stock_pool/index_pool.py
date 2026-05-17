"""Index-constituent-based StockPool.

Tracks the configurable A-share index universes (defaults to CSI300 + CSI500)
from the ``index_constituents`` DB table. No manual watchlist concept —
universe is entirely driven by exchange constituents history.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pangu.data.storage import Database
    from pangu.models import StockMeta

logger = logging.getLogger(__name__)


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
        """Return symbol → StockMeta sourced from ``index_constituents``."""
        from pangu.models import StockMeta

        meta: dict[str, StockMeta] = {}
        for row in self._storage.load_all_index_constituents():
            sym = row["symbol"]
            if sym not in meta:
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

        Iterates over ``self._indices``, calls ``index_stock_cons(symbol=code)``
        for each, enriches with sector via ``stock_individual_info_em``, and
        persists to the ``index_constituents`` DB table.

        Returns the total number of rows saved.
        """
        import akshare as ak

        from pangu.tz import today_str
        from pangu.utils import CircuitBreaker, retry_call

        circuit = CircuitBreaker()
        today = today_str()
        all_rows: list[dict[str, str]] = []

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
                sector = ""
                try:
                    info_df = retry_call(
                        lambda s=symbol: ak.stock_individual_info_em(symbol=s),
                        circuit=circuit,
                    )
                    if info_df is not None and not info_df.empty:
                        info_map = {str(r["item"]): str(r["value"]) for _, r in info_df.iterrows()}
                        sector = info_map.get("行业", "")
                except Exception:  # noqa: BLE001
                    logger.warning("sync_index(%s): sector lookup failed for %s", index_code, symbol)
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
                    logger.info("sync_index(%s): %d/%d stocks processed", index_code, idx, total)

        count = self._storage.save_index_constituents(all_rows)
        removed = self._storage.delete_stale_index_constituents(self._indices)
        if removed:
            logger.info("sync_index: removed %d constituents from unconfigured indices", removed)
        logger.info("sync_index: %d constituents saved across %d indices", count, len(self._indices))
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
        """Backfill sector classification for historical constituents via AkShare.

        Queries ``stock_individual_info_em`` for each symbol that has no sector
        in the DB. Updates all historical rows for that symbol.

        Parameters
        ----------
        symbols : set[str] or None
            Symbols to backfill. If None, queries DB for all symbols missing sector.

        Returns the number of DB rows updated.
        """
        import akshare as ak

        from pangu.utils import CircuitBreaker, retry_call

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

        logger.info("backfill_sectors: %d symbols to process", len(symbols))
        sector_map: dict[str, str] = {}
        failed = 0

        for i, symbol in enumerate(sorted(symbols), 1):
            try:
                info_df = retry_call(
                    lambda s=symbol: ak.stock_individual_info_em(symbol=s),
                    circuit=circuit,
                )
                if info_df is not None and not info_df.empty:
                    info_map = {str(r["item"]): str(r["value"]) for _, r in info_df.iterrows()}
                    sector = info_map.get("行业", "")
                    if sector:
                        sector_map[symbol] = sector
            except Exception:  # noqa: BLE001
                failed += 1
                logger.debug("backfill_sectors: failed for %s", symbol)

            if i % 50 == 0:
                logger.info(
                    "backfill_sectors: %d/%d symbols queried (%d found, %d failed)",
                    i,
                    len(symbols),
                    len(sector_map),
                    failed,
                )

        updated = self._storage.update_constituent_sectors(sector_map)
        logger.info(
            "backfill_sectors: %d symbols queried, %d sectors found, %d DB rows updated, %d failed",
            len(symbols),
            len(sector_map),
            updated,
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
