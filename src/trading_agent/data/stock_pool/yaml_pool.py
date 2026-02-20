"""YAML-backed StockPoolManager — PRD §4.1.4."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from trading_agent.data.fundamental.protocol import FundamentalDataProvider
    from trading_agent.data.market.protocol import MarketDataProvider
    from trading_agent.data.news.protocol import NewsDataProvider
    from trading_agent.data.storage import Database

logger = logging.getLogger(__name__)


class StockPoolManager:
    """YAML-persisted stock pool with automatic data initialization.

    When a stock is added via :meth:`add_to_watchlist`, the pool pulls
    historical daily bars, fundamentals, news, and announcements into
    SQLite so that downstream engines have data to work with immediately.

    Parameters
    ----------
    watchlist_path : str | Path
        Path to ``watchlist.yaml``.
    storage : Database
        SQLite storage layer for persisting fetched data.
    market_provider : MarketDataProvider
        Provider for daily K-line bars.
    news_provider : NewsDataProvider
        Provider for stock news and announcements.
    fundamental_provider : FundamentalDataProvider
        Provider for valuation and financial indicators.
    """

    def __init__(
        self,
        watchlist_path: str | Path,
        storage: Database,
        market_provider: MarketDataProvider,
        news_provider: NewsDataProvider,
        fundamental_provider: FundamentalDataProvider,
    ) -> None:
        self._path = Path(watchlist_path)
        self._storage = storage
        self._market = market_provider
        self._news = news_provider
        self._fundamental = fundamental_provider
        self._entries: list[dict[str, str]] = []
        self._load_yaml()
        self._backfill_missing_data()

    # -- YAML I/O --

    def _load_yaml(self) -> None:
        """Load watchlist entries from YAML file."""
        if not self._path.exists():
            self._entries = []
            return
        try:
            data = yaml.safe_load(self._path.read_text()) or {}
        except yaml.YAMLError:
            logger.warning("Failed to parse %s, using empty watchlist", self._path, exc_info=True)
            self._entries = []
            return
        self._entries = list(data.get("watchlist") or [])

    def _save_yaml(self) -> None:
        """Write current entries back to YAML file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"watchlist": self._entries}
        self._path.write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        )

    # -- Data backfill --

    def _backfill_missing_data(self) -> None:
        """Check watchlist stocks for missing or stale data and pull if needed."""
        from trading_agent.utils import date_str
        from trading_agent.tz import today_str

        today = today_str()
        cutoff = date_str(days_ago=7)

        for symbol in self.get_watchlist():
            bars = self._storage.load_daily_bars(symbol, "2000-01-01", today)
            if bars.empty:
                logger.info("No data for %s, running initial data pull…", symbol)
                self._init_stock_data(symbol)
            else:
                last_date = str(bars["date"].iloc[-1])
                if last_date < cutoff:
                    logger.info(
                        "Stale data for %s (last: %s), backfilling…",
                        symbol, last_date,
                    )
                    self._init_stock_data(symbol)

    # -- Data initialization --

    def _init_stock_data(self, symbol: str) -> None:
        """Pull historical data for *symbol* into SQLite.

        Failures are logged but do not prevent the stock from being added.
        """
        from trading_agent.utils import date_str

        end = date_str()
        start = date_str(days_ago=200)

        # 1. Historical daily bars (~120 trading days ≈ 200 calendar days)
        try:
            df = self._market.get_daily_bars(symbol, start, end)
            if df is not None and not df.empty:
                self._storage.save_daily_bars(symbol, df)
                logger.info("init %s: %d daily bars saved", symbol, len(df))
        except Exception:  # noqa: BLE001
            logger.warning("init %s: daily bars failed", symbol, exc_info=True)

        # 2. Fundamentals (valuation + financial indicators)
        try:
            val = self._fundamental.get_valuation(symbol)
            fin = self._fundamental.get_financial_indicator(symbol)
            if val or (fin is not None and not fin.empty):
                import pandas as pd

                today = date_str()
                row: dict[str, Any] = {"date": today}
                if val:
                    row.update({
                        "pe_ttm": val.get("pe_ttm"),
                        "pb": val.get("pb"),
                        "market_cap": val.get("market_cap"),
                    })
                if fin is not None and not fin.empty:
                    latest = fin.iloc[-1]
                    row.update({
                        "roe_ttm": latest.get("roe_ttm"),
                        "revenue_yoy": latest.get("revenue_yoy"),
                        "profit_yoy": latest.get("profit_yoy"),
                    })
                self._storage.save_fundamentals(symbol, pd.DataFrame([row]))
                logger.info("init %s: fundamentals saved", symbol)
        except Exception:  # noqa: BLE001
            logger.warning("init %s: fundamentals failed", symbol, exc_info=True)

        # 3. Recent stock news
        try:
            news = self._news.get_stock_news(symbol, limit=20)
            logger.info("init %s: %d news items fetched", symbol, len(news))
        except Exception:  # noqa: BLE001
            logger.warning("init %s: news failed", symbol, exc_info=True)

        # 4. Recent announcements
        try:
            anns = self._news.get_announcements(symbol, limit=20)
            logger.info("init %s: %d announcements fetched", symbol, len(anns))
        except Exception:  # noqa: BLE001
            logger.warning("init %s: announcements failed", symbol, exc_info=True)

    # -- Protocol methods --

    def get_watchlist(self) -> list[str]:
        """Return symbols from watchlist.yaml."""
        return [e["symbol"] for e in self._entries if "symbol" in e]

    def add_to_watchlist(
        self,
        symbol: str,
        *,
        name: str = "",
        sector: str = "",
    ) -> None:
        """Add *symbol* to watchlist, persist to YAML, and pull initial data."""
        if symbol in self.get_watchlist():
            return

        entry: dict[str, str] = {"symbol": symbol}
        if name:
            entry["name"] = name
        if sector:
            entry["sector"] = sector
        self._entries.append(entry)
        self._save_yaml()

        logger.info("Added %s to watchlist, starting data init…", symbol)
        self._init_stock_data(symbol)

    def remove_from_watchlist(self, symbol: str) -> None:
        """Remove *symbol* from watchlist YAML (historical data is kept)."""
        self._entries = [e for e in self._entries if e.get("symbol") != symbol]
        self._save_yaml()

    def get_factor_selected(self) -> list[str]:
        """Return factor-screened symbols from SQLite factor_pool (top-N)."""
        try:
            from trading_agent.tz import today_str
            pool = self._storage.load_factor_pool(today_str())
            if pool.empty:
                pool = self._storage.load_factor_pool_latest()
            if pool.empty:
                return []
            # Read top_n from config or default to 3
            top_n = 3
            try:
                from trading_agent.config import get_config
                cfg = get_config()
                top_n = cfg.get("strategy", {}).get("top_n", 3)
            except Exception:  # noqa: BLE001
                pass
            return pool[pool["rank"] <= top_n]["symbol"].tolist()
        except Exception:  # noqa: BLE001
            return []

    # -- Stock filtering --

    def _filter_stocks(self, symbols: list[str]) -> list[str]:
        """Exclude ST, suspended, and recently-listed stocks.

        Filter failures are logged but the stock is kept (conservative).
        """
        import akshare as ak

        from trading_agent.utils import CircuitBreaker, retry_call

        circuit = CircuitBreaker()
        result: list[str] = []

        for symbol in symbols:
            try:
                df = retry_call(
                    lambda s=symbol: ak.stock_individual_info_em(symbol=s),
                    circuit=circuit,
                )
                if df is None or df.empty:
                    result.append(symbol)
                    continue

                info: dict[str, Any] = {}
                for _, row in df.iterrows():
                    info[str(row["item"])] = row["value"]

                # ST check
                name = str(info.get("股票简称", ""))
                if "ST" in name.upper():
                    logger.info("filtered %s (%s): ST stock", symbol, name)
                    continue

                # IPO < 60 days check
                ipo_date_str = str(info.get("上市时间", ""))
                if ipo_date_str:
                    from datetime import datetime

                    from trading_agent.tz import _get_tz
                    from trading_agent.tz import now as _tz_now

                    try:
                        ipo_date = datetime.strptime(ipo_date_str, "%Y%m%d").replace(
                            tzinfo=_get_tz()
                        )
                        if (_tz_now() - ipo_date).days < 60:
                            logger.info("filtered %s: IPO < 60 days", symbol)
                            continue
                    except ValueError:
                        pass

                result.append(symbol)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "filter check failed for %s, keeping it", symbol, exc_info=True
                )
                result.append(symbol)

        return result

    # -- Trading calendar --

    def sync_trading_calendar(self) -> int:
        """Pull trading calendar from AkShare and save to SQLite.

        Returns the number of new dates inserted.
        """
        import akshare as ak

        from trading_agent.utils import CircuitBreaker, retry_call

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

    # -- CSI300 constituents --

    def sync_csi300_constituents(self) -> int:
        """Fetch CSI300 constituents and their sector from AkShare.

        Uses ``index_stock_cons("000300")`` for the constituent list and
        ``stock_individual_info_em(symbol)`` per stock for the sector
        ("行业") field.  Results are persisted in the ``index_constituents``
        DB table.

        Returns the number of rows saved.
        """
        import akshare as ak

        from trading_agent.utils import CircuitBreaker, retry_call
        from trading_agent.tz import today_str

        circuit = CircuitBreaker()
        df = retry_call(
            lambda: ak.index_stock_cons(symbol="000300"),
            circuit=circuit,
        )
        if df is None or df.empty:
            logger.warning("sync_csi300: no constituents returned")
            return 0

        today = today_str()
        rows: list[dict[str, str]] = []
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
                logger.warning("sync_csi300: sector lookup failed for %s", symbol)
            rows.append({
                "symbol": symbol,
                "name": name,
                "index_code": "000300",
                "sector": sector,
                "updated_date": today,
            })
            if idx % 50 == 0:
                logger.info("sync_csi300: %d/%d stocks processed", idx, total)

        count = self._storage.save_index_constituents(rows)
        logger.info("sync_csi300: %d constituents saved", count)
        return count

    def _get_csi300_stocks(self) -> list[str]:
        """Return CSI300 constituent symbols from DB."""
        rows = self._storage.load_index_constituents("000300")
        return [r["symbol"] for r in rows]

    def get_name_sector_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        """Return unified (name_map, sector_map) from DB + watchlist YAML.

        Priority: DB (index_constituents) first, then watchlist YAML fills
        any gaps for stocks not in DB (e.g. watchlist stocks outside CSI300).
        """
        name_map: dict[str, str] = {}
        sector_map: dict[str, str] = {}
        # 1. DB constituents (CSI300 etc.)
        for row in self._storage.load_index_constituents("000300"):
            sym = row["symbol"]
            if row.get("name"):
                name_map[sym] = row["name"]
            if row.get("sector"):
                sector_map[sym] = row["sector"]
        # 2. Watchlist YAML fills gaps
        for entry in self._entries:
            sym = entry.get("symbol", "")
            if not sym:
                continue
            if sym not in name_map and entry.get("name"):
                name_map[sym] = entry["name"]
            if sym not in sector_map and entry.get("sector"):
                sector_map[sym] = entry["sector"]
        return name_map, sector_map

    def get_factor_universe(self) -> list[str]:
        """Return watchlist + CSI300 (deduplicated) for factor computation."""
        seen: set[str] = set()
        result: list[str] = []
        for sym in self.get_watchlist() + self._get_csi300_stocks():
            if sym not in seen:
                seen.add(sym)
                result.append(sym)
        return result
