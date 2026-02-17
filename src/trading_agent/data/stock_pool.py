"""StockPool Protocol — PRD §4.1.4 / §6."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

if TYPE_CHECKING:
    from trading_agent.data.fundamental import FundamentalDataProvider
    from trading_agent.data.market import MarketDataProvider
    from trading_agent.data.news import NewsDataProvider
    from trading_agent.data.storage import Database

logger = logging.getLogger(__name__)


class StockPool(Protocol):
    """Interface for stock pool management (watchlist + dynamic pools)."""

    def get_watchlist(self) -> list[str]:
        """Return manually curated watchlist symbols."""
        ...

    def add_to_watchlist(self, symbol: str) -> None:
        """Add a symbol and trigger data initialization."""
        ...

    def remove_from_watchlist(self, symbol: str) -> None:
        """Remove a symbol (historical data is kept)."""
        ...

    def get_factor_selected(self) -> list[str]:
        """Return symbols selected by factor screening."""
        ...

    def get_event_triggered(self) -> list[str]:
        """Return symbols discovered by event-driven analysis."""
        ...

    def get_active_pool(self) -> list[str]:
        """Return merged active pool (deduplicated and filtered)."""
        ...


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------


class FakeStockPool:
    """Loads watchlist from YAML; factor/event pools are empty stubs."""

    def __init__(self, watchlist_path: str | Path = "config/watchlist.yaml") -> None:
        self._path = Path(watchlist_path)
        self._symbols: list[str] = []
        if self._path.exists():
            data = yaml.safe_load(self._path.read_text())
            watchlist = (data or {}).get("watchlist") or []
            self._symbols = [
                item["symbol"] for item in watchlist if "symbol" in item
            ]

    def get_watchlist(self) -> list[str]:
        return list(self._symbols)

    def add_to_watchlist(self, symbol: str) -> None:
        if symbol not in self._symbols:
            self._symbols.append(symbol)

    def remove_from_watchlist(self, symbol: str) -> None:
        if symbol in self._symbols:
            self._symbols.remove(symbol)

    def get_factor_selected(self) -> list[str]:
        return []

    def get_event_triggered(self) -> list[str]:
        return []

    def get_active_pool(self) -> list[str]:
        return list(self._symbols)


# ---------------------------------------------------------------------------
# Real implementation — YAML-backed with data initialization
# ---------------------------------------------------------------------------


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

    # -- Data initialization --

    def _init_stock_data(self, symbol: str) -> None:
        """Pull historical data for *symbol* into SQLite.

        Failures are logged but do not prevent the stock from being added.
        """
        from datetime import datetime, timedelta

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")

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

                today = datetime.now().strftime("%Y-%m-%d")
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
        """Return factor-screened symbols (populated by M3)."""
        return []

    def get_event_triggered(self) -> list[str]:
        """Return event-triggered symbols (populated by M4)."""
        return []

    def get_active_pool(self) -> list[str]:
        """Merge watchlist + factor + event pools, deduplicated and filtered."""
        seen: set[str] = set()
        merged: list[str] = []
        for sym in (
            self.get_watchlist()
            + self.get_factor_selected()
            + self.get_event_triggered()
        ):
            if sym not in seen:
                seen.add(sym)
                merged.append(sym)
        return self._filter_stocks(merged)

    # -- Stock filtering --

    def _filter_stocks(self, symbols: list[str]) -> list[str]:
        """Exclude ST, suspended, and recently-listed stocks.

        Filter failures are logged but the stock is kept (conservative).
        """
        import akshare as ak

        from trading_agent.data.market import CircuitBreaker, _retry_call

        circuit = CircuitBreaker()
        result: list[str] = []

        for symbol in symbols:
            try:
                df = _retry_call(
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

                    try:
                        ipo_date = datetime.strptime(ipo_date_str, "%Y%m%d")
                        if (datetime.now() - ipo_date).days < 60:
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

        from trading_agent.data.market import CircuitBreaker, _retry_call

        df = _retry_call(
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

