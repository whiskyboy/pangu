"""StockPool Protocol — PRD §4.1.4 / §6."""

from __future__ import annotations

from typing import Protocol

from pangu.models import StockMeta


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

    def get_all_symbols(self) -> list[str]:
        """Return all tracked symbols (watchlist + configured indices, deduplicated)."""
        ...

    def get_stock_metadata(self) -> dict[str, StockMeta]:
        """Return symbol → StockMeta mapping from DB + watchlist."""
        ...

    def sync_index_constituents(self) -> int:
        """Sync configured index constituents from upstream data source."""
        ...
