"""StockPool Protocol — index-constituent based universe."""

from __future__ import annotations

from typing import Protocol

from pangu.models import StockMeta


class StockPool(Protocol):
    """Interface for the index-constituent stock pool."""

    def get_all_symbols(self) -> list[str]:
        """Return all tracked symbols (deduplicated)."""
        ...

    def get_stock_metadata(self) -> dict[str, StockMeta]:
        """Return symbol → StockMeta mapping."""
        ...

    def sync_index_constituents(self) -> int:
        """Sync configured index constituents from upstream data source."""
        ...
