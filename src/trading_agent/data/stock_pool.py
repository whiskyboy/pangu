"""StockPool Protocol — PRD §4.1.4 / §6."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml


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

