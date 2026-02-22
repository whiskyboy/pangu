"""Core data models for PanGu."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Action(Enum):
    """Trading action direction."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStatus(Enum):
    """Signal lifecycle status."""

    NEW_ENTRY = "new_entry"
    SUSTAINED = "sustained"
    EXIT = "exit"


class Region(Enum):
    """News / data region tag."""

    DOMESTIC = "domestic"
    GLOBAL = "global"


@dataclass
class StockMeta:
    """Metadata for a single stock symbol."""

    name: str = ""
    sector: str = ""


class NewsCategory(str, Enum):
    """Distinguishes news articles from company announcements."""

    NEWS = "news"
    ANNOUNCEMENT = "announcement"


@dataclass
class NewsItem:
    """A single news item from any data source.

    Fields follow PRD §4.1.3.
    """

    timestamp: datetime
    title: str
    content: str
    source: str
    region: Region
    symbols: list[str] = field(default_factory=list)
    sentiment: float | None = None
    category: NewsCategory = NewsCategory.NEWS


@dataclass
class TradeSignal:
    """A trading signal produced by any strategy engine.

    Fields follow PRD §4.3.1.
    """

    timestamp: datetime
    symbol: str
    name: str
    action: Action
    signal_status: SignalStatus
    days_in_top_n: int
    price: float
    confidence: float
    source: str  # "factor" | "llm_event" | "merged"
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    factor_score: float | None = None
    prev_factor_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtest parameters (Phase 2).

    Fields follow PRD §7.
    """

    start_date: str
    end_date: str
    initial_cash: float = 100_000
    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.001
    max_position_pct: float = 0.2
