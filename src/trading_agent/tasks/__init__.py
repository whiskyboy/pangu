"""Task functions extracted from TradingScheduler."""

from trading_agent.tasks.sync_global_market import sync_global_market
from trading_agent.tasks.poll_news import poll_news
from trading_agent.tasks.sync_domestic_market import sync_domestic_market
from trading_agent.tasks.generate_signals import generate_signals
from trading_agent.tasks.sync_reference_data import sync_reference_data

__all__ = [
    "sync_global_market",
    "poll_news",
    "sync_domestic_market",
    "generate_signals",
    "sync_reference_data",
]
