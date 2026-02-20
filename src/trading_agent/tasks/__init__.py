"""Task functions extracted from TradingScheduler."""

from trading_agent.tasks.t1_global_market import sync_global_market
from trading_agent.tasks.t2_news import poll_news
from trading_agent.tasks.t3_domestic_market import sync_domestic_market
from trading_agent.tasks.t4_signals import generate_signals
from trading_agent.tasks.t5_reference import sync_reference_data

__all__ = [
    "sync_global_market",
    "poll_news",
    "sync_domestic_market",
    "generate_signals",
    "sync_reference_data",
]
