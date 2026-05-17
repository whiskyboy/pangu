"""Task functions extracted from TradingScheduler.

Numbering follows logical dependency (data → model → signals), not
scheduling order:
  T1 sync_reference_data    — monthly (1st @ 06:00)
  T2 poll_news              — hourly
  T3 sync_domestic_market   — trading days @ 18:00
  T4 sync_global_market     — trading days @ 07:00
  T5 update_model           — monthly (1st @ 02:00)
  T6 generate_signals       — trading days @ 08:15
"""

from pangu.tasks.generate_signals import generate_signals
from pangu.tasks.poll_news import poll_news
from pangu.tasks.sync_domestic_market import sync_domestic_market
from pangu.tasks.sync_global_market import sync_global_market
from pangu.tasks.sync_reference_data import sync_reference_data
from pangu.tasks.update_model import update_model

__all__ = [
    "sync_reference_data",
    "poll_news",
    "sync_domestic_market",
    "sync_global_market",
    "update_model",
    "generate_signals",
]
