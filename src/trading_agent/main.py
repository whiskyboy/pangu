"""TradingAgent main entry point — M1 smoke test.

Runs a single pass through the core signal pipeline using Fake data providers
to verify all components are correctly wired. This is NOT the final scheduler
entry point — see PRD §4.6 and M5 (TradingScheduler) for the production
7-task scheduling architecture.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pandas as pd

from trading_agent.config import load_settings
from trading_agent.data.fundamental import FakeFundamentalDataProvider
from trading_agent.data.market import FakeMarketDataProvider
from trading_agent.data.news import FakeNewsDataProvider
from trading_agent.data.stock_pool import FakeStockPool
from trading_agent.models import Action
from trading_agent.notification import NotificationManager
from trading_agent.notification.feishu import FeishuNotifier
from trading_agent.strategy.anomaly_detector import FakeAnomalyDetector
from trading_agent.strategy.factor_strategy import FakeFactorStrategy
from trading_agent.strategy.llm_engine import FakeLLMEventEngine
from trading_agent.strategy.signal_merger import SimpleSignalMerger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_ACTION_EMOJI = {Action.BUY: "🟢", Action.SELL: "🔴", Action.HOLD: "⚪"}


def _print_signal_summary(signals: list) -> None:
    """Print a human-readable summary of merged signals."""
    if not signals:
        logger.info("No signals generated")
        return

    print("\n" + "=" * 60)
    print("📊 Signal Summary")
    print("=" * 60)
    for sig in signals:
        emoji = _ACTION_EMOJI.get(sig.action, "?")
        print(
            f"  {emoji} {sig.action.value:4s} | {sig.name} ({sig.symbol}) "
            f"| ¥{sig.price:,.2f} | conf={sig.confidence:.2f} "
            f"| {sig.reason}"
        )
    print("=" * 60 + "\n")


async def smoke_test() -> None:
    """Run one pass of the T6 (intraday signal) flow with Fake data.

    Validates: config → stock pool → market data → news → factor signals
    → event signals → anomaly detection → signal merger → notification.
    """
    logger.info("=== TradingAgent M1 Smoke Test ===")

    # 1. Load config
    settings = load_settings()
    logger.info("Config loaded: timezone=%s", settings.system.get("timezone"))

    # 2. Initialize Fake components
    market = FakeMarketDataProvider()
    _fundamental = FakeFundamentalDataProvider()  # used in T5, kept for completeness
    news = FakeNewsDataProvider()
    stock_pool = FakeStockPool()
    factor_strategy = FakeFactorStrategy()
    llm_engine = FakeLLMEventEngine()
    anomaly_detector = FakeAnomalyDetector()

    strategy_cfg = settings.strategy
    merger = SimpleSignalMerger(
        factor_weight=strategy_cfg.get("factor_weight", 0.6),
        event_weight=strategy_cfg.get("event_weight", 0.4),
        buy_threshold=strategy_cfg.get("buy_threshold", 0.7),
        sell_threshold=strategy_cfg.get("sell_threshold", 0.3),
    )

    # 3. Setup notification
    notif_cfg = settings.notification
    notif_manager = NotificationManager()

    feishu_cfg = notif_cfg.get("feishu", {})
    app_id = feishu_cfg.get("app_id", "")
    app_secret = feishu_cfg.get("app_secret", "")
    feishu_notifier: FeishuNotifier | None = None

    if app_id and app_secret:
        open_id = os.environ.get("FEISHU_OPEN_ID", "")
        feishu_notifier = FeishuNotifier(
            app_id=app_id,
            app_secret=app_secret,
            open_id=open_id or None,
        )
        notif_manager.add_channel(feishu_notifier)
        logger.info(
            "Feishu notifier initialized (open_id=%s)",
            "set" if open_id else "not set",
        )
    else:
        logger.warning("Feishu not configured, skipping")

    # 4. Get active stock pool
    pool = stock_pool.get_active_pool()
    logger.info("Active pool: %s (%d stocks)", pool, len(pool))

    # 5. Get market data
    bars_map: dict[str, pd.DataFrame] = {}
    for symbol in pool:
        bars_map[symbol] = market.get_daily_bars(symbol, "2025-01-01", "2025-06-30")
    logger.info("Daily bars loaded for %d symbols", len(bars_map))

    quotes = market.get_realtime_quote(pool)
    global_snapshot = market.get_global_snapshot()
    logger.info("Realtime quotes: %d rows, global snapshot: %d rows", len(quotes), len(global_snapshot))

    # 6. Get news
    domestic_news = news.get_latest_news()
    global_news = news.get_global_news()
    logger.info("News: %d domestic, %d global", len(domestic_news), len(global_news))

    # 7. Generate factor signals
    # Use first symbol's bars as representative data for FakeFactorStrategy
    sample_bars = next(iter(bars_map.values())) if bars_map else pd.DataFrame()
    factor_signals = factor_strategy.generate_signals(sample_bars, pool)
    logger.info("Factor signals: %d", len(factor_signals))

    # 8. Generate event signals
    all_news = domestic_news + global_news
    event_signals = await llm_engine.analyze_news(all_news, pool)
    logger.info("Event signals: %d", len(event_signals))

    # 9. Anomaly detection
    anomaly_signals = anomaly_detector.detect(quotes, global_snapshot)
    logger.info("Anomaly signals: %d", len(anomaly_signals))

    # 10. Signal merger
    news_availability = {
        symbol: any(symbol in n.symbols for n in all_news)
        for symbol in pool
    }
    merged_signals = merger.merge(
        factor_signals, event_signals, anomaly_signals, news_availability,
    )
    logger.info("Merged signals: %d", len(merged_signals))

    # 11. Print summary
    _print_signal_summary(merged_signals)

    # 12. Push notifications
    if merged_signals:
        for signal in merged_signals:
            result = await notif_manager.notify(signal)
            if result:
                logger.info("Push result for %s: %s", signal.symbol, result)
    else:
        logger.info("No signals to push")

    logger.info("=== Smoke test complete ===")


def main() -> None:
    """Entry point for ``python -m trading_agent.main``."""
    asyncio.run(smoke_test())


if __name__ == "__main__":
    main()
