"""TradingAgent main entry point — M3.6 integration verification.

Runs a single pass through the core signal pipeline using real AkShare
data providers for market, news, fundamental data, and real factor/strategy
engines.  LLM event engine remains Fake until M4.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pandas as pd
import yaml

# Load .env if present (before config reads $ENV_VAR placeholders)
_env_path = Path(".env")
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            if key:
                os.environ.setdefault(key, val.strip())

from trading_agent.config import load_settings
from trading_agent.data.fundamental import AkShareFundamentalProvider
from trading_agent.data.market import AkShareMarketDataProvider
from trading_agent.data.news import AkShareNewsDataProvider
from trading_agent.data.stock_pool import StockPoolManager
from trading_agent.data.storage import Database
from trading_agent.factor.fundamental import FundamentalFactorEngine
from trading_agent.factor.macro import MacroFactorEngine
from trading_agent.factor.technical import PandasTAFactorEngine
from trading_agent.models import Action
from trading_agent.notification import NotificationManager
from trading_agent.notification.feishu import FeishuNotifier
from trading_agent.strategy.anomaly_detector import PriceVolumeAnomalyDetector
from trading_agent.strategy.factor_strategy import MultiFactorStrategy
from trading_agent.strategy.llm_engine import FakeLLMEventEngine
from trading_agent.strategy.signal_merger import SimpleSignalMerger
from trading_agent.tz import today_str

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
    """Run one pass of the data + signal pipeline with real engines.

    Real: market, news, fundamental, technical/fundamental/macro factors,
          multi-factor strategy, anomaly detector
    Fake: LLM event engine (M4)
    """
    logger.info("=== TradingAgent M3.6 Integration Verification ===")

    # 1. Load config
    settings = load_settings()
    logger.info("Config loaded: timezone=%s", settings.system.get("timezone"))

    # 2. Initialize SQLite
    db_path = settings.system.get("db_path", "data/trading_agent.db")
    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    db = Database(db_path)
    db.init_tables()
    logger.info("SQLite initialized: %s", db_path)

    # 3. Initialize real data providers
    market = AkShareMarketDataProvider(storage=db)
    news = AkShareNewsDataProvider(storage=db)
    fundamental = AkShareFundamentalProvider(storage=db)

    stock_pool = StockPoolManager(
        watchlist_path="config/watchlist.yaml",
        storage=db,
        market_provider=market,
        news_provider=news,
        fundamental_provider=fundamental,
    )

    # 4. Load sector mapping from YAML
    mapping_path = Path("config/global_market_mapping.yaml")
    sector_mapping: list[dict] = []
    if mapping_path.exists():
        with open(mapping_path) as f:
            raw = yaml.safe_load(f) or {}
        sector_mapping = raw.get("mappings", [])
    logger.info("Sector mapping: %d entries", len(sector_mapping))

    # 5. Real factor engines
    tech_engine = PandasTAFactorEngine()
    fund_engine = FundamentalFactorEngine()
    macro_engine = MacroFactorEngine()

    # 6. Real strategy + anomaly detector
    strategy_cfg = settings.strategy
    factor_strategy = MultiFactorStrategy(
        top_n=strategy_cfg.get("top_n", 3),
        risk_dampen_threshold=strategy_cfg.get("risk_dampen_threshold", -1.0),
        sector_mapping=sector_mapping,
    )

    anomaly_detector = PriceVolumeAnomalyDetector(
        sector_mapping=sector_mapping,
    )

    # Fake LLM (M4)
    llm_engine = FakeLLMEventEngine()

    merger = SimpleSignalMerger(
        factor_weight=strategy_cfg.get("factor_weight", 0.6),
        event_weight=strategy_cfg.get("event_weight", 0.4),
        buy_threshold=strategy_cfg.get("buy_threshold", 0.7),
        sell_threshold=strategy_cfg.get("sell_threshold", 0.3),
    )

    # 7. Setup notification
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

    # 8. Sync trading calendar
    cal_count = stock_pool.sync_trading_calendar()
    logger.info("Trading calendar: %d new dates synced", cal_count)

    # 9. Get active stock pool (with ST/IPO filtering)
    watchlist = stock_pool.get_watchlist()
    logger.info("Watchlist: %s (%d stocks)", watchlist, len(watchlist))
    pool = stock_pool.get_active_pool()
    logger.info("Active pool (after filtering): %s (%d stocks)", pool, len(pool))

    # 10. Fetch market data + compute technical factors
    today = today_str()
    tech_df: dict[str, pd.DataFrame] = {}
    for symbol in pool:
        try:
            bars = market.get_daily_bars(symbol, "2025-06-01", today)
            if bars is not None and not bars.empty:
                enriched = tech_engine.compute(bars)
                tech_df[symbol] = enriched
                logger.info("  %s: %d bars, %d factors", symbol, len(bars),
                            len(enriched.columns) - len(bars.columns))
            else:
                logger.warning("  %s: no bars returned", symbol)
        except Exception:  # noqa: BLE001
            logger.warning("  %s: daily bars / tech failed", symbol, exc_info=True)
    logger.info("Technical factors computed for %d/%d symbols", len(tech_df), len(pool))

    # 11. Fetch fundamentals + compute fundamental factors
    fund_rows: list[dict] = []
    for symbol in pool:
        try:
            val = fundamental.get_valuation(symbol)
            val["symbol"] = symbol
            fund_rows.append(val)
            logger.info("  %s: PE=%.1f, PB=%.1f, 市值=%.0f亿",
                        symbol,
                        val.get("pe_ttm", 0),
                        val.get("pb", 0),
                        val.get("market_cap", 0) / 1e8)
        except Exception:  # noqa: BLE001
            logger.warning("  %s: fundamentals failed", symbol, exc_info=True)

    fund_raw = pd.DataFrame(fund_rows)
    if not fund_raw.empty and "symbol" in fund_raw.columns:
        fund_raw = fund_raw.set_index("symbol")
    fund_df = fund_engine.compute(fund_raw)
    logger.info("Fundamental factors: %d symbols, columns=%s",
                len(fund_df), list(fund_df.columns))

    # 12. Fetch global snapshot + compute macro factors
    try:
        global_snapshot = market.get_global_snapshot()
        logger.info("Global snapshot: %d rows", len(global_snapshot))
    except Exception:  # noqa: BLE001
        logger.warning("Global snapshot failed", exc_info=True)
        global_snapshot = pd.DataFrame()

    macro_factors = macro_engine.compute(global_snapshot)
    logger.info("Macro factors: %s", {k: round(v, 4) for k, v in macro_factors.items()})

    # 13. Fetch news
    domestic_news = news.get_latest_news(limit=10)
    logger.info("Latest news: %d items", len(domestic_news))

    for symbol in pool[:2]:
        try:
            stock_news = news.get_stock_news(symbol, limit=5)
            logger.info("  %s stock news: %d items", symbol, len(stock_news))
        except Exception:  # noqa: BLE001
            logger.warning("  %s: stock news failed", symbol, exc_info=True)

    # 14. Fetch announcements
    for symbol in pool[:2]:
        try:
            anns = news.get_announcements(symbol, limit=5)
            logger.info("  %s announcements: %d items", symbol, len(anns))
        except Exception:  # noqa: BLE001
            logger.warning("  %s: announcements failed", symbol, exc_info=True)

    # 15. Generate factor signals (real engines)
    prev_pool = db.load_factor_pool_latest()
    pool_df, factor_signals = factor_strategy.generate_signals(
        tech_df, fund_df, macro_factors,
        prev_pool=prev_pool,
    )
    logger.info("Factor pool:\n%s", pool_df.to_string() if not pool_df.empty else "(empty)")
    logger.info("Factor signals: %d", len(factor_signals))
    for sig in factor_signals:
        logger.info("  %s %s %s conf=%.2f %s",
                    sig.action.value, sig.symbol, sig.name,
                    sig.confidence, sig.reason)

    # Save factor pool
    if not pool_df.empty:
        db.save_factor_pool(today, pool_df)
        logger.info("Factor pool saved to SQLite (%d rows)", len(pool_df))

    # 16. LLM event signals (fake)
    all_news = domestic_news
    event_signals = await llm_engine.analyze_news(all_news, pool)
    logger.info("Event signals (fake): %d", len(event_signals))

    # 17. Anomaly detection (real)
    try:
        quotes = market.get_realtime_quote(pool) if pool else pd.DataFrame()
    except Exception:  # noqa: BLE001
        logger.warning("Realtime quotes failed", exc_info=True)
        quotes = pd.DataFrame()
    anomaly_signals = anomaly_detector.detect(quotes, global_snapshot)
    logger.info("Anomaly signals: %d", len(anomaly_signals))
    for sig in anomaly_signals:
        logger.info("  %s %s conf=%.2f %s",
                    sig.action.value, sig.symbol, sig.confidence, sig.reason)

    # 18. Signal merger
    news_availability = {
        symbol: any(symbol in n.symbols for n in all_news)
        for symbol in pool
    }
    merged_signals = merger.merge(
        factor_signals, event_signals, anomaly_signals, news_availability,
    )
    logger.info("Merged signals: %d", len(merged_signals))
    _print_signal_summary(merged_signals)

    # 19. Push notifications
    if merged_signals:
        for signal in merged_signals:
            result = await notif_manager.notify(signal)
            if result:
                logger.info("Push result for %s: %s", signal.symbol, result)
    else:
        logger.info("No signals to push")

    # 20. DB summary
    _print_db_summary(db)

    logger.info("=== M3.6 Integration verification complete ===")


def _print_db_summary(db: Database) -> None:
    """Print row counts for all tables."""
    tables = [
        "daily_bars", "news_items", "fundamentals",
        "trade_signals", "trading_calendar", "global_snapshots",
        "data_sync_log", "factor_pool",
    ]
    print("\n" + "=" * 40)
    print("📦 SQLite Summary")
    print("=" * 40)
    for table in tables:
        try:
            row = db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
            print(f"  {table:25s} {row[0]:>8,} rows")
        except Exception:  # noqa: BLE001
            print(f"  {table:25s}    error")
    print("=" * 40 + "\n")


def main() -> None:
    """Entry point for ``python -m trading_agent.main``."""
    asyncio.run(smoke_test())


if __name__ == "__main__":
    main()
