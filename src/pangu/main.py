"""PanGu main entry point — M5.1.

Initializes all components and starts the APScheduler-based
TradingScheduler. Supports two modes:
  - ``--once``: run all tasks once and exit (for manual / first-run)
  - default: start scheduler and run indefinitely
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

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

from pangu.config import Settings, load_settings
from pangu.data.fundamental import AkShareFundamentalProvider
from pangu.data.market import AkShareMarketDataProvider
from pangu.data.news import AkShareNewsDataProvider
from pangu.data.stock_pool import StockPoolManager
from pangu.data.storage import Database
from pangu.factor.fundamental import FundamentalFactorEngine
from pangu.factor.macro import MacroFactorEngine
from pangu.factor.technical import PandasTAFactorEngine
from pangu.notification import NotificationManager
from pangu.notification.feishu import FeishuNotifier
from pangu.scheduler import Components, TradingScheduler
from pangu.strategy.factor import MultiFactorStrategy
from pangu.strategy.llm import LLMClient
from pangu.strategy.llm import LLMJudgeEngineImpl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_components() -> tuple[Components, str, Settings]:
    """Initialize all real components from config. Returns (components, timezone, settings)."""
    settings = load_settings()
    tz = settings.system.get("timezone", "Asia/Shanghai")

    # Log level
    log_level = settings.system.get("log_level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logger.info("Config loaded: timezone=%s, log_level=%s", tz, log_level)

    # SQLite
    db_path = settings.system.get("db_path", "data/pangu.db")
    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    db = Database(db_path)
    db.init_tables()
    logger.info("SQLite initialized: %s", db_path)

    # Data providers
    market = AkShareMarketDataProvider(storage=db)
    news = AkShareNewsDataProvider(storage=db)
    fundamental = AkShareFundamentalProvider(storage=db)

    sys_cfg = settings.system
    pool_cfg = settings.stock_pool
    stock_pool = StockPoolManager(
        watchlist_path=sys_cfg.get("watchlist_path", "config/watchlist.yaml"),
        storage=db,
        market_provider=market,
        news_provider=news,
        fundamental_provider=fundamental,
        min_listing_days=pool_cfg.get("min_listing_days", 60),
    )

    # Sector mapping
    mapping_path = Path(sys_cfg.get("global_market_mapping_path", "config/global_market_mapping.yaml"))
    sector_mapping: list[dict] = []
    if mapping_path.exists():
        with open(mapping_path) as f:
            raw = yaml.safe_load(f) or {}
        sector_mapping = raw.get("mappings", [])

    # Factor engines
    tech_engine = PandasTAFactorEngine()
    fund_engine = FundamentalFactorEngine()
    macro_engine = MacroFactorEngine()

    # Strategy + LLM
    strategy_cfg = settings.strategy
    factor_strategy = MultiFactorStrategy(
        top_n=strategy_cfg.get("top_n", 10),
        buy_threshold=strategy_cfg.get("buy_threshold", 0.7),
        sell_threshold=strategy_cfg.get("sell_threshold", 0.3),
        risk_dampen_threshold=strategy_cfg.get("risk_dampen_threshold", -1.0),
        sector_mapping=sector_mapping,
    )

    llm_cfg = settings.llm
    llm_client = LLMClient(
        model=llm_cfg.get("provider", "azure/gpt-4o-mini"),
        temperature=llm_cfg.get("temperature", 0.1),
    )
    judge_engine = LLMJudgeEngineImpl(llm_client)

    # Notification
    notif_manager = NotificationManager()
    notif_cfg = settings.notification
    if notif_cfg.get("enabled", True):
        feishu_cfg = notif_cfg.get("feishu", {})
        app_id = feishu_cfg.get("app_id", "")
        app_secret = feishu_cfg.get("app_secret", "")

        if app_id and app_secret:
            open_id = os.environ.get("FEISHU_OPEN_ID", "")
            feishu_notifier = FeishuNotifier(
                app_id=app_id,
                app_secret=app_secret,
                open_id=open_id or None,
            )
            notif_manager.add_channel(feishu_notifier)
            logger.info("Feishu notifier initialized (open_id=%s)",
                        "set" if open_id else "not set")
        else:
            logger.warning("Feishu not configured, skipping")
    else:
        logger.info("Notification disabled by config")

    components = Components(
        db=db,
        market=market,
        news=news,
        fundamental=fundamental,
        stock_pool=stock_pool,
        tech_engine=tech_engine,
        fund_engine=fund_engine,
        macro_engine=macro_engine,
        factor_strategy=factor_strategy,
        judge_engine=judge_engine,
        notif_manager=notif_manager,
    )
    return components, tz, settings


async def _run_scheduler() -> None:
    """Start scheduler and block until SIGINT/SIGTERM."""
    components, tz, settings = _build_components()
    scheduler = TradingScheduler(components, timezone=tz, scheduler_cfg=settings.scheduler)

    # First run: sync calendar to ensure trading day checks work
    await scheduler.sync_reference_data()

    scheduler.start()
    logger.info("TradingScheduler running — press Ctrl+C to stop")

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    import sys
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    scheduler.shutdown()
    logger.info("Goodbye")


async def _run_once() -> None:
    """Run all tasks once and exit."""
    components, _tz, settings = _build_components()
    scheduler = TradingScheduler(components, timezone=_tz, scheduler_cfg=settings.scheduler)
    await scheduler.run_once()


async def _run_init() -> None:
    """First-time initialization: sync reference data + domestic market only."""
    components, _tz = _build_components()
    scheduler = TradingScheduler(components, timezone=_tz)
    logger.info("=== init: syncing reference data + domestic market ===")
    await scheduler.sync_reference_data()
    await scheduler.sync_domestic_market()
    logger.info("=== init complete ===")


def main() -> None:
    """Entry point for ``python -m pangu.main``."""
    if "--init" in sys.argv:
        logger.info("=== PanGu — init mode ===")
        asyncio.run(_run_init())
    elif "--once" in sys.argv:
        logger.info("=== PanGu — run once ===")
        asyncio.run(_run_once())
    else:
        logger.info("=== PanGu — scheduler mode ===")
        asyncio.run(_run_scheduler())


if __name__ == "__main__":
    main()
