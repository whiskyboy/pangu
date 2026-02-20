"""TradingAgent main entry point — M5.1.

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

from trading_agent.config import load_settings
from trading_agent.data.fundamental import AkShareFundamentalProvider
from trading_agent.data.market import AkShareMarketDataProvider
from trading_agent.data.news import AkShareNewsDataProvider
from trading_agent.data.stock_pool import StockPoolManager
from trading_agent.data.storage import Database
from trading_agent.factor.fundamental import FundamentalFactorEngine
from trading_agent.factor.macro import MacroFactorEngine
from trading_agent.factor.technical import PandasTAFactorEngine
from trading_agent.notification import NotificationManager
from trading_agent.notification.feishu import FeishuNotifier
from trading_agent.scheduler import Components, TradingScheduler
from trading_agent.strategy.factor_strategy import MultiFactorStrategy
from trading_agent.strategy.llm_engine import LLMClient, LLMJudgeEngineImpl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_components() -> tuple[Components, str]:
    """Initialize all real components from config. Returns (components, timezone)."""
    settings = load_settings()
    tz = settings.system.get("timezone", "Asia/Shanghai")
    logger.info("Config loaded: timezone=%s", tz)

    # SQLite
    db_path = settings.system.get("db_path", "data/trading_agent.db")
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

    stock_pool = StockPoolManager(
        watchlist_path="config/watchlist.yaml",
        storage=db,
        market_provider=market,
        news_provider=news,
        fundamental_provider=fundamental,
    )

    # Sector mapping
    mapping_path = Path("config/global_market_mapping.yaml")
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
        top_n=strategy_cfg.get("top_n", 3),
        risk_dampen_threshold=strategy_cfg.get("risk_dampen_threshold", -1.0),
        sector_mapping=sector_mapping,
    )

    llm_cfg = settings.llm
    llm_client = LLMClient(
        model=llm_cfg.get("provider", "azure/gpt-4o-mini"),
        fallback_models=llm_cfg.get("fallback_providers", []),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 800),
    )
    judge_engine = LLMJudgeEngineImpl(llm_client)

    # Notification
    notif_manager = NotificationManager()
    notif_cfg = settings.notification
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
    return components, tz


async def _run_scheduler() -> None:
    """Start scheduler and block until SIGINT/SIGTERM."""
    components, tz = _build_components()
    scheduler = TradingScheduler(components, timezone=tz)

    # First run: sync calendar to ensure trading day checks work
    await scheduler.sync_calendar()

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
    components, _tz = _build_components()
    scheduler = TradingScheduler(components, timezone=_tz)
    await scheduler.run_once()


def main() -> None:
    """Entry point for ``python -m trading_agent.main``."""
    once = "--once" in sys.argv
    if once:
        logger.info("=== TradingAgent — run once ===")
        asyncio.run(_run_once())
    else:
        logger.info("=== TradingAgent — scheduler mode ===")
        asyncio.run(_run_scheduler())


if __name__ == "__main__":
    main()
