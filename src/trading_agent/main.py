"""TradingAgent main entry point — M4.6 integration.

Runs a single pass through the core signal pipeline using real AkShare
data providers, real factor engines, and LLM comprehensive judge engine.
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
from trading_agent.models import Action, TradeSignal
from trading_agent.notification import NotificationManager
from trading_agent.notification.feishu import FeishuNotifier
from trading_agent.strategy.factor_strategy import MultiFactorStrategy
from trading_agent.strategy.llm_engine import LLMClient, LLMJudgeEngineImpl
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
          multi-factor strategy, LLM comprehensive judge engine
    """
    logger.info("=== TradingAgent M4.6 Integration ===")

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

    # 6. Real strategy + LLM judge engine
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

    # 13. Fetch telegraph (market-wide news for LLM evidence package)
    telegraph = news.get_latest_news(limit=50)
    logger.info("Telegraph: %d items", len(telegraph))

    # 14. Generate factor pool (real engines)
    prev_pool = db.load_factor_pool_latest()
    pool_df, factor_signals = factor_strategy.generate_signals(
        tech_df, fund_df, macro_factors,
        prev_pool=prev_pool,
    )
    logger.info("Factor pool:\n%s", pool_df.to_string() if not pool_df.empty else "(empty)")
    logger.info("Factor signals (for fallback): %d", len(factor_signals))

    # Save factor pool
    if not pool_df.empty:
        db.save_factor_pool(today, pool_df)
        logger.info("Factor pool saved to SQLite (%d rows)", len(pool_df))

    # 15. Load stock name + sector map from watchlist
    wl_path = Path("config/watchlist.yaml")
    name_map: dict[str, str] = {}
    sector_map: dict[str, str] = {}
    if wl_path.exists():
        wl_data = yaml.safe_load(wl_path.read_text()) or {}
        for item in wl_data.get("watchlist", []):
            sym = item["symbol"]
            name_map[sym] = item.get("name", sym)
            sector_map[sym] = item.get("sector", "")

    # 16. Build factor details matrix for LLM prompt
    factor_matrix = factor_strategy._build_factor_matrix(
        pool, tech_df, fund_df, macro_factors,
        {s: sector_map.get(s, "") for s in pool},
    )

    # 17. Build LLM candidates (factor pool + watchlist, deduplicated)
    candidate_symbols = list(dict.fromkeys(
        list(pool_df["symbol"]) + watchlist if not pool_df.empty else watchlist
    ))

    candidates: list[dict] = []
    for sym in candidate_symbols:
        # Factor data
        row = pool_df[pool_df["symbol"] == sym] if not pool_df.empty else pd.DataFrame()
        f_score = float(row["score"].iloc[0]) if not row.empty else 0.5
        f_rank = int(row["rank"].iloc[0]) if not row.empty else len(candidate_symbols)
        f_details = (
            factor_matrix.loc[sym].to_dict()
            if sym in factor_matrix.index else {}
        )

        # Per-stock news + announcements
        try:
            s_news = news.get_stock_news(sym, limit=10)
        except Exception:  # noqa: BLE001
            logger.warning("  %s: stock news failed", sym, exc_info=True)
            s_news = []
        try:
            s_anns = news.get_announcements(sym, limit=5)
        except Exception:  # noqa: BLE001
            logger.warning("  %s: announcements failed", sym, exc_info=True)
            s_anns = []

        # Price (from latest bar)
        bars = tech_df.get(sym)
        price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else 0.0

        candidates.append({
            "symbol": sym,
            "name": name_map.get(sym, sym),
            "factor_score": f_score,
            "factor_rank": f_rank,
            "factor_details": f_details,
            "stock_news": s_news,
            "announcements": s_anns,
            "price": price,
        })

    logger.info("LLM candidates: %d stocks", len(candidates))

    # 18. LLM comprehensive judge (per-stock)
    signals: list[TradeSignal] = await judge_engine.judge_pool(
        candidates, telegraph=telegraph, global_market=global_snapshot,
    )
    logger.info("LLM signals: %d (call_count=%d)", len(signals), llm_client.call_count)
    _print_signal_summary(signals)

    # 19. Save signals + push notifications
    for signal in signals:
        db.save_trade_signal(signal)

    # Push: watchlist stocks always pushed (incl. HOLD); sort BUY > HOLD > SELL, then by factor_score desc
    _ACTION_ORDER = {Action.BUY: 0, Action.HOLD: 1, Action.SELL: 2}
    watchlist_set = set(watchlist)
    to_push = [s for s in signals if s.action != Action.HOLD or s.symbol in watchlist_set]
    to_push.sort(key=lambda s: (_ACTION_ORDER.get(s.action, 9), -(s.factor_score or 0)))

    if to_push:
        for signal in to_push:
            result = await notif_manager.notify(signal)
            if result:
                logger.info("Push result for %s: %s", signal.symbol, result)
    else:
        logger.info("No signals to push")

    # 20. DB summary
    _print_db_summary(db)

    logger.info("=== M4.6 Integration complete ===")


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
