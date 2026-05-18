"""PanGu main entry point.

Initializes all components and starts the APScheduler-based
TradingScheduler. Entry points:
  - ``_run_scheduler``: daemon mode (default)
  - ``_run_init``: one-time cold-start (backfill → factors → train → portfolio_state)
  - ``_run_signals``: manual trigger for T6 signal generation
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path


# Load .env if present (before config reads $ENV_VAR placeholders)
def load_env() -> None:
    """Load .env file into environment variables."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                if key:
                    os.environ.setdefault(key, val.strip())


load_env()

from pangu.config import Settings, load_settings  # noqa: E402
from pangu.data.fundamental import (  # noqa: E402
    AkShareFundamentalProvider,
    CompositeFundamentalProvider,
)
from pangu.data.market import BaoStockMarketDataProvider, CompositeMarketDataProvider  # noqa: E402
from pangu.data.news import AkShareNewsDataProvider  # noqa: E402
from pangu.data.stock_pool import IndexStockPool  # noqa: E402
from pangu.data.storage import Database  # noqa: E402
from pangu.factor.fundamental import FundamentalFactorEngine  # noqa: E402
from pangu.factor.technical import PandasTAFactorEngine  # noqa: E402
from pangu.notification import NotificationManager  # noqa: E402
from pangu.notification.feishu import FeishuNotifier  # noqa: E402
from pangu.portfolio import PortfolioState  # noqa: E402
from pangu.scheduler import Components, TradingScheduler  # noqa: E402
from pangu.strategy.llm import LLMClient, LLMJudgeEngineImpl  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_components() -> tuple[Components, str, Settings]:
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

    # Data providers — single primary, no fallback
    baostock_market = BaoStockMarketDataProvider()
    market = CompositeMarketDataProvider(storage=db, providers=[baostock_market])
    news = AkShareNewsDataProvider(storage=db)
    akshare_fund = AkShareFundamentalProvider()
    fundamental = CompositeFundamentalProvider(
        storage=db,
        providers=[akshare_fund],
    )

    pool_cfg = settings.stock_pool
    stock_pool = IndexStockPool(
        storage=db,
        indices=pool_cfg.get("indices"),
    )

    # Factor engines
    tech_engine = PandasTAFactorEngine()
    fund_engine = FundamentalFactorEngine()

    # LLM
    strategy_cfg = settings.strategy

    llm_cfg = settings.llm
    llm_client = LLMClient(
        model=llm_cfg.get("provider", "azure/gpt-4o-mini"),
        temperature=llm_cfg.get("temperature", 0.1),
    )
    judge_engine = LLMJudgeEngineImpl(llm_client)

    # ML scoring strategy — uses shared factory so T5 can rebuild after first training
    ml_cfg = settings.ml
    ml_enabled = bool(ml_cfg.get("enabled", False))
    from pangu.strategy.ml.ml_strategy import try_build_ml_strategy

    ml_strategy = try_build_ml_strategy(db, ml_cfg, strategy_cfg)
    if ml_enabled:
        if ml_strategy is not None:
            logger.info(
                "ML scoring enabled: model_dir=%s, window=%d, seeds=%d, top_n=%d, buy_pool=%d, sell_pool=%d, n_drop=%d",
                ml_cfg.get("model_dir", "models"),
                ml_strategy._scorer.window_id,
                ml_strategy._scorer.n_models,
                ml_strategy.top_n,
                ml_strategy.buy_candidate_size,
                ml_strategy.sell_candidate_size,
                ml_strategy.n_drop,
            )
        else:
            logger.warning(
                "ML enabled but no models found in %s; T6 will alert and skip on rebalance day. "
                "T5 will train and hot-swap on the next monthly run, or run `pangu run init` / `pangu train` first.",
                ml_cfg.get("model_dir", "models"),
            )

    # Portfolio state — virtual target portfolio JSON
    portfolio_cfg = settings.portfolio
    portfolio_state = PortfolioState(
        path=portfolio_cfg.get("state_path", "data/target_portfolio.json"),
    )

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
            logger.info("Feishu notifier initialized (open_id=%s)", "set" if open_id else "not set")
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
        judge_engine=judge_engine,
        notif_manager=notif_manager,
        ml_strategy=ml_strategy,
        ml_enabled=ml_enabled,
        portfolio_state=portfolio_state,
        initial_capital=float(settings.portfolio.get("initial_capital", 100_000.0)),
    )
    return components, tz, settings


async def _run_scheduler(*, initial_capital: float | None = None) -> None:
    """Start scheduler and block until SIGINT/SIGTERM."""
    components, tz, settings = build_components()
    if initial_capital is not None:
        components.initial_capital = initial_capital
        logger.info("Override initial_capital=%.2f", initial_capital)
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


async def _run_signals(*, initial_capital: float | None = None) -> None:
    """Manually trigger T6 signal generation (and T4 if today's snapshot is missing)."""
    components, tz, settings = build_components()
    if initial_capital is not None:
        components.initial_capital = initial_capital
        logger.info("Override initial_capital=%.2f", initial_capital)
    scheduler = TradingScheduler(components, timezone=tz, scheduler_cfg=settings.scheduler)
    logger.info("=== run signals: manual T6 trigger ===")
    await scheduler.run_signals()
    logger.info("=== run signals complete ===")


async def _run_init(*, force: bool = False) -> None:
    """First-time cold-start initialization.

    Executes the full bootstrap sequence. Each step is skipped if its
    output already looks fresh; use ``force=True`` to re-run everything.

      1. Sync trading calendar + index constituents (T1 idempotent)
      2. Backfill historical constituents + export pool YAML
      3. Backfill daily bars for the historical pool
      4. Backfill fundamentals (financial indicators + gross_margin + pub_dates)
      5. Compute Alpha158 factor panel → data/factors.parquet
      6. Train production model (single window, all history)
    """
    from datetime import datetime
    from pathlib import Path

    components, _tz, settings = build_components()
    db = components.db
    sys_cfg = settings.system
    init_start = sys_cfg.get("init_backfill_start", "2019-01-01")
    pool_cfg = settings.stock_pool
    pool_yaml_path = "config/backfill_stock_pool.yaml"

    logger.info("=== run init: cold-start initialization (force=%s) ===", force)
    logger.info("    backfill start date: %s", init_start)

    # ------------------------------------------------------------------
    # 1a. trading calendar (always runs — 5s, cheap, no skip)
    # ------------------------------------------------------------------
    logger.info("[1a/6] sync trading calendar...")
    components.stock_pool.sync_trading_calendar()

    # ------------------------------------------------------------------
    # 1b. current-date constituents + stock_profiles (cninfo, ~44min)
    #
    # Triggers when ANY of:
    #   * force flag
    #   * latest constituents snapshot is stale by >35 days (T1 monthly + 5d buffer)
    #   * stock_profiles is empty (upgrade-from-old-DB or first-time install)
    # ------------------------------------------------------------------
    from datetime import date as _date

    rows = db.load_all_index_constituents()
    latest_const_date = rows[0]["date"] if rows else None
    if latest_const_date:
        try:
            stale_days = (_date.today() - _date.fromisoformat(latest_const_date)).days
        except ValueError:
            stale_days = 10**6
    else:
        stale_days = 10**6
    profiles_count = db.count_stock_profiles()

    if force or stale_days > 35 or profiles_count == 0:
        logger.info(
            "[1b/6] sync index constituents + stock profiles (stale=%dd, profiles=%d)...",
            stale_days,
            profiles_count,
        )
        components.stock_pool.sync_index_constituents()
    else:
        logger.info(
            "[1b/6] ⏭ skipped: constituents fresh (%dd old), %d stock_profiles",
            stale_days,
            profiles_count,
        )

    # ------------------------------------------------------------------
    # 2. historical constituents backfill + pool YAML
    #
    # Triggers when YAML is missing OR older than 180 days (CSI 300/500
    # adjust every June and December — 180d ≈ one adjustment cycle).
    # ------------------------------------------------------------------
    import time

    pool_yaml = Path(pool_yaml_path)
    if pool_yaml.exists():
        yaml_age_days = (time.time() - pool_yaml.stat().st_mtime) / 86400.0
    else:
        yaml_age_days = float("inf")
    pool_yaml_stale = yaml_age_days > 180

    if force or pool_yaml_stale:
        logger.info(
            "[2/6] backfill historical constituents (semi-annual sampling, yaml_age=%.0fd)...",
            yaml_age_days,
        )
        from pangu.tz import today_str

        count, all_symbols = components.stock_pool.sync_historical_constituents(init_start, today_str())
        logger.info("    %d records, %d unique symbols", count, len(all_symbols))
        pool_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(pool_yaml, "w") as f:
            f.write("symbols:\n")
            for sym in sorted(all_symbols):
                f.write(f'- "{sym}"\n')
        logger.info("    exported pool to %s", pool_yaml)
    else:
        logger.info(
            "[2/6] ⏭ skipped: %s up to date (%.0fd old)",
            pool_yaml,
            yaml_age_days,
        )

    # Load pool for downstream steps
    import yaml

    with open(pool_yaml) as f:
        pool_data = yaml.safe_load(f) or {}
    pool = [str(s) for s in (pool_data.get("symbols") or [])]
    logger.info("    using %d-stock pool from %s", len(pool), pool_yaml)

    # ------------------------------------------------------------------
    # 3. backfill daily bars
    # ------------------------------------------------------------------
    from datetime import timedelta as _td

    from pangu.tz import today_str as _today_str

    latest_bar = db.get_latest_bar_date()
    bars_stale = (
        latest_bar is None or (datetime.now().date() - datetime.strptime(latest_bar, "%Y-%m-%d").date()).days > 7
    )
    # Pool-aware coverage: require ≥95% of pool symbols to have a bar within
    # the last 30 calendar days (handles holiday gaps). A global row count
    # could pass with massive coverage gaps in a partial backfill.
    coverage_since = (datetime.now().date() - _td(days=30)).strftime("%Y-%m-%d")
    covered_syms = db.count_symbols_with_bars(pool, coverage_since) if pool else 0
    pool_covered = len(pool) > 0 and covered_syms >= int(0.95 * len(pool))
    if force or bars_stale or not pool_covered:
        logger.info(
            "[3/6] backfill daily bars (%d stocks, this may take hours; coverage %d/%d)...",
            len(pool),
            covered_syms,
            len(pool),
        )
        today = _today_str()
        ok, fail = 0, 0
        for i, sym in enumerate(pool, 1):
            try:
                df = components.market.get_daily_bars(sym, init_start, today, force=force)
                if df is not None and not df.empty:
                    ok += 1
                else:
                    fail += 1
            except Exception:  # noqa: BLE001
                fail += 1
                logger.warning("    failed: %s", sym, exc_info=True)
            if i % 50 == 0 or i == len(pool):
                logger.info("    progress %d/%d (ok=%d fail=%d)", i, len(pool), ok, fail)
        # also backfill index bars
        for idx_code in pool_cfg.get("indices", ["000300"]):
            try:
                components.market.get_index_daily_bars(idx_code, init_start, today, force=force)
                logger.info("    index %s synced", idx_code)
            except Exception:  # noqa: BLE001
                logger.warning("    index %s failed", idx_code, exc_info=True)
    else:
        logger.info(
            "[3/6] ⏭ skipped: pool coverage %d/%d, latest=%s",
            covered_syms,
            len(pool),
            latest_bar,
        )

    # ------------------------------------------------------------------
    # 4. backfill fundamentals
    # ------------------------------------------------------------------
    latest_pub = db.get_latest_fundamentals_pub_date()
    pub_stale = (
        latest_pub is None or (datetime.now().date() - datetime.strptime(latest_pub, "%Y-%m-%d").date()).days > 120
    )
    # Pool-aware fundamentals coverage: ≥90% of pool symbols should have
    # at least one quarterly row in the past 200 days (one fiscal quarter
    # cycle + reporting lag).
    fund_since = (datetime.now().date() - _td(days=200)).strftime("%Y-%m-%d")
    covered_funds = db.count_symbols_with_fundamentals(pool, fund_since) if pool else 0
    pool_funds_covered = len(pool) > 0 and covered_funds >= int(0.90 * len(pool))
    if force or pub_stale or not pool_funds_covered:
        logger.info(
            "[4/6] backfill fundamentals + gross_margin + pub_dates (coverage %d/%d)...",
            covered_funds,
            len(pool),
        )
        today = _today_str()
        ok, fail = 0, 0
        for i, sym in enumerate(pool, 1):
            try:
                df = components.fundamental.get_financial_indicator(sym, start=init_start, force=force)
                if df is not None and not df.empty:
                    ok += 1
                else:
                    fail += 1
            except Exception:  # noqa: BLE001
                fail += 1
                logger.warning("    fundamentals failed: %s", sym, exc_info=True)
            if i % 50 == 0 or i == len(pool):
                logger.info("    progress %d/%d (ok=%d fail=%d)", i, len(pool), ok, fail)
        components.fundamental.refresh_gross_margin(init_start, today)
        components.fundamental.refresh_pub_dates(init_start, today)
    else:
        logger.info(
            "[4/6] ⏭ skipped: fundamentals pool coverage %d/%d, latest pub_date=%s",
            covered_funds,
            len(pool),
            latest_pub,
        )

    # ------------------------------------------------------------------
    # 5. compute Alpha158 factors
    # ------------------------------------------------------------------
    factors_path = Path("data/factors.parquet")
    latest_bar = db.get_latest_bar_date()  # refresh
    fresh_factors = (
        factors_path.exists()
        and latest_bar is not None
        and factors_path.stat().st_mtime > datetime.strptime(latest_bar, "%Y-%m-%d").timestamp()
    )
    if force or not fresh_factors:
        logger.info("[5/6] compute Alpha158 factor panel...")
        from pangu.factor.alpha158 import Alpha158Engine

        engine = Alpha158Engine()
        end_date = latest_bar or _today_str()
        bars: list = []
        funds: list = []
        for sym in pool:
            b = db.load_daily_bars(sym, init_start, end_date)
            if b is not None and not b.empty:
                bars.append(b)
            f = db.load_fundamentals_filled(sym, init_start, end_date)
            if f is not None and not f.empty:
                funds.append(f)
        import pandas as pd

        all_bars = pd.concat(bars, ignore_index=True) if bars else pd.DataFrame()
        all_funds = pd.concat(funds, ignore_index=True) if funds else pd.DataFrame()
        panel = engine.compute(all_bars, all_funds)
        factors_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(factors_path)
        logger.info(
            "    factors written: %s (%s rows × %d cols)",
            factors_path,
            f"{len(panel):,}",
            panel.shape[1],
        )
    else:
        logger.info("[5/6] ⏭ skipped: %s up to date", factors_path)

    # ------------------------------------------------------------------
    # 6. train production model
    # ------------------------------------------------------------------
    ml_cfg = settings.ml
    model_dir = Path(ml_cfg.get("model_dir", "models"))
    fresh_model = False
    age_days = float("inf")
    if model_dir.exists():
        model_files = list(model_dir.glob("wf_window_*_seed*.txt"))
        if model_files:
            newest = max(model_files, key=lambda p: p.stat().st_mtime)
            age_days = (datetime.now().timestamp() - newest.stat().st_mtime) / 86400.0
            fresh_model = age_days < 30
    if force or not fresh_model:
        logger.info("[6/6] train production model (single window, all history)...")
        from pangu.ml.model import train as _train

        _train(
            storage=db,
            factors_path=str(factors_path) if factors_path.exists() else None,
            model_dir=str(model_dir),
            first_train_start=ml_cfg.get("first_train_start", "2020-01-01"),
            val_months=ml_cfg.get("val_months", 3),
            time_decay_halflife=ml_cfg.get("time_decay_halflife", 120),
            n_seeds=ml_cfg.get("n_seeds", 5),
        )
    else:
        logger.info("[6/6] ⏭ skipped: model in %s is %.1f days old", model_dir, age_days)

    logger.info("=== run init complete ===")


def main() -> None:
    """Entry point for ``python -m pangu.main``."""
    if "--init" in sys.argv:
        logger.info("=== PanGu — init mode ===")
        asyncio.run(_run_init(force="--force" in sys.argv))
    elif "--signals" in sys.argv:
        logger.info("=== PanGu — signals mode ===")
        asyncio.run(_run_signals())
    else:
        logger.info("=== PanGu — scheduler mode ===")
        asyncio.run(_run_scheduler())


if __name__ == "__main__":
    main()
