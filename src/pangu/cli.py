"""PanGu CLI — command-line interface for stock pool management and task control."""

from __future__ import annotations

import asyncio
import sys

import click


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """PanGu — A-share quantitative trading signal system."""


# ---------------------------------------------------------------------------
# pool commands
# ---------------------------------------------------------------------------

@main.group()
def pool() -> None:
    """Manage the watchlist stock pool."""


@pool.command("list")
def pool_list() -> None:
    """List all stocks in the watchlist."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    entries = c.stock_pool._entries  # noqa: SLF001
    if not entries:
        click.echo("Watchlist is empty.")
        return

    click.echo(f"{'Symbol':<10} {'Name':<12} {'Sector'}")
    click.echo("─" * 40)
    for e in entries:
        sym = e.get("symbol", "")
        name = e.get("name", "")
        sector = e.get("sector", "")
        click.echo(f"{sym:<10} {name:<12} {sector}")
    click.echo(f"\nTotal: {len(entries)} stocks")


@pool.command("add")
@click.argument("input_str")
@click.option("--name", default="", help="Stock name (auto-resolved if omitted)")
@click.option("--sector", default="", help="Sector (auto-resolved if omitted)")
def pool_add(input_str: str, name: str, sector: str) -> None:
    """Add a stock to the watchlist (by symbol or name)."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    # If input is a name, show candidates for confirmation
    if not (input_str.isdigit() and len(input_str) == 6):
        candidates = c.stock_pool.resolve_stock(input_str)
        if not candidates:
            click.echo(f"❌ Could not resolve '{input_str}' to any stock.")
            sys.exit(1)
        if len(candidates) > 1:
            click.echo(f"Found {len(candidates)} matches for '{input_str}':")
            for i, cd in enumerate(candidates, 1):
                click.echo(f"  {i}. {cd['symbol']} {cd['name']} ({cd['sector']})")
            choice = click.prompt("Select", type=int, default=1)
            if choice < 1 or choice > len(candidates):
                click.echo("Invalid choice.")
                sys.exit(1)
            selected = candidates[choice - 1]
            name = name or selected["name"]
            sector = sector or selected["sector"]
            input_str = selected["symbol"]

    result = c.stock_pool.add_to_watchlist(input_str, name=name, sector=sector)
    if result:
        click.echo(f"✅ Added {result} to watchlist")
    else:
        click.echo(f"❌ Failed to add '{input_str}'")
        sys.exit(1)


@pool.command("remove")
@click.argument("input_str")
def pool_remove(input_str: str) -> None:
    """Remove a stock from the watchlist (by symbol or name)."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    result = c.stock_pool.remove_from_watchlist(input_str)
    if result:
        click.echo(f"✅ Removed {result} from watchlist")
    else:
        click.echo(f"❌ '{input_str}' not found in watchlist")
        sys.exit(1)


# ---------------------------------------------------------------------------
# run commands
# ---------------------------------------------------------------------------

@main.group()
def run() -> None:
    """Run trading tasks."""


@run.command("init")
def run_init() -> None:
    """First-time initialization: sync reference data + domestic market."""
    from pangu.main import _run_init
    asyncio.run(_run_init())


@run.command("once")
def run_once() -> None:
    """Run all tasks once and exit."""
    from pangu.main import _run_once
    asyncio.run(_run_once())


@run.command("start")
def run_start() -> None:
    """Start the scheduler (daemon mode)."""
    from pangu.main import _run_scheduler
    asyncio.run(_run_scheduler())


# ---------------------------------------------------------------------------
# backfill commands
# ---------------------------------------------------------------------------

@main.group()
def backfill() -> None:
    """Backfill historical data from BaoStock."""


def _load_pool_yaml(path: str) -> list[str] | None:
    """Load stock list from YAML file. Returns None on error."""
    import yaml
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        pool = data.get("stocks", [])
        if not pool:
            click.echo(f"Warning: No stocks found in {path}", err=True)
            return None
        return [str(s) for s in pool]
    except FileNotFoundError:
        click.echo(f"Error: File not found: {path}", err=True)
        return None
    except yaml.YAMLError as e:
        click.echo(f"Error: Invalid YAML in {path}: {e}", err=True)
        return None


@backfill.command("constituents")
@click.option("--start", default="2019-01-01", help="Start date for semi-annual sampling")
@click.option("--end", default=None, help="End date (default: today)")
def backfill_constituents(start: str, end: str | None) -> None:
    """Backfill historical index constituents (CSI300 + CSI500) semi-annually."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    count, all_symbols = c.stock_pool.sync_historical_constituents(start, end)
    click.echo(f"\n✅ Saved {count} constituent records, {len(all_symbols)} unique stocks")

    # Export unique symbols to YAML
    yaml_path = "config/backfill_stock_pool.yaml"
    with open(yaml_path, "w") as f:
        f.write("stocks:\n")
        for sym in sorted(all_symbols):
            f.write(f'- "{sym}"\n')
    click.echo(f"📄 Exported {len(all_symbols)} symbols to {yaml_path}")


@backfill.command("bars")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--force", is_flag=True, help="Bypass cache, fetch full range from providers")
@click.option("--pool", "pool_file", default=None, help="YAML file with stock list (default: current constituents)")
def backfill_bars(start: str, force: bool, pool_file: str | None) -> None:
    """Backfill daily OHLCV + PE/PB for all pool stocks."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    from pangu.tz import today_str

    if pool_file:
        pool = _load_pool_yaml(pool_file)
        if pool is None:
            return
        click.echo(f"Using pool from {pool_file}: {len(pool)} stocks")
    else:
        pool = c.stock_pool.get_all_symbols()
    today = today_str()
    total = len(pool)
    ok, fail = 0, 0

    mode = " (FORCE)" if force else ""
    click.echo(f"Backfilling {total} stocks from {start} to {today}{mode}...")
    for i, symbol in enumerate(pool, 1):
        try:
            df = c.market.get_daily_bars(symbol, start, today, force=force)
            if df is not None and not df.empty:
                ok += 1
            else:
                fail += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            click.echo(f"  ✗ {symbol}: {e}")
        if i % 50 == 0 or i == total:
            click.echo(f"  [{i}/{total}] ok={ok} fail={fail}")

    click.echo(f"\n✅ Backfill bars done: {ok} ok, {fail} failed")


@backfill.command("index")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--symbol", default="000300", help="Index code (default: 000300)")
@click.option("--force", is_flag=True, help="Bypass cache, fetch full range")
def backfill_index(start: str, symbol: str, force: bool) -> None:
    """Backfill index daily bars (default: CSI300)."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    from pangu.tz import today_str

    today = today_str()
    click.echo(f"Backfilling index {symbol} from {start} to {today}{'(FORCE)' if force else ''}...")
    df = c.market.get_index_daily_bars(symbol, start, today, force=force)
    click.echo(f"✅ Index {symbol}: {len(df)} bars loaded")


@backfill.command("fundamentals")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD, e.g. 2022-10-01)")
@click.option("--force", is_flag=True, help="Bypass cache, re-fetch from providers")
@click.option("--pool", "pool_file", default=None, help="YAML file with stock list")
def backfill_fundamentals(start: str, force: bool, pool_file: str | None) -> None:
    """Backfill quarterly financial indicators."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    if pool_file:
        pool = _load_pool_yaml(pool_file)
        if pool is None:
            return
        click.echo(f"Using pool from {pool_file}: {len(pool)} stocks")
    else:
        pool = c.stock_pool.get_all_symbols()
    total = len(pool)
    ok, fail = 0, 0
    consecutive_fails = 0

    click.echo(f"Backfilling fundamentals for {total} stocks from {start}{'(FORCE)' if force else ''}...")
    for i, symbol in enumerate(pool, 1):
        try:
            df = c.fundamental.get_financial_indicator(symbol, start=start, force=force)
            if df is not None and not df.empty:
                ok += 1
                consecutive_fails = 0
            else:
                fail += 1
                consecutive_fails += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            consecutive_fails += 1
            click.echo(f"  ✗ {symbol}: {e}")

        # Network recovery: pause on consecutive failures
        if consecutive_fails >= 5:
            import time
            click.echo(f"  ⚠️ {consecutive_fails} consecutive failures, waiting 60s...")
            time.sleep(60)
            consecutive_fails = 0

        if i % 50 == 0 or i == total:
            click.echo(f"  [{i}/{total}] ok={ok} fail={fail}")

    click.echo(f"\n✅ Backfill fundamentals done: {ok} ok, {fail} failed")




# ---------------------------------------------------------------------------
# backtest command
# ---------------------------------------------------------------------------

@main.command("backtest")
@click.option("--strategy", type=click.Choice(["baseline"]), default="baseline",
              help="Strategy to backtest")
@click.option("--start", default="2022-01-01", help="Start date")
@click.option("--end", default="2025-06-30", help="End date")
@click.option("--top-n", default=10, type=int, help="TopN stocks to hold")
@click.option("--capital", default=1_000_000, type=float, help="Initial capital (CNY)")
def backtest_cmd(strategy: str, start: str, end: str, top_n: int, capital: float) -> None:
    """Run local backtest for a given strategy."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    import pandas as pd
    from pangu.backtest.engine import BacktestEngine, make_universe_fn

    storage = c.market._storage

    # Use historical constituent union for data loading (survivorship-bias-free)
    pool = storage.load_constituents_union(start, end)
    if not pool:
        click.echo("WARNING: No historical constituents found, falling back to current pool")
        pool = c.stock_pool.get_all_symbols()

    click.echo(f"Loading data for {len(pool)} stocks (historical union), {start} ~ {end}...")

    # Load price data with 120-day warmup for factor computation
    from datetime import datetime, timedelta
    warmup_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")

    bars_list = []
    for sym in pool:
        df = storage.load_daily_bars(sym, warmup_start, end)
        if df is not None and not df.empty:
            df = df[["date", "open", "close", "high", "low", "volume", "adj_factor"]].copy()
            df["symbol"] = sym
            bars_list.append(df)

    if not bars_list:
        click.echo("ERROR: No data found")
        return

    all_bars = pd.concat(bars_list, ignore_index=True)
    all_bars["date"] = pd.to_datetime(all_bars["date"])

    # Separate warmup data (for factor computation) from backtest data (for prices)
    all_bars_bt = all_bars[all_bars["date"] >= start].copy()
    open_prices = all_bars_bt.pivot(index="date", columns="symbol", values="open")
    close_prices = all_bars_bt.pivot(index="date", columns="symbol", values="close")
    adj_factor = all_bars_bt.pivot(index="date", columns="symbol", values="adj_factor")
    volume_wide = all_bars_bt.pivot(index="date", columns="symbol", values="volume")

    # Load benchmark (CSI300)
    bench_df = storage.load_daily_bars("000300", start, end)
    if bench_df is None or bench_df.empty:
        click.echo("ERROR: No benchmark data (000300)")
        return
    bench_close = bench_df.set_index("date")["close"]
    bench_close.index = pd.to_datetime(bench_close.index)

    click.echo(f"Data loaded: {close_prices.shape[0]} days × {close_prices.shape[1]} stocks")

    # Compute scores based on strategy (uses full data including warmup)
    if strategy == "baseline":
        from pangu.backtest.scoring import compute_baseline_scores
        scores = compute_baseline_scores(all_bars, storage, start, end)
    else:
        click.echo(f"Unknown strategy: {strategy}")
        return

    if scores is None or scores.empty:
        click.echo("ERROR: No scores computed")
        return

    click.echo(f"Scores: {scores.shape}")

    # Point-in-time constituent filter + suspended stock exclusion
    universe_fn = make_universe_fn(storage, volume_wide)

    # Run backtest with dynamic constituents
    engine = BacktestEngine(top_n=top_n, initial_capital=capital)
    result = engine.run(scores, open_prices, close_prices, adj_factor, bench_close,
                        start, end, universe_fn=universe_fn)

    # Print results
    click.echo(f"\n{'='*60}")
    click.echo(f"Backtest Result: {strategy} ({start} ~ {end}, TopN={top_n})")
    click.echo(f"{'='*60}")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            click.echo(f"  {k:20s}: {v:+.4f}" if abs(v) < 10 else f"  {k:20s}: {v:.1f}")
        else:
            click.echo(f"  {k:20s}: {v}")
    click.echo(f"  {'rebalances':20s}: {len(result.rebalance_log)}")


@main.command()
def status() -> None:
    """Show database statistics and strategy returns."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()
    db = c.db
    s = db.get_db_stats()

    click.echo("📊 PanGu Database Status")
    click.echo("─" * 35)
    click.echo(f"  Daily bars:     {s['bars_count']:,} rows ({s['bars_symbols']} stocks)")
    click.echo(f"  Latest bar:     {s['latest_bar']}")
    click.echo(f"  Trade signals:  {s['signals_count']:,}")
    click.echo(f"  Latest signal:  {s['latest_signal']}")
    click.echo(f"  Calendar:       {s['calendar_count']:,} trading days")
    click.echo(f"  News:           {s['news_count']:,} items")

    # Watchlist
    watchlist = c.stock_pool.get_watchlist()
    click.echo(f"  Watchlist:      {len(watchlist)} stocks")

    # Strategy returns
    returns = db.get_signal_returns(30)
    has_returns = any(r["count"] > 0 for r in returns.values())
    if has_returns:
        click.echo("\n📈 30-Day Strategy Returns")
        click.echo("─" * 35)
        for key in ("1d", "3d", "5d"):
            r = returns[key]
            if r["count"] > 0:
                emoji = "📈" if r["avg_return"] >= 0 else "📉"
                click.echo(f"  {key.upper()}: {emoji} {r['avg_return']:+.2f}% ({r['count']} signals)")
