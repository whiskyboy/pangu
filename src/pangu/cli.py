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


@backfill.command("bars")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
def backfill_bars(start: str) -> None:
    """Backfill daily OHLCV + PE/PB for all pool stocks."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    from pangu.tz import today_str

    pool = c.stock_pool.get_all_symbols()
    today = today_str()
    total = len(pool)
    ok, fail = 0, 0

    click.echo(f"Backfilling {total} stocks from {start} to {today}...")
    for i, symbol in enumerate(pool, 1):
        try:
            df = c.market.get_daily_bars(symbol, start, today)
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
def backfill_index(start: str, symbol: str) -> None:
    """Backfill index daily bars (default: CSI300)."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    from pangu.tz import today_str

    today = today_str()
    click.echo(f"Backfilling index {symbol} from {start} to {today}...")
    df = c.market.get_index_daily_bars(symbol, start, today)
    click.echo(f"✅ Index {symbol}: {len(df)} bars loaded")


@backfill.command("fundamentals")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD, e.g. 2022-10-01)")
def backfill_fundamentals(start: str) -> None:
    """Backfill quarterly financial indicators."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    pool = c.stock_pool.get_all_symbols()
    total = len(pool)
    ok, fail = 0, 0

    click.echo(f"Backfilling fundamentals for {total} stocks from {start}...")
    for i, symbol in enumerate(pool, 1):
        try:
            df = c.fundamental.get_financial_indicator(symbol, start=start)
            if df is not None and not df.empty:
                ok += 1
            else:
                fail += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            click.echo(f"  ✗ {symbol}: {e}")
        if i % 50 == 0 or i == total:
            click.echo(f"  [{i}/{total}] ok={ok} fail={fail}")

    click.echo(f"\n✅ Backfill fundamentals done: {ok} ok, {fail} failed")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

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
