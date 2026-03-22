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
# train commands
# ---------------------------------------------------------------------------

@main.group()
def train() -> None:
    """Train ML models."""


@train.command("walkforward")
@click.option("--factors", "factors_path", default=None,
              help="Pre-computed factors.parquet (default: compute from DB)")
@click.option("--model-dir", default="models", help="Directory to save trained models")
@click.option("--output", default="data/score_matrix.parquet", help="Score matrix output path")
@click.option("--pool", "pool_file", default=None, help="YAML file with symbols list (overrides default)")
@click.option("--label-horizon", default="5", type=str,
              help="Forward return horizon(s) in trading days. Single int (e.g. '5') or "
              "comma-separated for multi-horizon fusion (e.g. '5,10,20'). "
              "Multi-horizon fuses raw excess returns across horizons, "
              "aligning with TopkDropout strategies where holding periods vary.")
@click.option("--label-horizon-weights", default=None, type=str,
              help="Weights for multi-horizon fusion: comma-separated floats, e.g. '0.5,0.3,0.2'. "
              "Must match --label-horizon length. Default: equal weights.")
@click.option("--train-months", default=18, type=int,
              help="Training window length in months (default: 18)")
@click.option("--first-train-start", default="2020-01-01",
              help="First window training start date (default: 2020-01-01)")
@click.option("--last-test-end", default="2025-12-31",
              help="Last window test end date; windows are generated until test_end ≤ this date "
              "(default: 2025-12-31)")
@click.option("--params-file", default=None, type=click.Path(exists=True),
              help="LightGBM params JSON file (overrides defaults). "
              "Default params when not set: objective=mae, num_leaves=31, "
              "learning_rate=0.02, n_estimators=2000, subsample=0.8, "
              "colsample_bytree=0.7, min_child_samples=100, early_stopping=200")
@click.option("--normalize-label/--no-normalize-label", default=False,
              help="Cross-sectional z-score on labels (Qlib CSZScoreNorm). "
              "Model learns relative outperformance instead of absolute excess returns. "
              "(default: disabled)")
@click.option("--mode", type=click.Choice(["regression", "ranking"]), default="regression",
              help="Training mode: regression (MAE, default) or ranking (LambdaRank NDCG)")
@click.option("--n-bins", default=10, type=int,
              help="Number of relevance bins for ranking mode (default: 10 = decile). "
              "Ignored in regression mode.")
@click.option("--early-stop-metric", type=click.Choice(["mae", "rankic"]), default="mae",
              help="Early stopping metric for regression mode: mae (default) or rankic "
              "(daily Spearman rank correlation — aligns stopping with ranking quality). "
              "Ignored in ranking mode.")
@click.option("--time-decay-halflife", default=0, type=click.IntRange(min=0),
              help="Half-life in trading days for exponential time-decay sample weights. "
              "0 = no decay (uniform, default). Typical: 40 (~2mo), 80 (~4mo), 120 (~6mo). "
              "Recent samples get higher weight; oldest in 18mo window get ~0.12 with halflife=120.")
@click.option("--train-subsample-stride", default=0, type=click.IntRange(min=0),
              help="Random block subsampling stride for training data. "
              "0 = no subsampling (default). Divides training dates into blocks of this size "
              "and randomly selects one date per block. Reduces label overlap redundancy. "
              "Typically set equal to label horizon (e.g. 5 for 5-day labels).")
def train_walkforward_cmd(factors_path: str | None, model_dir: str, output: str,
                          pool_file: str | None, label_horizon: str,
                          label_horizon_weights: str | None,
                          train_months: int, first_train_start: str,
                          last_test_end: str, params_file: str | None,
                          normalize_label: bool, mode: str, n_bins: int,
                          early_stop_metric: str, time_decay_halflife: int,
                          train_subsample_stride: int) -> None:
    """Run Walk-Forward LightGBM training."""
    import json
    import logging
    import math

    from pangu.main import build_components, load_env
    from pangu.ml.model import train_walk_forward

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    load_env()
    c, _, _ = build_components()
    storage = c.market._storage

    # Parse horizon: single int or comma-separated list
    parsed_horizon: int | list[int] = 5
    parsed_weights: list[float] | None = None
    parts = [h.strip() for h in label_horizon.split(",") if h.strip()]
    if not parts:
        raise click.BadParameter("--label-horizon cannot be empty", param_hint="label_horizon")
    int_parts = [int(p) for p in parts]
    if any(h <= 0 for h in int_parts):
        raise click.BadParameter("All horizons must be positive integers", param_hint="label_horizon")
    parsed_horizon = int_parts if len(int_parts) > 1 else int_parts[0]
    if isinstance(parsed_horizon, list):
        click.echo(f"Multi-horizon label fusion: {parsed_horizon}")

    if label_horizon_weights:
        if not isinstance(parsed_horizon, list):
            raise click.BadParameter(
                "--label-horizon-weights requires multi-horizon (e.g. --label-horizon 5,10,20)",
                param_hint="label_horizon_weights",
            )
        parsed_weights = [float(w.strip()) for w in label_horizon_weights.split(",") if w.strip()]
        if any(not math.isfinite(w) or w < 0 for w in parsed_weights):
            raise click.BadParameter(
                "All weights must be non-negative finite numbers", param_hint="label_horizon_weights"
            )
        if sum(parsed_weights) <= 0:
            raise click.BadParameter("Weights must sum to a positive value", param_hint="label_horizon_weights")
        if len(parsed_weights) != len(parsed_horizon):
            raise click.BadParameter(
                f"Weight count ({len(parsed_weights)}) must match horizon count ({len(parsed_horizon)})",
                param_hint="label_horizon_weights",
            )
        click.echo(f"Horizon weights: {parsed_weights}")

    # Load custom LightGBM params from JSON file
    params = None
    if params_file:
        with open(params_file) as f:
            params = json.load(f)
        click.echo(f"Loaded custom params from {params_file}: {params}")

    score_matrix = train_walk_forward(
        storage=storage,
        factors_path=factors_path,
        model_dir=model_dir,
        output_path=output,
        params=params,
        label_horizon=parsed_horizon,
        label_horizon_weights=parsed_weights,
        train_months=train_months,
        first_train_start=first_train_start,
        last_test_end=last_test_end,
        normalize_label=normalize_label,
        mode=mode,
        n_bins=n_bins,
        early_stop_metric=early_stop_metric,
        time_decay_halflife=time_decay_halflife,
        train_subsample_stride=train_subsample_stride or None,
    )

    click.echo("\n✅ Walk-Forward training complete")
    click.echo(f"  Score matrix: {score_matrix.shape[0]} days × {score_matrix.shape[1]} stocks")
    click.echo(f"  Output: {output}")
    click.echo(f"  Models: {model_dir}/wf_window_*.txt")


# ---------------------------------------------------------------------------
# backfill commands
# ---------------------------------------------------------------------------

@main.group()
def backfill() -> None:
    """Backfill historical data from upstream providers."""


def _load_pool_yaml(path: str) -> list[str] | None:
    """Load stock list from YAML file (key: ``symbols``). Returns None on error."""
    import yaml
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        pool = data.get("symbols") or []
        if not pool:
            click.echo(f"Warning: No symbols found in {path}", err=True)
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
@click.option("--output", "-o", default="config/backfill_stock_pool.yaml", help="Output YAML path for stock pool")
def backfill_constituents(start: str, end: str | None, output: str) -> None:
    """Backfill historical index constituents (CSI300 + CSI500) semi-annually."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    count, all_symbols = c.stock_pool.sync_historical_constituents(start, end)
    click.echo(f"\n✅ Saved {count} constituent records, {len(all_symbols)} unique stocks")

    # Export unique symbols to YAML
    with open(output, "w") as f:
        f.write("symbols:\n")
        for sym in sorted(all_symbols):
            f.write(f'- "{sym}"\n')
    click.echo(f"📄 Exported {len(all_symbols)} symbols to {output}")


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
        if i % 10 == 0 or i == total:
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

        if i % 10 == 0 or i == total:
            click.echo(f"  [{i}/{total}] ok={ok} fail={fail}")

    click.echo(f"\n✅ Backfill fundamentals done: {ok} ok, {fail} failed")

    # Gross margin supplement via stock_yjbb_em (per-quarter batch API)
    from pangu.tz import now as tz_now
    today = tz_now().strftime("%Y-%m-%d")
    click.echo(f"\nBackfilling gross_margin ({start} → {today})...")
    gm_ok, gm_fail = c.fundamental.refresh_gross_margin(start, today)
    click.echo(f"✅ gross_margin done: {gm_ok} quarters ok, {gm_fail} failed")

    # Publication dates backfill via cninfo disclosure schedule (PIT fix)
    click.echo(f"\nBackfilling pub_dates ({start} → {today})...")
    pd_ok, pd_fail = c.fundamental.refresh_pub_dates(start, today)
    click.echo(f"✅ pub_dates done: {pd_ok} quarters ok, {pd_fail} failed")




# ---------------------------------------------------------------------------
# backtest command
# ---------------------------------------------------------------------------

@main.command("backtest")
@click.option("--strategy", type=click.Choice(["baseline", "lgb"]), default="baseline",
              help="Strategy to backtest")
@click.option("--start", default="2022-01-01", help="Start date")
@click.option("--end", default="2025-12-31", help="End date")
@click.option("--top-n", default=30, type=int, help="TopN stocks to hold (default: 30)")
@click.option("--n-drop", default=10, type=int,
              help="TopkDropout: stocks to drop and replace each rebalance (default: 10). "
              "Set to 0 to disable dropout and use pure top-N replacement.")
@click.option("--capital", default=1_000_000, type=float, help="Initial capital (CNY)")
@click.option("--stamp-tax", default=0.001, type=float,
              help="Sell stamp tax rate (default: 0.001)")
@click.option("--commission", default=0.0003, type=float,
              help="Broker commission rate (default: 0.0003)")
@click.option("--slippage", default=0.001, type=float,
              help="Slippage rate (default: 0.001)")
@click.option("--exclude-prefixes", default="688,689",
              help="Comma-separated stock code prefixes to exclude (default: 688,689 STAR market)")
@click.option("--pool", "pool_file", default=None, help="YAML file with symbols list (overrides default)")
@click.option("--scores", "scores_path", default=None,
              help="Parquet file with pre-computed scores (for lgb strategy)")
@click.option("--plot/--no-plot", default=True,
              help="Generate equity curve chart (default: --plot)")
@click.option("--plot-output", default=None,
              help="Chart output path (default: data/backtest_{strategy}_{date}.png)")
def backtest_cmd(strategy: str, start: str, end: str, top_n: int, n_drop: int,
                 capital: float,
                 stamp_tax: float, commission: float, slippage: float,
                 exclude_prefixes: str,
                 pool_file: str | None, scores_path: str | None,
                 plot: bool, plot_output: str | None) -> None:
    """Run local backtest for a given strategy."""
    from pangu.main import build_components, load_env
    load_env()
    c, _, _ = build_components()

    import pandas as pd

    from pangu.backtest.engine import BacktestEngine, make_universe_fn

    storage = c.market._storage

    # Pool: explicit YAML override → constituents union (default)
    if pool_file:
        pool = _load_pool_yaml(pool_file)
        if pool is None:
            return
        click.echo(f"Using pool from {pool_file}: {len(pool)} stocks")
    else:
        pool = storage.load_constituents_union(start, end)
        if not pool:
            click.echo("ERROR: No historical constituents found. Run 'pangu backfill constituents' first.")
            return
        click.echo(f"Pool from constituents union: {len(pool)} stocks")

    # Load price data with 120-day warmup for factor computation
    from datetime import datetime, timedelta
    warmup_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")

    bars_list = []
    for sym in pool:
        df = storage.load_daily_bars(sym, warmup_start, end)
        if df is not None and not df.empty:
            df = df[["date", "open", "close", "high", "low", "volume", "adj_factor", "is_st"]].copy()
            df["symbol"] = sym
            bars_list.append(df)

    if not bars_list:
        click.echo("ERROR: No data found")
        return

    all_bars = pd.concat(bars_list, ignore_index=True)
    all_bars["date"] = pd.to_datetime(all_bars["date"])

    # Separate warmup data (for factor computation) from backtest data (for prices)
    # DB stores unadjusted prices; engine uses them directly for execution
    all_bars_bt = all_bars[all_bars["date"] >= start].copy()
    open_prices = all_bars_bt.pivot(index="date", columns="symbol", values="open")
    close_prices = all_bars_bt.pivot(index="date", columns="symbol", values="close")
    volume_wide = all_bars_bt.pivot(index="date", columns="symbol", values="volume")
    adj_factor_wide = all_bars_bt.pivot(index="date", columns="symbol", values="adj_factor")
    is_st_wide = all_bars_bt.pivot(index="date", columns="symbol", values="is_st")

    # Load benchmark (CSI300) — include days before start for pre-start close
    bench_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=15)).strftime("%Y-%m-%d")
    bench_df = storage.load_daily_bars("000300", bench_start, end)
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
    elif strategy == "lgb":
        if not scores_path:
            click.echo("ERROR: --scores required for lgb strategy (e.g. data/score_matrix.parquet)")
            return
        scores = pd.read_parquet(scores_path)
        scores.index = pd.to_datetime(scores.index)
        # Validate score coverage vs backtest range
        score_start = scores.index.min().strftime("%Y-%m-%d")
        score_end = scores.index.max().strftime("%Y-%m-%d")
        if scores.index.min() > pd.Timestamp(start):
            click.echo(f"WARNING: Score matrix starts at {score_start}, "
                       f"but backtest starts at {start}. "
                       f"Days before {score_start} will have no signals.")
        if scores.index.max() < pd.Timestamp(end):
            click.echo(f"WARNING: Score matrix ends at {score_end}, "
                       f"but backtest ends at {end}. "
                       f"Days after {score_end} will have no signals.")
        # Filter to backtest date range
        scores = scores[(scores.index >= start) & (scores.index <= end)]
        click.echo(f"Loaded scores from {scores_path}: {scores.shape}")
    else:
        click.echo(f"Unknown strategy: {strategy}")
        return

    if scores is None or scores.empty:
        click.echo("ERROR: No scores computed")
        return

    click.echo(f"Scores: {scores.shape}")

    # Point-in-time constituent filter (suspension handled at execution time)
    universe_fn = make_universe_fn(storage)

    # Parse exclude_prefixes
    prefixes = tuple(
        p.strip() for p in exclude_prefixes.split(",") if p.strip()
    ) if exclude_prefixes else ()

    # Run backtest with dynamic constituents
    engine = BacktestEngine(
        top_n=top_n, n_drop=n_drop, initial_capital=capital,
        stamp_tax=stamp_tax, commission=commission, slippage=slippage,
    )
    result = engine.run(scores, open_prices, close_prices, bench_close,
                        start, end, universe_fn=universe_fn, volume=volume_wide,
                        adj_factor=adj_factor_wide, is_st=is_st_wide,
                        exclude_prefixes=prefixes)

    # Print results
    click.echo(f"\n{'='*60}")
    dropout_str = f", n_drop={n_drop}" if n_drop > 0 else ""
    click.echo(f"Backtest Result: {strategy} ({start} ~ {end}, TopN={top_n}{dropout_str})")
    click.echo(f"{'='*60}")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            click.echo(f"  {k:20s}: {v:+.4f}" if abs(v) < 10 else f"  {k:20s}: {v:.1f}")
        else:
            click.echo(f"  {k:20s}: {v}")
    click.echo(f"  {'rebalances':20s}: {len(result.rebalance_log)}")

    # Plot equity curve
    if plot:
        from datetime import datetime

        from pangu.backtest.plot import plot_equity_curve
        if plot_output is None:
            plot_output = f"data/backtest_{strategy}_{datetime.now().strftime('%Y%m%d')}.png"
        path = plot_equity_curve(result.nav, result.benchmark_nav, strategy, plot_output,
                                initial_capital=capital)
        click.echo(f"\n📈 Chart saved: {path}")


# ---------------------------------------------------------------------------
# evaluate-scores command
# ---------------------------------------------------------------------------

@main.command("evaluate-scores")
@click.option("--scores", "scores_path", default="data/score_matrix.parquet",
              help="Path to score matrix parquet (default: data/score_matrix.parquet)")
@click.option("--top-n", "top_n_str", default="10,30,50",
              help="Comma-separated Top-N values to evaluate (default: 10,30,50)")
@click.option("--json", "output_json", is_flag=True, default=False,
              help="Output results as JSON instead of table")
def evaluate_scores_cmd(scores_path: str, top_n_str: str, output_json: bool) -> None:
    """Diagnose score matrix quality (discrimination, stability, rank stability)."""
    import json
    from pathlib import Path

    import pandas as pd

    from pangu.ml.score_evaluator import evaluate_scores, format_report

    path = Path(scores_path)
    if not path.exists():
        click.echo(f"ERROR: Score file not found: {scores_path}")
        return

    scores = pd.read_parquet(path)
    scores.index = pd.to_datetime(scores.index)
    click.echo(f"Loaded scores: {scores.shape[0]} days × {scores.shape[1]} stocks "
               f"({scores.index.min().date()} ~ {scores.index.max().date()})")

    try:
        top_ns = [int(x.strip()) for x in top_n_str.split(",")]
    except ValueError:
        click.echo("ERROR: Invalid --top-n value. Expected comma-separated integers.", err=True)
        return
    results = evaluate_scores(scores, top_ns=top_ns)

    if output_json:
        click.echo(json.dumps(results, indent=2, default=str))
    else:
        click.echo(format_report(results))


# ---------------------------------------------------------------------------
# evaluate-models command
# ---------------------------------------------------------------------------

@main.command("evaluate-models")
@click.option("--model-dir", default="models", help="Directory with wf_window_*.txt files")
@click.option("--top-n", default=20, type=int, help="Number of top features to show")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON")
def evaluate_models_cmd(model_dir: str, top_n: int, output_json: bool) -> None:
    """Diagnose walk-forward model quality (importance, drift, underfitting)."""
    import json
    from pathlib import Path

    from pangu.ml.model_evaluator import evaluate_models, format_model_report

    path = Path(model_dir)
    if not path.is_dir():
        click.echo(f"ERROR: Model directory not found: {model_dir}", err=True)
        return

    model_files = list(path.glob("wf_window_*.txt"))
    if not model_files:
        click.echo(f"ERROR: No wf_window_*.txt files in {model_dir}", err=True)
        return

    click.echo(f"Loading {len(model_files)} window models from {model_dir}/")
    results = evaluate_models(model_dir, top_n=top_n)

    if output_json:
        click.echo(json.dumps(results, indent=2, default=str))
    else:
        click.echo(format_model_report(results))



@main.command("compute-factors")
@click.option("--start", default="2019-01-01", help="Start date (include warmup)")
@click.option("--end", default="2025-12-31", help="End date")
@click.option("--output", default="data/factors.parquet", help="Output parquet path")
@click.option("--pool", "pool_file", default=None, help="YAML file with symbols list (overrides default)")
def compute_factors_cmd(start: str, end: str, output: str, pool_file: str | None) -> None:
    """Compute Alpha158 + fundamental factors for all stocks."""
    import time
    from pathlib import Path

    import pandas as pd

    from pangu.factor.alpha158 import Alpha158Engine
    from pangu.main import build_components, load_env

    load_env()
    c, _, _ = build_components()
    storage = c.market._storage

    # Determine stock pool: explicit YAML override → constituents union (default)
    if pool_file:
        pool = _load_pool_yaml(pool_file)
        if pool is None:
            return
        click.echo(f"Pool from {pool_file}: {len(pool)} stocks")
    else:
        pool = storage.load_constituents_union(start, end)
        if not pool:
            click.echo("ERROR: No historical constituents found. Run 'pangu backfill constituents' first.")
            return
        click.echo(f"Pool from constituents union: {len(pool)} stocks")

    # Load bars
    t0 = time.time()
    bar_frames = []
    for sym in pool:
        df = storage.load_daily_bars(sym, start, end)
        if df is not None and not df.empty:
            bar_frames.append(df)

    if not bar_frames:
        click.echo("ERROR: No bar data found")
        return

    all_bars = pd.concat(bar_frames, ignore_index=True)
    click.echo(f"  Loaded {len(all_bars):,} bar rows ({len(bar_frames)} stocks) in {time.time()-t0:.1f}s")

    # Load fundamentals
    t1 = time.time()
    fund_frames = []
    for sym in pool:
        df = storage.load_fundamentals_filled(sym, start, end)
        if df is not None and not df.empty:
            fund_frames.append(df)

    fundamentals = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()
    click.echo(f"  Loaded {len(fundamentals):,} fundamental rows in {time.time()-t1:.1f}s")

    # Compute
    t2 = time.time()
    engine = Alpha158Engine()
    panel = engine.compute(all_bars, fundamentals)
    click.echo(f"  Computed {panel.shape[1]} factors × {panel.shape[0]:,} rows in {time.time()-t2:.1f}s")

    # Save
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)
    file_mb = Path(output).stat().st_size / 1024 / 1024
    click.echo(f"  Saved to {output} ({file_mb:.1f} MB)")

    # Coverage stats
    total = panel.shape[0]
    non_nan = panel.notna().sum()
    coverage = (non_nan / total * 100).describe()
    click.echo("\n  Factor coverage (% non-NaN):")
    click.echo(f"    min: {coverage['min']:.1f}%, mean: {coverage['mean']:.1f}%, max: {coverage['max']:.1f}%")


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
