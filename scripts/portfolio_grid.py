"""Portfolio parameter grid search (v2: 3-Fold Temporal CV).

Searches over top_n × n_drop × score_smooth_halflife combinations
using a pre-trained model's val score matrix. Objective is 3-fold
mean Sharpe for consistency with HPO v2.

Usage:
    uv run python scripts/portfolio_grid.py --scores data/score_matrix_val.parquet
    uv run python scripts/portfolio_grid.py --scores data/score_matrix_val.parquet --output data/grid_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pangu.backtest.engine import BacktestEngine, make_universe_fn
from pangu.main import build_components, load_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("pangu.data").setLevel(logging.WARNING)

# Default grid (per plan.md Phase C)
TOP_N_VALUES = [20, 30, 50]
N_DROP_VALUES = [0, 3, 5, 8, 10, 12, 15]
SMOOTH_HALFLIFE_VALUES = [0, 1, 2, 3, 5]

# Full val backtest window
VAL_BACKTEST_START = "2022-01-01"
VAL_BACKTEST_END = "2025-08-31"

# 3-Fold Temporal CV (same as optuna_hpo.py)
VAL_FOLDS = [
    ("2022-01-01", "2023-02-28"),  # Fold 1: 14 months
    ("2023-03-01", "2024-04-30"),  # Fold 2: 14 months
    ("2024-05-01", "2025-08-31"),  # Fold 3: 16 months
]


def load_backtest_data(storage, start: str, end: str) -> dict:
    """Load price/benchmark data for backtest."""
    warmup_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")

    pool = storage.load_constituents_union(start, end)
    logger.info("Pool: %d stocks", len(pool))

    bars_list = []
    for sym in pool:
        df = storage.load_daily_bars(sym, warmup_start, end)
        if df is not None and not df.empty:
            df = df[["date", "open", "close", "volume", "adj_factor", "is_st"]].copy()
            df["symbol"] = sym
            bars_list.append(df)

    all_bars = pd.concat(bars_list, ignore_index=True)
    all_bars["date"] = pd.to_datetime(all_bars["date"])
    all_bars_bt = all_bars[all_bars["date"] >= start].copy()

    open_prices = all_bars_bt.pivot(index="date", columns="symbol", values="open")
    close_prices = all_bars_bt.pivot(index="date", columns="symbol", values="close")
    volume_wide = all_bars_bt.pivot(index="date", columns="symbol", values="volume")
    adj_factor_wide = all_bars_bt.pivot(index="date", columns="symbol", values="adj_factor")
    is_st_wide = all_bars_bt.pivot(index="date", columns="symbol", values="is_st")

    bench_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=15)).strftime("%Y-%m-%d")
    bench_df = storage.load_daily_bars("000300", bench_start, end)
    bench_close = bench_df.set_index("date")["close"]
    bench_close.index = pd.to_datetime(bench_close.index)

    universe_fn = make_universe_fn(storage)

    return {
        "open_prices": open_prices,
        "close_prices": close_prices,
        "volume": volume_wide,
        "adj_factor": adj_factor_wide,
        "is_st": is_st_wide,
        "bench_close": bench_close,
        "universe_fn": universe_fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Portfolio grid search")
    parser.add_argument("--scores", required=True, help="Path to score_matrix_val.parquet")
    parser.add_argument("--start", default=VAL_BACKTEST_START, help="Backtest start")
    parser.add_argument("--end", default=VAL_BACKTEST_END, help="Backtest end")
    parser.add_argument("--output", default="data/grid_results.json", help="Output JSON")
    args = parser.parse_args()

    load_env()
    c, _, _ = build_components()
    storage = c.market._storage

    # Load scores
    raw_scores = pd.read_parquet(args.scores)
    raw_scores.index = pd.to_datetime(raw_scores.index)
    raw_scores = raw_scores[
        (raw_scores.index >= pd.Timestamp(args.start))
        & (raw_scores.index <= pd.Timestamp(args.end))
    ]
    logger.info("Scores: %d days × %d stocks", raw_scores.shape[0], raw_scores.shape[1])

    # Load backtest data
    logger.info("Loading backtest data...")
    bt_data = load_backtest_data(storage, args.start, args.end)

    # Grid search
    combos = list(product(TOP_N_VALUES, N_DROP_VALUES, SMOOTH_HALFLIFE_VALUES))
    total = len(combos)
    logger.info("Grid search: %d combinations", total)

    results = []
    for i, (top_n, n_drop, smooth_hl) in enumerate(combos):
        # n_drop must be < top_n to be valid
        if n_drop >= top_n:
            continue

        # Apply EMA smoothing
        if smooth_hl > 0:
            scores = raw_scores.ewm(halflife=smooth_hl, min_periods=1).mean()
        else:
            scores = raw_scores

        # 3-Fold Temporal CV
        fold_sharpes = []
        fold_metrics_list = []
        for fold_start, fold_end in VAL_FOLDS:
            scores_fold = scores[
                (scores.index >= pd.Timestamp(fold_start))
                & (scores.index <= pd.Timestamp(fold_end))
            ]
            if scores_fold.empty:
                fold_sharpes.append(-999.0)
                continue

            try:
                engine = BacktestEngine(top_n=top_n, n_drop=n_drop)
                result = engine.run(
                    scores_fold,
                    bt_data["open_prices"],
                    bt_data["close_prices"],
                    bt_data["bench_close"],
                    start=fold_start,
                    end=fold_end,
                    universe_fn=bt_data["universe_fn"],
                    volume=bt_data["volume"],
                    adj_factor=bt_data["adj_factor"],
                    is_st=bt_data["is_st"],
                )
                fold_sharpes.append(result.metrics.get("sharpe", -999.0))
                fold_metrics_list.append(result.metrics)
            except Exception as e:
                logger.warning("Grid combo top_n=%d n_drop=%d fold %s~%s failed: %s",
                               top_n, n_drop, fold_start, fold_end, e)
                fold_sharpes.append(-999.0)

        mean_sharpe = float(np.mean(fold_sharpes))
        std_sharpe = float(np.std(fold_sharpes))

        # Also run full-period for reference
        try:
            engine = BacktestEngine(top_n=top_n, n_drop=n_drop)
            full_result = engine.run(
                scores,
                bt_data["open_prices"],
                bt_data["close_prices"],
                bt_data["bench_close"],
                start=args.start,
                end=args.end,
                universe_fn=bt_data["universe_fn"],
                volume=bt_data["volume"],
                adj_factor=bt_data["adj_factor"],
                is_st=bt_data["is_st"],
            )
            full_metrics = full_result.metrics
        except Exception as e:
            logger.warning("Grid combo top_n=%d n_drop=%d full-period failed: %s", top_n, n_drop, e)
            full_metrics = {"sharpe": -999.0}

        entry = {
            "top_n": top_n,
            "n_drop": n_drop,
            "score_smooth_halflife": smooth_hl,
            "mean_sharpe": mean_sharpe,
            "sharpe_std": std_sharpe,
            "sharpe_fold1": fold_sharpes[0],
            "sharpe_fold2": fold_sharpes[1],
            "sharpe_fold3": fold_sharpes[2],
            "sharpe_full": full_metrics.get("sharpe", -999.0),
            **full_metrics,
        }
        results.append(entry)

        if (i + 1) % 10 == 0 or i + 1 == total:
            logger.info("[%d/%d] top_n=%d, n_drop=%d, smooth=%d → mean_Sharpe=%.4f (%.3f/%.3f/%.3f)",
                        i + 1, total, top_n, n_drop, smooth_hl,
                        mean_sharpe, fold_sharpes[0], fold_sharpes[1], fold_sharpes[2])

    # Sort by mean Sharpe (3-fold)
    results.sort(key=lambda x: x["mean_sharpe"], reverse=True)

    # Print top 10
    print(f"\n{'='*110}")
    print(f"Top-10 Portfolio Configurations (out of {len(results)}, ranked by 3-fold mean Sharpe)")
    print(f"{'='*110}")
    print(f"{'#':>3} {'TopN':>5} {'Drop':>5} {'EMA':>4} {'MeanS':>7} {'F1':>7} {'F2':>7} "
          f"{'F3':>7} {'Std':>6} {'FullS':>7} {'AnnRet':>8} {'MaxDD':>8} {'Turn':>6}")
    print("-" * 110)

    for i, r in enumerate(results[:10]):
        print(f"{i+1:>3} {r['top_n']:>5} {r['n_drop']:>5} {r['score_smooth_halflife']:>4} "
              f"{r['mean_sharpe']:>+7.3f} "
              f"{r['sharpe_fold1']:>+7.3f} {r['sharpe_fold2']:>+7.3f} {r['sharpe_fold3']:>+7.3f} "
              f"{r['sharpe_std']:>6.3f} {r['sharpe_full']:>+7.3f} "
              f"{r.get('annual_return', 0):>+8.4f} {r.get('max_drawdown', 0):>+8.4f} "
              f"{r.get('annual_turnover', 0):>6.1f}")

    # Marginal effects
    print(f"\n{'='*60}")
    print("Marginal Effects (mean Sharpe by parameter value)")
    print(f"{'='*60}")
    df = pd.DataFrame(results)

    for param in ["top_n", "n_drop", "score_smooth_halflife"]:
        means = df.groupby(param)["mean_sharpe"].mean().sort_index()
        print(f"\n{param}:")
        for val, sharpe in means.items():
            print(f"  {val:>4}: {sharpe:+.4f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "n_combos": len(results)}, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)

    # Print best
    best = results[0]
    print(f"\n{'='*60}")
    print(f"Best: top_n={best['top_n']}, n_drop={best['n_drop']}, "
          f"smooth={best['score_smooth_halflife']} → mean_Sharpe={best['mean_sharpe']:.4f} "
          f"(folds: {best['sharpe_fold1']:.3f}/{best['sharpe_fold2']:.3f}/{best['sharpe_fold3']:.3f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
