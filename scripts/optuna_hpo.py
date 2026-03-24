"""Optuna Bayesian HPO for LightGBM Walk-Forward training (v2).

19-dimensional joint search over training hyperparameters.
Objective: 3-Fold Temporal CV mean Sharpe (anti-overfitting).

Folds split the val period (2022-01 ~ 2025-08) into 3 non-overlapping
sub-periods. Each trial trains once → runs 3 backtests → returns mean Sharpe.

Usage:
    uv run python scripts/optuna_hpo.py                # 80 trials
    uv run python scripts/optuna_hpo.py --n-trials 20  # quick test
    uv run python scripts/optuna_hpo.py --resume        # resume from checkpoint

Storage: data/optuna_study.db (SQLite, supports checkpoint/resume)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pangu.backtest.engine import BacktestEngine, make_universe_fn
from pangu.main import build_components, load_env
from pangu.ml.model import train_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers during HPO
logging.getLogger("pangu.ml.model").setLevel(logging.WARNING)
logging.getLogger("pangu.ml.dataset").setLevel(logging.WARNING)
logging.getLogger("pangu.data").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Fixed parameters (per plan.md)
# ---------------------------------------------------------------------------
FIXED_TRAIN = dict(
    first_train_start="2020-01-01",
    last_test_end="2025-12-31",
    val_months=3,
    test_months=3,
)

# Comparable backtest window = intersection of all possible val score ranges
VAL_BACKTEST_START = "2022-01-01"
VAL_BACKTEST_END = "2025-08-31"

# 3-Fold Temporal CV: split val period into non-overlapping sub-periods
VAL_FOLDS = [
    ("2022-01-01", "2023-02-28"),  # Fold 1: 14 months
    ("2023-03-01", "2024-04-30"),  # Fold 2: 14 months
    ("2024-05-01", "2025-08-31"),  # Fold 3: 16 months
]

# Fixed backtest params for Phase A/B (Phase C tunes these)
BACKTEST_TOP_N = 30
BACKTEST_N_DROP = 10


def _load_backtest_data(
    storage: object,
    start: str,
    end: str,
) -> dict:
    """Load all data needed for backtest. Cached across trials."""
    warmup_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")

    pool = storage.load_constituents_union(start, end)
    logger.info("Pool: %d stocks", len(pool))

    bars_list = []
    for sym in pool:
        df = storage.load_daily_bars(sym, warmup_start, end)
        if df is not None and not df.empty:
            df = df[["date", "open", "close", "high", "low", "volume", "adj_factor", "is_st"]].copy()
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


def run_backtest(
    scores: pd.DataFrame,
    bt_data: dict,
    start: str = VAL_BACKTEST_START,
    end: str = VAL_BACKTEST_END,
    top_n: int = BACKTEST_TOP_N,
    n_drop: int = BACKTEST_N_DROP,
) -> dict:
    """Run backtest and return metrics dict."""
    # Clip scores to backtest range
    scores_clipped = scores[
        (scores.index >= pd.Timestamp(start)) & (scores.index <= pd.Timestamp(end))
    ]
    if scores_clipped.empty:
        return {"sharpe": -999.0}

    engine = BacktestEngine(top_n=top_n, n_drop=n_drop)
    result = engine.run(
        scores_clipped,
        bt_data["open_prices"],
        bt_data["close_prices"],
        bt_data["bench_close"],
        start=start,
        end=end,
        universe_fn=bt_data["universe_fn"],
        volume=bt_data["volume"],
        adj_factor=bt_data["adj_factor"],
        is_st=bt_data["is_st"],
    )
    return result.metrics


def create_objective(storage: object, bt_data: dict, factors_path: str | None):
    """Create Optuna objective function (closure over shared data)."""

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()

        # --- Sample hyperparameters (19 dims + conditionals) ---
        num_leaves = trial.suggest_int("num_leaves", 7, 63)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 200, 5000)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        min_child_samples = trial.suggest_int("min_child_samples", 30, 300)
        max_bin = trial.suggest_categorical("max_bin", [63, 127, 255])
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 30, 500)
        min_iterations = trial.suggest_int("min_iterations", 20, 300)

        mode = trial.suggest_categorical("mode", ["regression", "ranking"])
        normalize_label = trial.suggest_categorical("normalize_label", [True, False])
        time_decay_halflife = trial.suggest_int("time_decay_halflife", 0, 200)
        train_months = trial.suggest_categorical("train_months", [12, 15, 18, 21, 24])
        step_months = trial.suggest_categorical("step_months", [1, 2, 3])
        train_subsample_stride = trial.suggest_int("train_subsample_stride", 0, 10)

        # label_horizon: single or multi-horizon
        label_horizon_str = trial.suggest_categorical(
            "label_horizon", ["5", "10", "5,10", "5,10,20"]
        )
        parts = [int(x) for x in label_horizon_str.split(",")]
        label_horizon: int | list[int] = parts if len(parts) > 1 else parts[0]

        # Conditional: multi-horizon weights
        label_horizon_weights = None
        if isinstance(label_horizon, list) and len(label_horizon) == 2:
            w0 = trial.suggest_float("label_weight_0", 0.2, 0.8)
            label_horizon_weights = [w0, 1.0 - w0]
        elif isinstance(label_horizon, list) and len(label_horizon) == 3:
            w0 = trial.suggest_float("label_weight_0", 0.1, 0.7)
            w1 = trial.suggest_float("label_weight_1", 0.1, min(0.7, 1.0 - w0))
            label_horizon_weights = [w0, w1, 1.0 - w0 - w1]

        # Conditional: ranking mode → n_bins
        n_bins = 10
        if mode == "ranking":
            n_bins = trial.suggest_int("n_bins", 5, 20)

        early_stop_metric = trial.suggest_categorical("early_stop_metric", ["mae", "rankic"])

        # --- Build LightGBM params ---
        lgb_params = {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_samples": min_child_samples,
            "max_bin": max_bin,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
        }

        # --- Train (n_seeds=1 for exploration) ---
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _, val_score_matrix = train_walk_forward(
                    storage=storage,
                    factors_path=factors_path,
                    model_dir=tmpdir,
                    output_dir=tmpdir,
                    params=lgb_params,
                    label_horizon=label_horizon,
                    label_horizon_weights=label_horizon_weights,
                    train_months=train_months,
                    normalize_label=normalize_label,
                    mode=mode,
                    n_bins=n_bins,
                    early_stop_metric=early_stop_metric,
                    time_decay_halflife=time_decay_halflife,
                    train_subsample_stride=train_subsample_stride or None,
                    step_months=step_months,
                    n_seeds=1,
                    early_stopping_rounds=early_stopping_rounds,
                    min_iterations=min_iterations,
                    **FIXED_TRAIN,
                )
            except Exception as e:
                logger.warning("Trial %d training failed: %s", trial.number, e)
                return -999.0

        # --- 3-Fold Temporal CV backtest ---
        fold_sharpes = []
        for i, (fold_start, fold_end) in enumerate(VAL_FOLDS):
            try:
                fold_metrics = run_backtest(val_score_matrix, bt_data, start=fold_start, end=fold_end)
            except Exception as e:
                logger.warning("Trial %d fold %d backtest failed: %s", trial.number, i + 1, e)
                fold_metrics = {"sharpe": -999.0}
            fold_sharpe = fold_metrics.get("sharpe", -999.0)
            fold_sharpes.append(fold_sharpe)
            trial.set_user_attr(f"sharpe_fold{i + 1}", fold_sharpe)
            for k, v in fold_metrics.items():
                if isinstance(v, (int, float)):
                    trial.set_user_attr(f"{k}_fold{i + 1}", v)

        mean_sharpe = float(np.mean(fold_sharpes))
        std_sharpe = float(np.std(fold_sharpes))
        trial.set_user_attr("sharpe_std", std_sharpe)

        # Also run full-period backtest for reference (not used in objective)
        try:
            full_metrics = run_backtest(val_score_matrix, bt_data)
        except Exception as e:
            logger.warning("Trial %d full-period backtest failed: %s", trial.number, e)
            full_metrics = {"sharpe": -999.0}
        trial.set_user_attr("sharpe_full", full_metrics.get("sharpe", -999.0))
        for k, v in full_metrics.items():
            if isinstance(v, (int, float)):
                trial.set_user_attr(k, v)

        elapsed = time.time() - t0
        logger.info(
            "Trial %d: mean_Sharpe=%.4f (folds: %.3f/%.3f/%.3f, std=%.3f) "
            "full_Sharpe=%.4f (%.0fs, mode=%s, metric=%s, leaves=%d, lr=%.4f, "
            "train=%dmo, step=%d)",
            trial.number, mean_sharpe,
            fold_sharpes[0], fold_sharpes[1], fold_sharpes[2], std_sharpe,
            full_metrics.get("sharpe", float("nan")),
            elapsed, mode, early_stop_metric, num_leaves, learning_rate,
            train_months, step_months,
        )
        trial.set_user_attr("elapsed_seconds", elapsed)

        return mean_sharpe

    return objective


def print_top_trials(study: optuna.Study, n: int = 10) -> str:
    """Print top N trials and return formatted string."""
    trials = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)
    lines = [f"\n{'='*100}", f"Top-{n} Trials (out of {len(study.trials)} total)", f"{'='*100}"]
    lines.append(
        f"{'#':>4} {'Trial':>6} {'MeanS':>7} {'F1':>7} {'F2':>7} {'F3':>7} {'Std':>6} "
        f"{'FullS':>7} {'Mode':>10} {'Metric':>7} {'Lv':>4} {'LR':>8} "
        f"{'Train':>5} {'Step':>4} {'Hz':>8}"
    )
    lines.append("-" * 100)

    for i, t in enumerate(trials[:n]):
        if t.value is None:
            continue
        p = t.params
        ua = t.user_attrs
        lines.append(
            f"{i+1:>4} {t.number:>6} {t.value:>+7.3f} "
            f"{ua.get('sharpe_fold1', float('nan')):>+7.3f} "
            f"{ua.get('sharpe_fold2', float('nan')):>+7.3f} "
            f"{ua.get('sharpe_fold3', float('nan')):>+7.3f} "
            f"{ua.get('sharpe_std', float('nan')):>6.3f} "
            f"{ua.get('sharpe_full', float('nan')):>+7.3f} "
            f"{p.get('mode', '?'):>10} "
            f"{p.get('early_stop_metric', '?'):>7} "
            f"{p.get('num_leaves', '?'):>4} "
            f"{p.get('learning_rate', 0):>8.4f} "
            f"{p.get('train_months', '?'):>5} {p.get('step_months', '?'):>4} "
            f"{p.get('label_horizon', '?'):>8}"
        )

    output = "\n".join(lines)
    print(output)
    return output


def print_param_importance(study: optuna.Study) -> str:
    """Print parameter importance and return formatted string."""
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as e:
        msg = f"Could not compute importance: {e}"
        print(msg)
        return msg

    lines = ["\nParameter Importance:", "-" * 40]
    for param, imp in importance.items():
        lines.append(f"  {param:30s}: {imp:.4f}")

    output = "\n".join(lines)
    print(output)
    return output


def save_results_to_json(study: optuna.Study, path: str) -> None:
    """Save study results to JSON for downstream consumption."""
    trials_data = []
    for t in sorted(study.trials, key=lambda t: t.value or -999, reverse=True):
        if t.value is None:
            continue
        trials_data.append({
            "trial_number": t.number,
            "sharpe": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        })

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        importance = {}

    result = {
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_mean_sharpe": study.best_value if study.best_value else None,
        "best_params": study.best_params if study.best_trial else None,
        "param_importance": importance,
        "val_folds": VAL_FOLDS,
        "trials": trials_data,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for LightGBM Walk-Forward")
    parser.add_argument("--n-trials", type=int, default=80, help="Number of trials (default: 80)")
    parser.add_argument("--study-name", default="lgb_hpo_v4", help="Optuna study name")
    parser.add_argument("--storage", default="sqlite:///data/optuna_study.db",
                        help="Optuna storage URL")
    parser.add_argument("--factors", default="data/factors.parquet",
                        help="Path to pre-computed factors.parquet")
    parser.add_argument("--output", default="data/hpo_results.json",
                        help="Output JSON for results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume existing study (don't create new)")
    args = parser.parse_args()

    load_env()
    c, _, _ = build_components()
    storage_db = c.market._storage

    # Pre-load backtest data (shared across all trials)
    logger.info("Pre-loading backtest data for %s ~ %s ...", VAL_BACKTEST_START, VAL_BACKTEST_END)
    bt_data = _load_backtest_data(storage_db, VAL_BACKTEST_START, VAL_BACKTEST_END)
    logger.info("Backtest data ready.")

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    existing = len(study.trials)
    remaining = max(0, args.n_trials - existing) if not args.resume else args.n_trials
    if existing > 0:
        logger.info("Resuming study '%s': %d existing trials, running %d more",
                     args.study_name, existing, remaining)
    else:
        logger.info("Starting new study '%s': %d trials", args.study_name, args.n_trials)
        remaining = args.n_trials

    if remaining <= 0:
        logger.info("Study already has %d trials (target %d). Showing results.",
                     existing, args.n_trials)
    else:
        objective = create_objective(storage_db, bt_data, args.factors)
        study.optimize(objective, n_trials=remaining)

    # Print and save results
    print_top_trials(study)
    print_param_importance(study)
    save_results_to_json(study, args.output)

    # Print best config for easy copy-paste
    if study.best_trial:
        bt = study.best_trial
        ua = bt.user_attrs
        print(f"\n{'='*60}")
        print(f"Best Trial #{bt.number}: Mean Sharpe = {study.best_value:.4f}")
        print(f"  Folds: {ua.get('sharpe_fold1', '?'):.3f} / "
              f"{ua.get('sharpe_fold2', '?'):.3f} / "
              f"{ua.get('sharpe_fold3', '?'):.3f}  "
              f"(std={ua.get('sharpe_std', '?'):.3f})")
        print(f"  Full-period Sharpe: {ua.get('sharpe_full', '?'):.4f}")
        print(f"{'='*60}")
        print(json.dumps(study.best_params, indent=2, default=str))


if __name__ == "__main__":
    main()
