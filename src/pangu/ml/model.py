"""LightGBM model wrapper and Walk-Forward training orchestrator.

Provides:
- LGBModel: fit/predict/save/load with early stopping
- train_walk_forward(): full 17-window Walk-Forward training pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from pangu.ml.dataset import (
    build_window_datasets,
    compute_labels,
    generate_walk_forward_windows,
    load_factor_panel,
)

if TYPE_CHECKING:
    from pangu.data.storage import DataStorage

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict = {
    "objective": "mae",
    "metric": "mae",
    "num_leaves": 15,
    "learning_rate": 0.01,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 200,
    "random_state": 42,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 100


class LGBModel:
    """LightGBM regression model wrapper."""

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Train with early stopping on validation set.

        Returns dict with 'best_iteration' and 'val_mse'.
        """
        n_estimators = self.params.pop("n_estimators", 500)
        self.model = lgb.LGBMRegressor(n_estimators=n_estimators, **self.params)
        self.params["n_estimators"] = n_estimators  # restore

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iter = self.model.best_iteration_
        val_pred = self.model.predict(X_val)
        val_mse = float(np.mean((y_val.values - val_pred) ** 2))

        return {"best_iteration": best_iter, "val_mse": val_mse}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict scores. Returns Series with same index as X."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index, name="score")

    def save(self, path: str) -> None:
        """Save model to LightGBM native text format."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(path)

    @classmethod
    def load(cls, path: str) -> "LGBModel":
        """Load model from file."""
        obj = cls()
        booster = lgb.Booster(model_file=path)
        obj.model = lgb.LGBMRegressor()
        obj.model._Booster = booster
        obj.model.fitted_ = True
        obj.model._n_features = booster.num_feature()
        return obj

    def feature_importance(self) -> pd.Series:
        """Return feature importance (gain-based), sorted descending."""
        if self.model is None:
            raise RuntimeError("No model.")
        imp = self.model.feature_importances_
        names = self.model.feature_name_
        return pd.Series(imp, index=names, name="importance").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Per-window evaluation metrics
# ---------------------------------------------------------------------------

def _compute_ic(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute daily IC and Rank IC, averaged across dates.

    Returns dict with 'ic_mean', 'ic_std', 'rank_ic_mean', 'rank_ic_std'.
    """
    df = pd.DataFrame({"true": y_true.values, "pred": y_pred.values}, index=y_true.index)

    ic_list = []
    rank_ic_list = []
    for date, group in df.groupby(level="date"):
        if len(group) < 5:
            continue
        t, p = group["true"], group["pred"]
        if t.std() < 1e-12 or p.std() < 1e-12:
            continue
        ic, _ = stats.pearsonr(t, p)
        ric, _ = stats.spearmanr(t, p)
        ic_list.append(ic)
        rank_ic_list.append(ric)

    if not ic_list:
        return {"ic_mean": np.nan, "ic_std": np.nan,
                "rank_ic_mean": np.nan, "rank_ic_std": np.nan}

    return {
        "ic_mean": np.mean(ic_list),
        "ic_std": np.std(ic_list),
        "rank_ic_mean": np.mean(rank_ic_list),
        "rank_ic_std": np.std(rank_ic_list),
    }


# ---------------------------------------------------------------------------
# Walk-Forward orchestrator
# ---------------------------------------------------------------------------

def train_walk_forward(
    storage: "DataStorage",
    factors_path: str | None = None,
    model_dir: str = "models",
    output_path: str = "data/score_matrix.parquet",
    params: dict | None = None,
    label_horizon: int = 5,
    train_months: int = 18,
    val_months: int = 3,
    test_months: int = 3,
    first_train_start: str = "2020-01-01",
    last_test_end: str = "2025-12-31",
) -> pd.DataFrame:
    """Execute full Walk-Forward training.

    Steps:
    1. Load/compute factor panel (166 cols) for full date range
    2. Compute N-day excess return labels
    3. For each window:
       a. Build train/val/test datasets (constituent-filtered)
       b. Train LGBModel with early stopping
       c. Predict test scores
       d. Save model
       e. Log metrics (IC, Rank IC, Val MSE)
    4. Concatenate test scores → score_matrix (date × symbol)
    5. Save to parquet

    Returns
    -------
    score_matrix : DataFrame (date × symbol), values = predicted scores.
    """
    windows = generate_walk_forward_windows(
        train_months=train_months,
        val_months=val_months,
        test_months=test_months,
        first_train_start=first_train_start,
        last_test_end=last_test_end,
    )
    global_start = windows[0].train_start
    global_end = windows[-1].test_end

    logger.info(
        "Walk-Forward: %d windows, train=%dmo, test covers %s ~ %s",
        len(windows), train_months, windows[0].test_start, global_end,
    )

    # Determine pool: all historical constituents across entire range
    pool = storage.load_constituents_union(global_start, global_end)
    logger.info("Pool: %d stocks, range: %s ~ %s", len(pool), global_start, global_end)

    # 1. Load factor panel
    logger.info("Loading factor panel...")
    panel = load_factor_panel(storage, pool, global_start, global_end, factors_path)
    if panel.empty:
        raise ValueError("Empty factor panel. Check data availability.")

    # Validate factor coverage vs first training window
    panel_dates = panel.index.get_level_values("date")
    panel_start = panel_dates.min().strftime("%Y-%m-%d")
    if panel_start > windows[0].train_start:
        logger.warning(
            "Factor data starts at %s, but first train window starts at %s. "
            "Early training samples will be reduced.",
            panel_start, windows[0].train_start,
        )
    logger.info("Factor panel: %s rows × %d cols", f"{panel.shape[0]:,}", panel.shape[1])

    # 2. Compute labels
    logger.info("Computing %d-day excess return labels...", label_horizon)
    labels = compute_labels(storage, pool, global_start, global_end, horizon=label_horizon)
    n_valid = labels.notna().sum()
    logger.info("Labels: %s valid / %s total", f"{n_valid:,}", f"{len(labels):,}")

    # 3. Walk-Forward training
    all_test_scores: list[pd.Series] = []
    window_metrics: list[dict] = []

    for w in windows:
        logger.info(
            "Window %02d: train %s~%s | val %s~%s | test %s~%s",
            w.window_id, w.train_start, w.train_end,
            w.val_start, w.val_end, w.test_start, w.test_end,
        )

        datasets = build_window_datasets(panel, labels, w, storage)
        X_train, y_train = datasets["train"]
        X_val, y_val = datasets["val"]
        X_test, y_test = datasets["test"]

        if X_train.empty or X_val.empty:
            logger.warning("  Window %02d: insufficient data, skipping", w.window_id)
            continue

        logger.info(
            "  Samples: train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        # Train
        model = LGBModel(params)
        fit_info = model.fit(X_train, y_train, X_val, y_val)

        # Predict test
        if not X_test.empty:
            test_scores = model.predict(X_test)
            all_test_scores.append(test_scores)

            # Metrics
            ic_metrics = _compute_ic(y_test, test_scores)
        else:
            ic_metrics = {"ic_mean": np.nan, "rank_ic_mean": np.nan}

        metrics = {
            "window": w.window_id,
            "best_iter": fit_info["best_iteration"],
            "val_mse": fit_info["val_mse"],
            **ic_metrics,
        }
        window_metrics.append(metrics)

        logger.info(
            "  best_iter=%d  val_mse=%.6f  IC=%.4f  RankIC=%.4f",
            metrics["best_iter"], metrics["val_mse"],
            metrics.get("ic_mean", float("nan")),
            metrics.get("rank_ic_mean", float("nan")),
        )

        # Save model
        model_path = str(Path(model_dir) / f"wf_window_{w.window_id:02d}.txt")
        model.save(model_path)

    # 4. Assemble score matrix
    if not all_test_scores:
        raise ValueError("No test scores produced. Check data coverage.")

    combined = pd.concat(all_test_scores)
    # Pivot to (date × symbol) — compatible with BacktestEngine.run(scores=...)
    score_matrix = combined.unstack(level="symbol")
    score_matrix.index = pd.to_datetime(score_matrix.index)
    score_matrix = score_matrix.sort_index()

    # 5. Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    score_matrix.to_parquet(output_path)
    logger.info("Score matrix saved: %s (%d days × %d stocks)",
                output_path, score_matrix.shape[0], score_matrix.shape[1])

    # Summary
    metrics_df = pd.DataFrame(window_metrics)
    logger.info("\n=== Walk-Forward Summary ===")
    logger.info("Windows: %d", len(metrics_df))
    logger.info("Mean IC: %.4f ± %.4f", metrics_df["ic_mean"].mean(), metrics_df["ic_mean"].std())
    logger.info("Mean Rank IC: %.4f ± %.4f",
                metrics_df["rank_ic_mean"].mean(), metrics_df["rank_ic_mean"].std())
    logger.info("Mean Val MSE: %.6f", metrics_df["val_mse"].mean())

    return score_matrix
