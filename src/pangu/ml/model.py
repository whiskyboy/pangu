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
    compute_groups,
    compute_labels,
    compute_time_decay_weights,
    discretize_labels,
    generate_walk_forward_windows,
    load_factor_panel,
)

if TYPE_CHECKING:
    from pangu.data.storage import DataStorage

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict = {
    "objective": "mae",
    "metric": "mae",
    "num_leaves": 31,
    "learning_rate": 0.02,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 100,
    "random_state": 42,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 200
MIN_ITERATIONS = 50
VALID_EARLY_STOP_METRICS = ("mae", "rankic")

RANKER_DEFAULT_PARAMS: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10, 30],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 100,
    "lambdarank_truncation_level": 30,
    "lambdarank_norm": True,
    "random_state": 42,
    "verbosity": -1,
}


def _make_rankic_eval_metric(val_dates: np.ndarray):
    """Create a RankIC evaluation metric for LightGBM early stopping.

    Groups predictions by date and computes mean Spearman rank correlation.
    This aligns early stopping with ranking quality rather than pointwise MAE.
    """
    unique_dates = np.unique(val_dates)
    date_masks = []
    for d in unique_dates:
        mask = val_dates == d
        if mask.sum() >= 5:
            date_masks.append(mask)

    def rankic_metric(y_true, y_pred):
        rank_ics = []
        for mask in date_masks:
            t = y_true[mask]
            p = y_pred[mask]
            if np.std(t) < 1e-12 or np.std(p) < 1e-12:
                continue
            ric, _ = stats.spearmanr(t, p)
            if not np.isnan(ric):
                rank_ics.append(ric)
        mean_ric = float(np.mean(rank_ics)) if rank_ics else 0.0
        return "rankic", mean_ric, True

    return rankic_metric


class LGBModel:
    """LightGBM regression model wrapper."""

    def __init__(self, params: dict | None = None, early_stop_metric: str = "mae"):
        if early_stop_metric not in VALID_EARLY_STOP_METRICS:
            raise ValueError(f"early_stop_metric must be one of {VALID_EARLY_STOP_METRICS}")
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stop_metric = early_stop_metric
        self.model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> dict:
        """Train with early stopping on validation set.

        Returns dict with 'best_iteration' and 'val_mse'.
        When early_stop_metric='rankic', early stopping uses daily mean
        Spearman rank correlation instead of MAE.

        Parameters
        ----------
        sample_weight : Series or None
            Per-sample weights for training data (e.g. time-decay).
            Validation set is always unweighted to keep evaluation objective.
        """
        n_estimators = self.params.pop("n_estimators", 500)

        fit_kwargs: dict = {}
        model_params = dict(self.params)

        if self.early_stop_metric == "rankic":
            val_dates = y_val.index.get_level_values("date").values
            fit_kwargs["eval_metric"] = _make_rankic_eval_metric(val_dates)
            model_params["metric"] = "None"

        self.model = lgb.LGBMRegressor(n_estimators=n_estimators, **model_params)
        self.params["n_estimators"] = n_estimators  # restore

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
            **fit_kwargs,
        )

        best_iter = self.model.best_iteration_

        # Guard against pathological early stopping: if model stopped before
        # MIN_ITERATIONS, retrain without early stopping using MIN_ITERATIONS.
        # Only trigger when early stopping fired (best_iter < n_estimators).
        if best_iter < MIN_ITERATIONS and best_iter < n_estimators:
            logger.info(
                "  Early stop at %d < MIN_ITERATIONS=%d, retraining with %d rounds",
                best_iter, MIN_ITERATIONS, MIN_ITERATIONS,
            )
            retrain_params = {k: v for k, v in model_params.items() if k != "n_estimators"}
            self.model = lgb.LGBMRegressor(n_estimators=MIN_ITERATIONS, **retrain_params)
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)],
                **fit_kwargs,
            )
            best_iter = MIN_ITERATIONS

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


class LGBRankerModel:
    """LightGBM LambdaRank model for stock ranking.

    Uses LGBMRanker with NDCG optimization. Requires integer relevance labels
    (produced by ``discretize_labels()``) and per-day group sizes
    (produced by ``compute_groups()``).
    """

    def __init__(self, params: dict | None = None):
        self.params = {**RANKER_DEFAULT_PARAMS, **(params or {})}
        self.model: lgb.LGBMRanker | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: list[int],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        groups_val: list[int],
    ) -> dict:
        """Train with early stopping on validation NDCG.

        Returns dict with 'best_iteration' and 'val_ndcg'.
        """
        n_estimators = self.params.pop("n_estimators", 2000)
        self.model = lgb.LGBMRanker(n_estimators=n_estimators, **self.params)
        self.params["n_estimators"] = n_estimators  # restore

        self.model.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=[(X_val, y_val)],
            eval_group=[groups_val],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iter = self.model.best_iteration_

        # Guard against pathological early stopping
        if best_iter < MIN_ITERATIONS and best_iter < n_estimators:
            logger.info(
                "  Early stop at %d < MIN_ITERATIONS=%d, retraining with %d rounds",
                best_iter, MIN_ITERATIONS, MIN_ITERATIONS,
            )
            retrain_params = {k: v for k, v in self.params.items() if k != "n_estimators"}
            self.model = lgb.LGBMRanker(n_estimators=MIN_ITERATIONS, **retrain_params)
            self.model.fit(
                X_train, y_train,
                group=groups_train,
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            best_iter = MIN_ITERATIONS

        return {"best_iteration": best_iter, "val_ndcg": float("nan")}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict ranking scores. Returns Series with same index as X."""
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
    def load(cls, path: str) -> "LGBRankerModel":
        """Load model from file."""
        obj = cls()
        booster = lgb.Booster(model_file=path)
        obj.model = lgb.LGBMRanker()
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
    normalize_label: bool = False,
    mode: str = "regression",
    n_bins: int = 10,
    early_stop_metric: str = "mae",
    time_decay_halflife: int = 0,
) -> pd.DataFrame:
    """Execute full Walk-Forward training.

    Steps:
    1. Load/compute factor panel (166 cols) for full date range
    2. Compute N-day excess return labels (with optional CS z-score)
    3. For each window:
       a. Build train/val/test datasets (constituent-filtered)
       b. Train LGBModel or LGBRankerModel
       c. Predict test scores
       d. Save model
       e. Log metrics (IC, Rank IC, Val MSE)
    4. Concatenate test scores → score_matrix (date × symbol)
    5. Save to parquet

    Parameters
    ----------
    normalize_label : bool
        If True, apply Qlib-style cross-sectional z-score to labels
        before training. Disabled by default.
    mode : str
        Training mode: "regression" (default, MAE objective) or
        "ranking" (LambdaRank NDCG objective). Ranking mode discretizes
        labels into n_bins relevance grades and uses LGBMRanker.
    n_bins : int
        Number of relevance grade bins for ranking mode (default 10 = decile).
        Ignored when mode="regression".
    early_stop_metric : str
        Metric for early stopping in regression mode: "mae" (default,
        built-in MAE) or "rankic" (custom daily Spearman rank correlation).
        RankIC aligns stopping with ranking quality rather than pointwise error.
        Ignored when mode="ranking".
    time_decay_halflife : int
        Half-life in trading days for exponential time-decay sample weights.
        0 = no decay (uniform weights, default). Typical values:
        40 (~2 months), 80 (~4 months), 120 (~6 months).
        Only applied to training samples; validation is always unweighted.

    Returns
    -------
    score_matrix : DataFrame (date × symbol), values = predicted scores.
    """
    if mode not in ("regression", "ranking"):
        raise ValueError(f"mode must be 'regression' or 'ranking', got '{mode}'")

    is_ranking = mode == "ranking"
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
        "Walk-Forward: %d windows, train=%dmo, mode=%s, early_stop=%s, "
        "time_decay_halflife=%dd, test covers %s ~ %s",
        len(windows), train_months, mode, early_stop_metric,
        time_decay_halflife, windows[0].test_start, global_end,
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
    labels = compute_labels(
        storage, pool, global_start, global_end,
        horizon=label_horizon, normalize=normalize_label,
    )
    n_valid = labels.notna().sum()
    logger.info("Labels: %s valid / %s total (normalize=%s)",
                f"{n_valid:,}", f"{len(labels):,}", normalize_label)

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

        # Compute sample weights (time decay)
        train_weights = None
        if time_decay_halflife > 0:
            train_weights = compute_time_decay_weights(X_train.index, time_decay_halflife)
            logger.info(
                "  Time decay: halflife=%dd, weight range [%.3f, %.3f]",
                time_decay_halflife, train_weights.min(), train_weights.max(),
            )

        # Train
        if is_ranking:
            # Discretize labels for LambdaRank (per-day percentile → 0..n_bins-1)
            y_train_rank = discretize_labels(y_train, n_bins=n_bins)
            y_val_rank = discretize_labels(y_val, n_bins=n_bins)
            groups_train = compute_groups(X_train.index)
            groups_val = compute_groups(X_val.index)
            model = LGBRankerModel(params)
            fit_info = model.fit(
                X_train, y_train_rank, groups_train,
                X_val, y_val_rank, groups_val,
            )
        else:
            model = LGBModel(params, early_stop_metric=early_stop_metric)
            fit_info = model.fit(X_train, y_train, X_val, y_val, sample_weight=train_weights)

        # Predict test (IC always computed against original continuous labels)
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
            **ic_metrics,
        }
        if is_ranking:
            metrics["val_ndcg"] = fit_info.get("val_ndcg", float("nan"))
        else:
            metrics["val_mse"] = fit_info.get("val_mse", float("nan"))
        window_metrics.append(metrics)

        val_metric_str = (
            f"val_ndcg={metrics.get('val_ndcg', float('nan')):.6f}"
            if is_ranking else
            f"val_mse={metrics.get('val_mse', float('nan')):.6f}"
        )
        logger.info(
            "  best_iter=%d  %s  IC=%.4f  RankIC=%.4f",
            metrics["best_iter"], val_metric_str,
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
    logger.info("\n=== Walk-Forward Summary (%s) ===", mode)
    logger.info("Windows: %d", len(metrics_df))
    logger.info("Mean IC: %.4f ± %.4f", metrics_df["ic_mean"].mean(), metrics_df["ic_mean"].std())
    logger.info("Mean Rank IC: %.4f ± %.4f",
                metrics_df["rank_ic_mean"].mean(), metrics_df["rank_ic_mean"].std())
    if is_ranking:
        if "val_ndcg" in metrics_df.columns:
            logger.info("Mean Val NDCG: %.6f", metrics_df["val_ndcg"].mean())
    elif "val_mse" in metrics_df.columns:
        logger.info("Mean Val MSE: %.6f", metrics_df["val_mse"].mean())

    return score_matrix
