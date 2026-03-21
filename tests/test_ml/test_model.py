"""Tests for ml.model — LGBModel fit/predict/save/load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.ml.model import LGBModel, MIN_ITERATIONS, _compute_ic

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Generate synthetic regression data with known signal."""
    rng = np.random.default_rng(42)
    n_features = 10

    dates = pd.bdate_range("2024-01-01", periods=100)
    symbols = [f"S{i:03d}" for i in range(20)]

    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    X = pd.DataFrame(
        rng.standard_normal((len(idx), n_features)).astype("float32"),
        index=idx,
        columns=[f"F{i}" for i in range(n_features)],
    )
    # y = linear combination + noise (strong signal for test reliability)
    true_weights = rng.standard_normal(n_features) * 2
    y = pd.Series(
        X.values @ true_weights + rng.normal(0, 0.3, len(idx)),
        index=idx,
        name="label",
    )
    return X, y


# Light params for unit tests (small dataset)
_TEST_PARAMS = {
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 20,
}


@pytest.fixture
def train_val_split(synthetic_data):
    X, y = synthetic_data
    dates = X.index.get_level_values("date").unique().sort_values()
    train_dates = dates[:35]
    val_dates = dates[35:]
    X_train = X.loc[train_dates]
    y_train = y.loc[train_dates]
    X_val = X.loc[val_dates]
    y_val = y.loc[val_dates]
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# LGBModel
# ---------------------------------------------------------------------------

class TestLGBModel:
    def test_fit_returns_metrics(self, train_val_split):
        X_train, y_train, X_val, y_val = train_val_split
        model = LGBModel(_TEST_PARAMS)
        info = model.fit(X_train, y_train, X_val, y_val)
        assert "best_iteration" in info
        assert "val_mse" in info
        assert info["best_iteration"] > 0
        assert info["val_mse"] >= 0

    def test_predict_shape(self, train_val_split):
        X_train, y_train, X_val, y_val = train_val_split
        model = LGBModel(_TEST_PARAMS)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        assert len(preds) == len(X_val)
        assert preds.index.equals(X_val.index)

    def test_predict_not_constant(self, train_val_split):
        """Predictions should vary (model learned something)."""
        X_train, y_train, X_val, y_val = train_val_split
        model = LGBModel(_TEST_PARAMS)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        assert preds.std() > 0.01

    def test_save_load_roundtrip(self, train_val_split, tmp_path):
        X_train, y_train, X_val, y_val = train_val_split
        model = LGBModel(_TEST_PARAMS)
        model.fit(X_train, y_train, X_val, y_val)
        preds_before = model.predict(X_val)

        path = str(tmp_path / "model.txt")
        model.save(path)

        loaded = LGBModel.load(path)
        preds_after = loaded.predict(X_val)

        pd.testing.assert_series_equal(preds_before, preds_after, check_names=False)

    def test_feature_importance(self, train_val_split):
        X_train, y_train, X_val, y_val = train_val_split
        model = LGBModel(_TEST_PARAMS)
        model.fit(X_train, y_train, X_val, y_val)
        imp = model.feature_importance()
        assert len(imp) == X_train.shape[1]
        assert imp.sum() > 0

    def test_predict_before_fit_raises(self):
        model = LGBModel(_TEST_PARAMS)
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(pd.DataFrame({"a": [1]}))

    def test_custom_params(self, train_val_split):
        X_train, y_train, X_val, y_val = train_val_split
        model = LGBModel({"num_leaves": 15, "learning_rate": 0.1})
        assert model.params["num_leaves"] == 15
        assert model.params["learning_rate"] == 0.1
        info = model.fit(X_train, y_train, X_val, y_val)
        assert info["val_mse"] >= 0

    def test_handles_nan_features(self, train_val_split):
        """LightGBM should handle NaN features natively."""
        X_train, y_train, X_val, y_val = train_val_split
        # Inject NaN
        X_train_nan = X_train.copy()
        X_train_nan.iloc[:50, 0] = np.nan
        model = LGBModel(_TEST_PARAMS)
        info = model.fit(X_train_nan, y_train, X_val, y_val)
        assert info["val_mse"] >= 0
        preds = model.predict(X_val)
        assert preds.notna().all()

    def test_min_iterations_protection(self):
        """When early stopping triggers before MIN_ITERATIONS, model retrains."""
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2024-01-01", periods=50)
        symbols = [f"S{i:03d}" for i in range(10)]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        # Pure noise — early stopping should fire very quickly
        X = pd.DataFrame(
            rng.standard_normal((len(idx), 5)).astype("float32"),
            index=idx, columns=[f"F{i}" for i in range(5)],
        )
        y = pd.Series(rng.standard_normal(len(idx)), index=idx, name="label")

        split = 25 * 10
        model = LGBModel({"num_leaves": 4, "learning_rate": 0.05, "min_child_samples": 5})
        info = model.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        # Protection should have kicked in: exactly MIN_ITERATIONS
        assert info["best_iteration"] == MIN_ITERATIONS
        assert model.model is not None

    def test_no_retrain_when_all_estimators_used(self, train_val_split):
        """When model uses all n_estimators (no early stop), don't retrain."""
        X_train, y_train, X_val, y_val = train_val_split
        # Strong signal + small n_estimators: early stopping won't fire before 30
        model = LGBModel({
            "num_leaves": 15, "learning_rate": 0.05,
            "min_child_samples": 20, "n_estimators": 30,
        })
        info = model.fit(X_train, y_train, X_val, y_val)
        # best_iteration == 30 (completed, no early stop) so no retrain
        assert info["best_iteration"] == 30


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

class TestComputeIC:
    def test_perfect_correlation(self):
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B", "C", "D", "E"]],
            names=["date", "symbol"],
        )
        y = pd.Series(range(10), index=idx, dtype=float)
        metrics = _compute_ic(y, y)
        assert abs(metrics["ic_mean"] - 1.0) < 0.01
        assert abs(metrics["rank_ic_mean"] - 1.0) < 0.01

    def test_zero_correlation(self):
        rng = np.random.default_rng(42)
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(pd.bdate_range("2024-01-01", periods=20)),
             [f"S{i}" for i in range(10)]],
            names=["date", "symbol"],
        )
        y_true = pd.Series(rng.standard_normal(200), index=idx)
        y_pred = pd.Series(rng.standard_normal(200), index=idx)
        metrics = _compute_ic(y_true, y_pred)
        # Should be close to 0 with random data
        assert abs(metrics["ic_mean"]) < 0.3
