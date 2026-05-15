"""Tests for MLScorer — model discovery and ensemble scoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pangu.ml.scorer import _MODEL_PATTERN, MLScorer

# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestModelPattern:
    def test_standard_name(self):
        m = _MODEL_PATTERN.match("wf_window_16_seed0.txt")
        assert m is not None
        assert m.group(1) == "16"
        assert m.group(2) == "0"

    def test_high_window(self):
        m = _MODEL_PATTERN.match("wf_window_99_seed4.txt")
        assert m is not None
        assert int(m.group(1)) == 99
        assert int(m.group(2)) == 4

    def test_non_match(self):
        assert _MODEL_PATTERN.match("random_file.txt") is None
        assert _MODEL_PATTERN.match("wf_window_abc_seed0.txt") is None


# ---------------------------------------------------------------------------
# Scorer tests (with real LightGBM mini models)
# ---------------------------------------------------------------------------


def _make_mini_model(path: Path, n_features: int = 10) -> None:
    """Train and save a tiny LightGBM model for testing."""
    import lightgbm as lgb

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(50, n_features), columns=[f"f{i}" for i in range(n_features)])
    y = rng.randn(50)
    model = lgb.LGBMRegressor(n_estimators=5, num_leaves=4, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(str(path))


class TestMLScorer:
    def test_no_model_dir_raises(self):
        db = MagicMock()
        with pytest.raises(FileNotFoundError, match="not found"):
            MLScorer(model_dir="/nonexistent/path", db=db)

    def test_empty_dir_raises(self, tmp_path):
        db = MagicMock()
        with pytest.raises(FileNotFoundError, match="No model files"):
            MLScorer(model_dir=str(tmp_path), db=db)

    def test_discovers_latest_window(self, tmp_path):
        """Should discover and load models from the latest window only."""
        # Create models for window 10 and 16
        for win in (10, 16):
            for seed in range(3):
                _make_mini_model(tmp_path / f"wf_window_{win}_seed{seed}.txt")

        db = MagicMock()
        scorer = MLScorer(model_dir=str(tmp_path), db=db)

        assert scorer.window_id == 16
        assert scorer.n_models == 3

    def test_reload_picks_new_window(self, tmp_path):
        """After adding new model files, reload() should discover them."""
        # Start with window 5
        for seed in range(2):
            _make_mini_model(tmp_path / f"wf_window_5_seed{seed}.txt")

        db = MagicMock()
        scorer = MLScorer(model_dir=str(tmp_path), db=db)
        assert scorer.window_id == 5
        assert scorer.n_models == 2

        # Add window 6
        for seed in range(3):
            _make_mini_model(tmp_path / f"wf_window_6_seed{seed}.txt")

        scorer.reload()
        assert scorer.window_id == 6
        assert scorer.n_models == 3

    def test_score_returns_series(self, tmp_path):
        """score() should return a Series indexed by symbol."""
        n_features = 191  # Match Alpha158 count
        for seed in range(2):
            _make_mini_model(tmp_path / f"wf_window_0_seed{seed}.txt", n_features=n_features)

        # Mock db and engine: return fake factors
        db = MagicMock()
        scorer = MLScorer(model_dir=str(tmp_path), db=db)

        rng = np.random.RandomState(0)
        fake_factors = pd.DataFrame(
            rng.randn(5, n_features),
            index=["600000", "600001", "600002", "000001", "000002"],
            columns=[f"f{i}" for i in range(n_features)],
        )
        # Patch compute_latest to return our fake factors
        scorer._engine = MagicMock()
        scorer._engine.compute_latest.return_value = fake_factors

        result = scorer.score("2025-01-01", list(fake_factors.index))
        assert isinstance(result, pd.Series)
        assert result.name == "score"
        assert len(result) == 5
        assert set(result.index) == set(fake_factors.index)

    def test_score_empty_factors(self, tmp_path):
        """score() should return empty Series when compute_latest returns empty."""
        for seed in range(2):
            _make_mini_model(tmp_path / f"wf_window_0_seed{seed}.txt")

        db = MagicMock()
        scorer = MLScorer(model_dir=str(tmp_path), db=db)
        scorer._engine = MagicMock()
        scorer._engine.compute_latest.return_value = pd.DataFrame()

        result = scorer.score("2025-01-01", ["600000"])
        assert result.empty
