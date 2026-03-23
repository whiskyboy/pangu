"""Tests for model_evaluator module."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from pangu.ml.model_evaluator import (
    MIN_TREES_THRESHOLD,
    _compute_feature_drift,
    _compute_global_importance,
    _compute_per_window_summary,
    _compute_zero_importance,
    _load_window_boosters,
    evaluate_models,
    format_model_report,
)


@pytest.fixture
def trained_boosters(tmp_path) -> list[tuple[int, list[lgb.Booster]]]:
    """Train 3 small LightGBM models and save as wf_window files.

    Returns the loaded (window_id, [Booster]) list and also writes model files
    to tmp_path for load tests.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 10
    feature_names = [f"F{i}" for i in range(n_features)]
    results: list[tuple[int, list[lgb.Booster]]] = []

    for wid in range(1, 4):
        X = rng.normal(size=(n_samples, n_features))
        # Weight shifts across windows to create feature drift
        weights = rng.dirichlet(np.ones(n_features))
        y = X @ weights + rng.normal(0, 0.1, n_samples)

        ds_train = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=False)
        params = {
            "objective": "regression",
            "num_leaves": 8,
            "learning_rate": 0.1,
            "verbose": -1,
            "min_data_in_leaf": 5,
        }
        booster = lgb.train(params, ds_train, num_boost_round=60)

        path = tmp_path / f"wf_window_{wid:02d}.txt"
        booster.save_model(str(path))
        results.append((wid, [booster]))

    return results


@pytest.fixture
def model_dir(trained_boosters, tmp_path) -> str:
    """Return path to a directory with saved model files."""
    # Files are already written by trained_boosters fixture
    return str(tmp_path)


class TestLoadWindowBoosters:
    def test_loads_correct_count(self, model_dir: str) -> None:
        windows = _load_window_boosters(model_dir)
        assert len(windows) == 3

    def test_sorted_by_window_id(self, model_dir: str) -> None:
        windows = _load_window_boosters(model_dir)
        ids = [wid for wid, _ in windows]
        assert ids == sorted(ids)

    def test_skips_non_model_files(self, model_dir: str, tmp_path) -> None:
        (tmp_path / "notes.txt").write_text("not a model")
        (tmp_path / "random.json").write_text("{}")
        windows = _load_window_boosters(model_dir)
        assert len(windows) == 3

    def test_empty_dir(self, tmp_path) -> None:
        windows = _load_window_boosters(str(tmp_path))
        assert windows == []

    def test_missing_dir(self) -> None:
        windows = _load_window_boosters("/nonexistent/path")
        assert windows == []

    def test_skips_corrupted_file(self, tmp_path) -> None:
        (tmp_path / "wf_window_99.txt").write_text("corrupted data")
        windows = _load_window_boosters(str(tmp_path))
        assert all(wid != 99 for wid, _ in windows)

    def test_multi_seed_grouped(self, model_dir: str, trained_boosters, tmp_path) -> None:
        """Multi-seed files should be grouped by window, not deduplicated."""
        import shutil
        for wid in range(1, 4):
            src = tmp_path / f"wf_window_{wid:02d}.txt"
            dst = tmp_path / f"wf_window_{wid:02d}_seed1.txt"
            shutil.copy(str(src), str(dst))
        windows = _load_window_boosters(model_dir)
        assert len(windows) == 3
        for _wid, boosters in windows:
            assert len(boosters) == 2  # original + seed1

    def test_seed_only_files(self, trained_boosters, tmp_path) -> None:
        """When only seed-suffixed files exist, groups by window."""
        import shutil
        seed_dir = tmp_path / "seeds_only"
        seed_dir.mkdir()
        for wid in range(1, 4):
            src = tmp_path / f"wf_window_{wid:02d}.txt"
            for seed in range(3):
                dst = seed_dir / f"wf_window_{wid:02d}_seed{seed}.txt"
                shutil.copy(str(src), str(dst))
        windows = _load_window_boosters(str(seed_dir))
        assert len(windows) == 3
        for _wid, boosters in windows:
            assert len(boosters) == 3


class TestGlobalImportance:
    def test_top_n_limit(self, trained_boosters) -> None:
        result = _compute_global_importance(trained_boosters, top_n=5)
        assert len(result["top_features"]) == 5

    def test_sorted_descending(self, trained_boosters) -> None:
        result = _compute_global_importance(trained_boosters, top_n=10)
        pcts = [f["mean_pct"] for f in result["top_features"]]
        assert pcts == sorted(pcts, reverse=True)

    def test_percentages_reasonable(self, trained_boosters) -> None:
        result = _compute_global_importance(trained_boosters, top_n=10)
        for f in result["top_features"]:
            assert 0 <= f["mean_pct"] <= 100
            assert f["std_pct"] >= 0

    def test_windows_used_count(self, trained_boosters) -> None:
        result = _compute_global_importance(trained_boosters, top_n=10)
        for f in result["top_features"]:
            assert 1 <= f["windows_used"] <= f["windows_total"]


class TestPerWindowSummary:
    def test_correct_window_count(self, trained_boosters) -> None:
        summaries = _compute_per_window_summary(trained_boosters)
        assert len(summaries) == 3

    def test_has_tree_counts(self, trained_boosters) -> None:
        summaries = _compute_per_window_summary(trained_boosters)
        for s in summaries:
            assert s["num_trees"] > 0
            assert isinstance(s["top5_features"], list)

    def test_underfitting_flag(self, trained_boosters) -> None:
        summaries = _compute_per_window_summary(trained_boosters)
        for s in summaries:
            assert s["underfitting"] == (s["num_trees"] < MIN_TREES_THRESHOLD)

    def test_multiseed_tree_range(self, trained_boosters) -> None:
        """Multi-seed windows should report n_seeds and tree count range."""
        multiseed = [(wid, boosters * 2) for wid, boosters in trained_boosters]
        summaries = _compute_per_window_summary(multiseed)
        for s in summaries:
            assert s["n_seeds"] == 2
            assert s["num_trees"] == s["num_trees_max"]  # identical boosters


class TestFeatureDrift:
    def test_pair_count(self, trained_boosters) -> None:
        result = _compute_feature_drift(trained_boosters)
        assert len(result["pairs"]) == 2  # 3 windows → 2 pairs

    def test_jaccard_range(self, trained_boosters) -> None:
        result = _compute_feature_drift(trained_boosters)
        for p in result["pairs"]:
            assert 0.0 <= p["jaccard"] <= 1.0
        assert 0.0 <= result["mean_jaccard"] <= 1.0

    def test_identical_models_jaccard_one(self, trained_boosters) -> None:
        _, booster = trained_boosters[0]
        same = [(1, booster), (2, booster)]
        result = _compute_feature_drift(same)
        assert result["pairs"][0]["jaccard"] == pytest.approx(1.0)

    def test_single_window_no_drift(self, trained_boosters) -> None:
        result = _compute_feature_drift(trained_boosters[:1])
        assert result["pairs"] == []
        assert np.isnan(result["mean_jaccard"])


class TestZeroImportance:
    def test_no_always_zero_in_trained(self, trained_boosters) -> None:
        result = _compute_zero_importance(trained_boosters)
        # With 60 rounds and 10 features, all should be used
        assert result["n_windows"] == 3


class TestEvaluateModels:
    def test_returns_all_sections(self, model_dir: str) -> None:
        results = evaluate_models(model_dir, top_n=5)
        assert "global_importance" in results
        assert "per_window" in results
        assert "feature_drift" in results
        assert "zero_importance" in results

    def test_empty_dir_returns_empty(self, tmp_path) -> None:
        empty = str(tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        results = evaluate_models(empty)
        assert results["per_window"] == []


class TestFormatModelReport:
    def test_produces_string(self, model_dir: str) -> None:
        results = evaluate_models(model_dir, top_n=5)
        report = format_model_report(results)
        assert isinstance(report, str)

    def test_contains_all_sections(self, model_dir: str) -> None:
        results = evaluate_models(model_dir, top_n=5)
        report = format_model_report(results)
        assert "GLOBAL FEATURE IMPORTANCE" in report
        assert "PER-WINDOW SUMMARY" in report
        assert "FEATURE DRIFT" in report
        assert "ZERO-IMPORTANCE" in report
        assert "Interpretation" in report
