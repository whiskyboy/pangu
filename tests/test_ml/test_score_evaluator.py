"""Tests for score_evaluator module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.ml.score_evaluator import (
    _compute_discrimination,
    _compute_rank_stability,
    _compute_stability,
    evaluate_scores,
    format_report,
)


@pytest.fixture
def stable_scores() -> pd.DataFrame:
    """Score matrix where rankings barely change — high stability."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=60)
    symbols = [f"S{i:03d}" for i in range(50)]

    # Each stock has a slow random walk (persistent signal + small daily drift)
    base = np.linspace(-0.05, 0.05, len(symbols))
    data = np.zeros((len(dates), len(symbols)))
    data[0] = base
    for t in range(1, len(dates)):
        data[t] = data[t - 1] + rng.normal(0, 0.001, len(symbols))

    return pd.DataFrame(data, index=dates, columns=symbols)


@pytest.fixture
def noisy_scores() -> pd.DataFrame:
    """Score matrix where rankings change wildly — low stability."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2024-01-01", periods=60)
    symbols = [f"S{i:03d}" for i in range(50)]

    # Pure noise, no persistent signal
    data = rng.normal(0, 0.01, (len(dates), len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


class TestComputeDiscrimination:
    def test_keys_present(self, stable_scores: pd.DataFrame) -> None:
        result = _compute_discrimination(stable_scores, [10, 30])
        assert "cross_sectional_std_mean" in result
        assert "cross_sectional_std_median" in result
        assert "p90_p10_spread_mean" in result
        assert "top10_boundary_margin_mean" in result
        assert "top30_boundary_margin_mean" in result

    def test_spread_positive(self, stable_scores: pd.DataFrame) -> None:
        result = _compute_discrimination(stable_scores, [10])
        assert result["cross_sectional_std_mean"] > 0
        assert result["p90_p10_spread_mean"] > 0
        assert result["top10_boundary_margin_mean"] > 0

    def test_wider_spread_than_noisy(self, stable_scores: pd.DataFrame, noisy_scores: pd.DataFrame) -> None:
        # stable_scores has wider deliberate spread (linspace -0.05 to 0.05)
        s = _compute_discrimination(stable_scores, [10])
        n = _compute_discrimination(noisy_scores, [10])
        assert s["cross_sectional_std_mean"] > n["cross_sectional_std_mean"]


class TestComputeStability:
    def test_keys_present(self, stable_scores: pd.DataFrame) -> None:
        result = _compute_stability(stable_scores)
        assert "score_autocorr_1d" in result
        assert "score_autocorr_5d" in result
        assert "temporal_std_to_cs_std_ratio" in result

    def test_stable_has_high_autocorr(self, stable_scores: pd.DataFrame) -> None:
        result = _compute_stability(stable_scores)
        assert result["score_autocorr_1d"] > 0.9

    def test_noisy_has_low_autocorr(self, noisy_scores: pd.DataFrame) -> None:
        result = _compute_stability(noisy_scores)
        assert result["score_autocorr_1d"] < 0.3

    def test_stable_has_low_temporal_ratio(self, stable_scores: pd.DataFrame) -> None:
        result = _compute_stability(stable_scores)
        # Stable: per-stock temporal noise is small vs cross-sectional spread
        assert result["temporal_std_to_cs_std_ratio"] < 0.5


class TestComputeRankStability:
    def test_keys_present(self, stable_scores: pd.DataFrame) -> None:
        result = _compute_rank_stability(stable_scores, [10])
        assert "daily_top10_overlap_mean" in result
        assert "daily_top10_jaccard_mean" in result
        assert "weekly_top10_overlap_mean" in result

    def test_stable_high_overlap(self, stable_scores: pd.DataFrame) -> None:
        result = _compute_rank_stability(stable_scores, [10])
        assert result["daily_top10_overlap_mean"] > 0.8

    def test_noisy_low_overlap(self, noisy_scores: pd.DataFrame) -> None:
        result = _compute_rank_stability(noisy_scores, [10])
        assert result["daily_top10_overlap_mean"] < 0.5


class TestEvaluateScores:
    def test_returns_all_sections(self, stable_scores: pd.DataFrame) -> None:
        results = evaluate_scores(stable_scores, top_ns=[10])
        assert "discrimination" in results
        assert "stability" in results
        assert "rank_stability" in results

    def test_default_top_ns(self, stable_scores: pd.DataFrame) -> None:
        results = evaluate_scores(stable_scores)
        r = results["rank_stability"]
        assert "weekly_top10_overlap_mean" in r
        assert "weekly_top30_overlap_mean" in r
        assert "weekly_top50_overlap_mean" in r

    def test_handles_sparse_matrix(self) -> None:
        """Score matrix with many NaNs should not crash."""
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2024-01-01", periods=30)
        symbols = [f"S{i:03d}" for i in range(20)]
        data = rng.normal(0, 0.01, (len(dates), len(symbols)))
        # Set 40% to NaN
        mask = rng.random(data.shape) < 0.4
        data[mask] = np.nan
        scores = pd.DataFrame(data, index=dates, columns=symbols)

        results = evaluate_scores(scores, top_ns=[5])
        assert results["discrimination"]["cross_sectional_std_mean"] > 0


class TestFormatReport:
    def test_produces_string(self, stable_scores: pd.DataFrame) -> None:
        results = evaluate_scores(stable_scores, top_ns=[10])
        report = format_report(results)
        assert isinstance(report, str)
        assert "DISCRIMINATION" in report
        assert "STABILITY" in report
        assert "RANK STABILITY" in report

    def test_contains_metrics(self, stable_scores: pd.DataFrame) -> None:
        results = evaluate_scores(stable_scores, top_ns=[10])
        report = format_report(results)
        assert "Cross-sectional std" in report
        assert "autocorr" in report
        assert "Overlap" in report
