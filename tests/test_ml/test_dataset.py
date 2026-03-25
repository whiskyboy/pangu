"""Tests for ml.dataset — Walk-Forward splitting and label computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.ml.dataset import (
    WindowSplit,
    build_window_datasets,
    compute_groups,
    compute_labels,
    discretize_labels,
    generate_walk_forward_windows,
    load_factor_panel,
)

# ---------------------------------------------------------------------------
# Walk-Forward windows
# ---------------------------------------------------------------------------

class TestWalkForwardWindows:
    def test_default_17_windows(self):
        windows = generate_walk_forward_windows()
        assert len(windows) == 17

    def test_first_window_dates(self):
        w = generate_walk_forward_windows()[0]
        assert w.window_id == 1
        assert w.train_start == "2020-01-01"
        assert w.train_end == "2021-06-30"
        assert w.val_start == "2021-07-01"
        assert w.val_end == "2021-09-30"
        assert w.test_start == "2021-10-01"
        assert w.test_end == "2021-12-31"

    def test_last_window_dates(self):
        w = generate_walk_forward_windows()[-1]
        assert w.window_id == 17
        assert w.train_start == "2024-01-01"
        assert w.train_end == "2025-06-30"
        assert w.test_start == "2025-10-01"
        assert w.test_end == "2025-12-31"

    def test_windows_non_overlapping_test(self):
        """Test periods should not overlap between consecutive windows."""
        windows = generate_walk_forward_windows()
        for i in range(len(windows) - 1):
            assert windows[i].test_end < windows[i + 1].test_start

    def test_no_train_test_overlap(self):
        """Train period must end before test period starts."""
        for w in generate_walk_forward_windows():
            assert w.train_end < w.val_start
            assert w.val_end < w.test_start

    def test_custom_params(self):
        windows = generate_walk_forward_windows(
            train_months=12, val_months=2, test_months=2,
            first_train_start="2021-01-01", last_test_end="2022-08-31",
        )
        assert len(windows) == 3
        assert windows[0].train_start == "2021-01-01"
        # 12 months training → end 2021-12-31
        assert windows[0].train_end == "2021-12-31"

    def test_3month_step(self):
        """Each window slides forward by 3 months (test_months)."""
        windows = generate_walk_forward_windows()
        # Window 2 train starts 3 months after window 1
        assert windows[1].train_start == "2020-04-01"
        assert windows[2].train_start == "2020-07-01"

    def test_frozen_dataclass(self):
        w = generate_walk_forward_windows()[0]
        with pytest.raises(AttributeError):
            w.window_id = 99

    # -- Expanding window tests --

    def test_expanding_same_window_count(self):
        """Expanding mode produces the same number of windows as fixed."""
        fixed = generate_walk_forward_windows()
        expanding = generate_walk_forward_windows(expanding=True)
        assert len(expanding) == len(fixed)

    def test_expanding_train_start_fixed(self):
        """All expanding windows have the same train_start."""
        windows = generate_walk_forward_windows(expanding=True)
        for w in windows:
            assert w.train_start == "2020-01-01"

    def test_expanding_train_grows(self):
        """Training period grows with each window in expanding mode."""
        windows = generate_walk_forward_windows(expanding=True)
        prev_end = None
        for w in windows:
            if prev_end is not None:
                assert w.train_end > prev_end
            prev_end = w.train_end

    def test_expanding_first_window_same_as_fixed(self):
        """First window is identical in both modes."""
        fixed = generate_walk_forward_windows()
        expanding = generate_walk_forward_windows(expanding=True)
        assert fixed[0] == expanding[0]

    def test_expanding_val_test_match_fixed(self):
        """Val and test periods should be the same in both modes."""
        fixed = generate_walk_forward_windows()
        expanding = generate_walk_forward_windows(expanding=True)
        for f, e in zip(fixed, expanding):
            assert f.val_start == e.val_start
            assert f.val_end == e.val_end
            assert f.test_start == e.test_start
            assert f.test_end == e.test_end

    def test_expanding_last_window_train_months(self):
        """Last expanding window should have much more training data."""
        windows = generate_walk_forward_windows(expanding=True)
        last = windows[-1]
        # Last window: train 2020-01-01 ~ 2025-06-30 = 66 months
        assert last.train_start == "2020-01-01"
        assert last.train_end == "2025-06-30"

    def test_expanding_no_train_val_overlap(self):
        """Train must end before val starts in expanding mode."""
        for w in generate_walk_forward_windows(expanding=True):
            assert w.train_end < w.val_start
            assert w.val_end < w.test_start


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

class TestComputeLabels:
    """Test label computation with a mock storage."""

    @pytest.fixture
    def mock_storage(self):
        """Minimal mock that returns deterministic bar data."""
        class _MockStorage:
            def load_daily_bars(self, symbol, start, end):
                dates = pd.bdate_range(start, end)
                rng = np.random.default_rng(hash(symbol) % 2**32)
                close = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
                return pd.DataFrame({
                    "date": dates.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "close": close,
                    "open": close * 1.01,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "volume": np.full(len(dates), 1_000_000),
                    "amount": close * 1_000_000,
                    "adj_factor": np.ones(len(dates)),
                })

            def load_fundamentals_filled(self, symbol, start, end):
                return pd.DataFrame()
        return _MockStorage()

    def test_label_shape(self, mock_storage):
        labels = compute_labels(mock_storage, ["A", "B"], "2024-01-01", "2024-03-31")
        assert isinstance(labels, pd.Series)
        assert labels.name == "label"
        assert labels.index.names == ["date", "symbol"]

    def test_label_last_horizon_nan(self, mock_storage):
        """Last 5 trading days should have NaN labels (no future data)."""
        labels = compute_labels(mock_storage, ["A"], "2024-01-01", "2024-03-31", horizon=5)
        dates = labels.index.get_level_values("date").unique().sort_values()
        last_5 = dates[-5:]
        for d in last_5:
            assert labels.loc[d].isna().all()

    def test_label_not_all_nan(self, mock_storage):
        labels = compute_labels(mock_storage, ["A", "B"], "2024-01-01", "2024-06-30")
        assert labels.notna().sum() > 0

    def test_normalize_mean_zero_std_one(self, mock_storage):
        """With normalize=True, per-day labels should have mean≈0, std≈1."""
        pool = [f"S{i}" for i in range(30)]
        labels = compute_labels(
            mock_storage, pool, "2024-01-01", "2024-06-30",
            normalize=True,
        )
        unstacked = labels.unstack("symbol")
        daily_mean = unstacked.mean(axis=1).dropna()
        daily_std = unstacked.std(axis=1).dropna()
        assert daily_mean.abs().max() < 1e-10
        assert daily_std.between(0.99, 1.01).all()

    def test_normalize_false_raw_labels(self, mock_storage):
        """With normalize=False, labels should be raw excess returns."""
        pool = ["A", "B"]
        raw = compute_labels(mock_storage, pool, "2024-01-01", "2024-06-30", normalize=False)
        normed = compute_labels(mock_storage, pool, "2024-01-01", "2024-06-30", normalize=True)
        # Raw labels should not have mean=0 per day (unless by coincidence)
        # But normalized and raw should have same ranking
        for date in raw.index.get_level_values("date").unique()[:5]:
            r = raw.loc[date].dropna()
            n = normed.loc[date].dropna()
            if len(r) >= 2 and len(n) >= 2:
                shared = r.index.intersection(n.index)
                if len(shared) >= 2:
                    assert r[shared].corr(n[shared]) == pytest.approx(1.0, abs=1e-10)

    def test_normalize_preserves_ranking(self, mock_storage):
        """Z-score is monotonic — ranking must be preserved."""
        pool = [f"S{i}" for i in range(20)]
        raw = compute_labels(mock_storage, pool, "2024-01-01", "2024-03-31", normalize=False)
        normed = compute_labels(mock_storage, pool, "2024-01-01", "2024-03-31", normalize=True)
        for date in raw.index.get_level_values("date").unique()[:10]:
            r = raw.loc[date].dropna().sort_values()
            n = normed.loc[date].dropna()
            if len(r) >= 2:
                n_aligned = n.reindex(r.index).dropna()
                if len(n_aligned) >= 2:
                    # Same ordering
                    assert list(r.index) == list(n_aligned.sort_values().index)


# ---------------------------------------------------------------------------
# load_factor_panel
# ---------------------------------------------------------------------------

class TestLoadFactorPanel:
    def test_parquet_path_priority(self, tmp_path):
        """When parquet exists, should load from it instead of DB."""
        # Create a small parquet
        idx = pd.MultiIndex.from_tuples(
            [("2024-01-01", "A"), ("2024-01-01", "B")],
            names=["date", "symbol"],
        )
        panel = pd.DataFrame(
            np.random.randn(2, 3).astype("float32"),
            index=idx,
            columns=["F1", "F2", "F3"],
        )
        pq_path = str(tmp_path / "test.parquet")
        panel.to_parquet(pq_path)

        # load_factor_panel should read from parquet (storage not used)
        result = load_factor_panel(None, ["A", "B"], "2024-01-01", "2024-12-31", pq_path)
        assert len(result) == 2
        assert list(result.columns) == ["F1", "F2", "F3"]


# ---------------------------------------------------------------------------
# build_window_datasets — data leakage prevention
# ---------------------------------------------------------------------------

class TestBuildWindowDatasets:
    def test_train_excludes_last_horizon_days(self):
        """Training set should not include last label_horizon days to prevent leakage."""
        dates = pd.bdate_range("2024-01-01", periods=100)
        symbols = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        panel = pd.DataFrame(
            np.random.randn(len(idx), 3).astype("float32"),
            index=idx, columns=["F1", "F2", "F3"],
        )
        labels = pd.Series(np.random.randn(len(idx)), index=idx, name="label")

        window = WindowSplit(
            window_id=1,
            train_start=str(dates[0].date()),
            train_end=str(dates[79].date()),  # 80 days train
            val_start=str(dates[80].date()),
            val_end=str(dates[89].date()),
            test_start=str(dates[90].date()),
            test_end=str(dates[99].date()),
        )

        class MockStorage:
            def load_constituents_union(self, s, e): return ["A", "B"]
            def load_constituents_for_date(self, d): return ["A", "B"]

        datasets = build_window_datasets(panel, labels, window, MockStorage(), label_horizon=5)
        X_train, _ = datasets["train"]
        train_dates = X_train.index.get_level_values("date").unique().sort_values()

        # Should exclude last 5 trading days of train period
        assert len(train_dates) == 75  # 80 - 5
        assert train_dates[-1] < pd.Timestamp(dates[75])


# ---------------------------------------------------------------------------
# LambdaRank helpers
# ---------------------------------------------------------------------------

class TestDiscretizeLabels:
    @pytest.fixture
    def sample_labels(self):
        """200 stocks × 10 days continuous labels."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        symbols = [f"S{i:03d}" for i in range(200)]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        rng = np.random.default_rng(42)
        return pd.Series(rng.normal(0, 0.03, len(idx)), index=idx, name="label")

    def test_output_is_integer(self, sample_labels):
        result = discretize_labels(sample_labels, n_bins=10)
        assert result.dtype == pd.Int32Dtype()

    def test_range_0_to_nbins_minus_1(self, sample_labels):
        for n_bins in [5, 10, 20]:
            result = discretize_labels(sample_labels, n_bins=n_bins)
            assert result.min() == 0
            assert result.max() == n_bins - 1

    def test_uniform_distribution_per_day(self, sample_labels):
        """Each bin should have approximately equal count per day."""
        n_bins = 10
        result = discretize_labels(sample_labels, n_bins=n_bins)
        per_day = result.groupby(level="date").value_counts()
        for date in result.index.get_level_values("date").unique():
            counts = per_day.loc[date]
            # 200 stocks / 10 bins = 20 per bin, allow ±1
            assert counts.min() >= 19
            assert counts.max() <= 21

    def test_preserves_ranking(self, sample_labels):
        """Higher return → higher (or equal) label."""
        result = discretize_labels(sample_labels, n_bins=10)
        for date in sample_labels.index.get_level_values("date").unique()[:3]:
            raw = sample_labels.loc[date]
            binned = result.loc[date]
            # Sort by raw return, binned labels should be non-decreasing
            sorted_idx = raw.sort_values().index
            binned_sorted = binned.reindex(sorted_idx).values
            assert np.all(np.diff(binned_sorted) >= 0)

    def test_nan_preserved(self):
        """NaN labels should stay NaN after discretization."""
        idx = pd.MultiIndex.from_tuples(
            [("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C")],
            names=["date", "symbol"],
        )
        labels = pd.Series([0.05, np.nan, -0.02], index=idx, name="label")
        result = discretize_labels(labels, n_bins=5)
        assert pd.isna(result.loc[("2024-01-01", "B")])
        assert pd.notna(result.loc[("2024-01-01", "A")])

    def test_index_names_preserved(self, sample_labels):
        result = discretize_labels(sample_labels)
        assert result.index.names == ["date", "symbol"]
        assert result.name == "label"


class TestComputeGroups:
    def test_basic(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        symbols = ["A", "B", "C"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        groups = compute_groups(idx)
        assert groups == [3, 3, 3]
        assert sum(groups) == len(idx)

    def test_unequal_groups(self):
        """Days may have different stock counts (missing data)."""
        idx = pd.MultiIndex.from_tuples([
            ("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C"),
            ("2024-01-02", "A"), ("2024-01-02", "B"),
        ], names=["date", "symbol"])
        groups = compute_groups(idx)
        assert sum(groups) == 5
        assert len(groups) == 2

    def test_sorted_chronologically(self):
        """Groups must be in date order; unsorted input raises."""
        idx = pd.MultiIndex.from_tuples([
            ("2024-01-03", "A"),
            ("2024-01-01", "B"), ("2024-01-01", "C"),
            ("2024-01-02", "A"),
        ], names=["date", "symbol"])
        with pytest.raises(ValueError, match="sorted by date"):
            compute_groups(idx)

    def test_sorted_input_works(self):
        """Sorted input returns correct groups."""
        idx = pd.MultiIndex.from_tuples([
            ("2024-01-01", "B"), ("2024-01-01", "C"),
            ("2024-01-02", "A"),
            ("2024-01-03", "A"),
        ], names=["date", "symbol"])
        groups = compute_groups(idx)
        assert groups == [2, 1, 1]
