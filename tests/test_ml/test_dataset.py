"""Tests for ml.dataset — Walk-Forward splitting and label computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pangu.ml.dataset import (
    WindowSplit,
    build_window_datasets,
    compute_labels,
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
