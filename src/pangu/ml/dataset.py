"""Walk-Forward dataset construction for LightGBM training.

Handles:
- Factor panel loading (parquet-first, DB fallback)
- Label computation (5-day excess return vs CSI300)
- Walk-Forward window generation (17 windows)
- Train/val/test split with constituent-aware filtering
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pangu.data.storage import DataStorage


# ---------------------------------------------------------------------------
# Walk-Forward window definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowSplit:
    """Date ranges for one Walk-Forward window."""

    window_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def generate_walk_forward_windows(
    train_months: int = 18,
    val_months: int = 3,
    test_months: int = 3,
    first_train_start: str = "2020-01-01",
    n_windows: int = 17,
) -> list[WindowSplit]:
    """Generate Walk-Forward window date ranges.

    Each window slides forward by ``test_months`` months.
    Returns list of ``n_windows`` WindowSplit objects.
    """
    from dateutil.relativedelta import relativedelta

    start = datetime.strptime(first_train_start, "%Y-%m-%d")
    step = relativedelta(months=test_months)
    windows: list[WindowSplit] = []

    for i in range(n_windows):
        offset = relativedelta(months=test_months * i)
        t_start = start + offset
        t_end = t_start + relativedelta(months=train_months) - relativedelta(days=1)
        v_start = t_end + relativedelta(days=1)
        v_end = v_start + relativedelta(months=val_months) - relativedelta(days=1)
        te_start = v_end + relativedelta(days=1)
        te_end = te_start + relativedelta(months=test_months) - relativedelta(days=1)

        windows.append(WindowSplit(
            window_id=i + 1,
            train_start=t_start.strftime("%Y-%m-%d"),
            train_end=t_end.strftime("%Y-%m-%d"),
            val_start=v_start.strftime("%Y-%m-%d"),
            val_end=v_end.strftime("%Y-%m-%d"),
            test_start=te_start.strftime("%Y-%m-%d"),
            test_end=te_end.strftime("%Y-%m-%d"),
        ))

    return windows


# ---------------------------------------------------------------------------
# Factor panel loading
# ---------------------------------------------------------------------------

def load_factor_panel(
    storage: "DataStorage",
    pool: list[str],
    start: str,
    end: str,
    factors_path: str | None = None,
) -> pd.DataFrame:
    """Load factor panel: parquet-first, DB fallback.

    Parameters
    ----------
    storage : DataStorage
        Database handle for bar and fundamental data.
    pool : list[str]
        Stock symbols to include.
    start, end : str
        Date range (inclusive).
    factors_path : str or None
        Path to pre-computed factors.parquet. If provided and exists,
        loads and slices from it (fast). Otherwise computes from DB.

    Returns
    -------
    DataFrame with MultiIndex(date, symbol) × 166 factor columns, float32.
    """
    if factors_path and Path(factors_path).exists():
        panel = pd.read_parquet(factors_path)
        # Slice to requested date range and pool
        dates = panel.index.get_level_values("date")
        symbols = panel.index.get_level_values("symbol")
        mask = (dates >= start) & (dates <= end) & (symbols.isin(pool))
        return panel.loc[mask]

    # Fallback: compute from DB
    from pangu.factor.alpha158 import Alpha158Engine

    bar_frames = []
    fund_frames = []
    for sym in pool:
        bars = storage.load_daily_bars(sym, start, end)
        if bars is not None and not bars.empty:
            bar_frames.append(bars)
        fund = storage.load_fundamentals_filled(sym, start, end)
        if fund is not None and not fund.empty:
            fund_frames.append(fund)

    if not bar_frames:
        return pd.DataFrame()

    all_bars = pd.concat(bar_frames, ignore_index=True)
    fundamentals = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()

    engine = Alpha158Engine()
    return engine.compute(all_bars, fundamentals)


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def compute_labels(
    storage: "DataStorage",
    pool: list[str],
    start: str,
    end: str,
    horizon: int = 5,
) -> pd.Series:
    """Compute forward N-day excess return labels.

    label = stock_return_Nd - benchmark_return_Nd
    Uses unadjusted close prices (consistent with backtest engine).

    Parameters
    ----------
    storage : DataStorage
    pool : list[str]
        Symbols to compute labels for.
    start, end : str
        Date range. Note: last ``horizon`` trading days will have NaN labels.
    horizon : int
        Forward return horizon in trading days (default 5).

    Returns
    -------
    Series with MultiIndex(date, symbol), name="label".
    """
    # Load stock close prices (unadjusted)
    bar_frames = []
    for sym in pool:
        bars = storage.load_daily_bars(sym, start, end)
        if bars is not None and not bars.empty:
            bar_frames.append(bars[["date", "symbol", "close"]])

    if not bar_frames:
        return pd.Series(dtype="float64", name="label")

    all_bars = pd.concat(bar_frames, ignore_index=True)
    all_bars["date"] = pd.to_datetime(all_bars["date"])
    close_wide = all_bars.pivot(index="date", columns="symbol", values="close")

    # Stock forward return
    stock_ret = close_wide.shift(-horizon) / close_wide - 1

    # Benchmark (CSI300) forward return
    bench_df = storage.load_daily_bars("000300", start, end)
    if bench_df is None or bench_df.empty:
        raise ValueError("No CSI300 benchmark data. Run 'pangu backfill index' first.")
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_close = bench_df.set_index("date")["close"]
    bench_ret = bench_close.shift(-horizon) / bench_close - 1
    # Align to stock dates
    bench_ret = bench_ret.reindex(close_wide.index)

    # Excess return: stock - benchmark (broadcast across columns)
    excess = stock_ret.sub(bench_ret, axis=0)

    # Stack to MultiIndex Series
    label = excess.stack(future_stack=True)
    label.index.names = ["date", "symbol"]
    label.name = "label"
    return label


# ---------------------------------------------------------------------------
# Window dataset builder
# ---------------------------------------------------------------------------

def build_window_datasets(
    panel: pd.DataFrame,
    labels: pd.Series,
    window: WindowSplit,
    storage: "DataStorage",
    label_horizon: int = 5,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Build train/val/test datasets for one Walk-Forward window.

    Stock pool filtering:
    - Train: constituents_union(train_start, train_end) — maximize samples
    - Val: constituents_for_date(val_start) — simulate real trading pool
    - Test: constituents_for_date(test_start) — simulate real trading pool

    For training data, the last ``label_horizon`` trading days are excluded
    to prevent data leakage (their labels use prices from the validation period).

    Returns
    -------
    {"train": (X, y), "val": (X, y), "test": (X, y)}
    Each X is a DataFrame (rows=samples, cols=166 factors).
    Each y is a Series of labels.
    NaN labels are dropped.
    """
    dates = panel.index.get_level_values("date")

    result = {}
    for split_name, d_start, d_end, pool_fn in [
        ("train", window.train_start, window.train_end, "union"),
        ("val", window.val_start, window.val_end, "snapshot"),
        ("test", window.test_start, window.test_end, "snapshot"),
    ]:
        # Date filter
        mask = (dates >= d_start) & (dates <= d_end)
        subset = panel.loc[mask]

        # Exclude last label_horizon trading days from training set
        # to prevent data leakage (labels peek into validation period)
        if split_name == "train" and not subset.empty:
            unique_dates = subset.index.get_level_values("date").unique().sort_values()
            if len(unique_dates) > label_horizon:
                cutoff = unique_dates[-label_horizon]
                subset = subset.loc[subset.index.get_level_values("date") < cutoff]

        if subset.empty:
            result[split_name] = (pd.DataFrame(), pd.Series(dtype="float64"))
            continue

        # Stock pool filter
        if pool_fn == "union":
            pool = set(storage.load_constituents_union(d_start, d_end))
        else:
            pool = set(storage.load_constituents_for_date(d_start))

        symbols = subset.index.get_level_values("symbol")
        subset = subset.loc[symbols.isin(pool)]

        if subset.empty:
            result[split_name] = (pd.DataFrame(), pd.Series(dtype="float64"))
            continue

        # Align labels
        common_idx = subset.index.intersection(labels.index)
        X = subset.loc[common_idx]
        y = labels.loc[common_idx]

        # Drop NaN labels (last horizon days + missing data)
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        result[split_name] = (X, y)

    return result
