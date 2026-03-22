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
    step_months: int | None = None,
    first_train_start: str = "2020-01-01",
    last_test_end: str = "2025-12-31",
) -> list[WindowSplit]:
    """Generate Walk-Forward window date ranges.

    Each window slides forward by ``step_months`` months (default: same as
    ``test_months``).  When ``step_months < test_months``, test periods
    overlap and the same date may be scored by multiple windows — the caller
    is responsible for averaging the overlapping predictions.
    """
    from dateutil.relativedelta import relativedelta

    if step_months is None:
        step_months = test_months
    elif step_months > test_months:
        raise ValueError(
            f"step_months ({step_months}) cannot exceed test_months ({test_months}). "
            f"This would create gaps in score coverage."
        )

    start = datetime.strptime(first_train_start, "%Y-%m-%d")
    cutoff = datetime.strptime(last_test_end, "%Y-%m-%d")
    windows: list[WindowSplit] = []

    i = 0
    while True:
        offset = relativedelta(months=step_months * i)
        t_start = start + offset
        t_end = t_start + relativedelta(months=train_months) - relativedelta(days=1)
        v_start = t_end + relativedelta(days=1)
        v_end = v_start + relativedelta(months=val_months) - relativedelta(days=1)
        te_start = v_end + relativedelta(days=1)
        te_end = te_start + relativedelta(months=test_months) - relativedelta(days=1)

        if te_end > cutoff:
            break

        windows.append(WindowSplit(
            window_id=i + 1,
            train_start=t_start.strftime("%Y-%m-%d"),
            train_end=t_end.strftime("%Y-%m-%d"),
            val_start=v_start.strftime("%Y-%m-%d"),
            val_end=v_end.strftime("%Y-%m-%d"),
            test_start=te_start.strftime("%Y-%m-%d"),
            test_end=te_end.strftime("%Y-%m-%d"),
        ))
        i += 1

    if not windows:
        raise ValueError(
            f"No windows fit: first_train_start={first_train_start}, "
            f"last_test_end={last_test_end}, "
            f"window size={train_months}+{val_months}+{test_months} months"
        )

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
    DataFrame with MultiIndex(date, symbol) × 191 factor columns, float32.
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
    horizon: int | list[int] = 5,
    horizon_weights: list[float] | None = None,
    winsorize: float | None = 0.2,
    normalize: bool = False,
) -> pd.Series:
    """Compute forward N-day excess return labels.

    label = stock_return_Nd - benchmark_return_Nd
    Uses forward-adjusted close prices (close × adj_factor) so that
    ex-dividend/split price gaps do not bias the return calculation.

    When ``horizon`` is a list with multiple values (e.g. [5, 10, 20]),
    multi-horizon label fusion is applied: weighted sum of raw excess
    returns across horizons. Since horizon returns overlap (5d ⊂ 10d ⊂ 20d),
    the raw sum naturally creates time-decay weighting:
    days 1-5 counted 3x, days 6-10 counted 2x, days 11-20 counted 1x.

    No z-score normalization is applied — preserving absolute return
    magnitude is critical for Top-N selection strategies.

    Parameters
    ----------
    storage : DataStorage
    pool : list[str]
        Symbols to compute labels for.
    start, end : str
        Date range. Note: last ``max(horizon)`` trading days will have
        NaN labels.
    horizon : int or list[int]
        Forward return horizon in trading days (default 5).
        Pass a list (e.g. [5, 10, 20]) for multi-horizon label fusion.
    horizon_weights : list[float] or None
        Weights for each horizon in multi-horizon fusion.
        Must match length of ``horizon`` list. If None, equal weights.
    winsorize : float or None
        If set, clip labels to [-winsorize, +winsorize] to reduce
        the impact of extreme returns on MSE/MAE loss (default 0.2).
    normalize : bool
        If True, apply Qlib-style cross-sectional z-score to labels.
        Each day's labels are standardized to mean=0, std=1 across all stocks,
        so the model learns to predict relative outperformance rather than
        absolute excess returns. Disabled by default — raw excess returns
        perform better for Top-N selection strategies.

    Returns
    -------
    Series with MultiIndex(date, symbol), name="label".
    """
    # Resolve horizon to a list
    horizon_list: list[int] = horizon if isinstance(horizon, list) else [horizon]
    if not horizon_list:
        raise ValueError("horizon cannot be empty")
    if any(h <= 0 for h in horizon_list):
        raise ValueError(f"All horizons must be positive integers, got {horizon_list}")
    if len(horizon_list) > 1:
        if horizon_weights is not None:
            if len(horizon_weights) != len(horizon_list):
                raise ValueError(
                    f"horizon_weights length {len(horizon_weights)} != "
                    f"horizons length {len(horizon_list)}"
                )
            paired = sorted(zip(horizon_list, horizon_weights))
            horizon_list = [h for h, _ in paired]
            horizon_weights = [w for _, w in paired]
        else:
            horizon_list = sorted(horizon_list)

    # Load stock close prices and adj_factor for forward-adjusted returns
    bar_frames = []
    for sym in pool:
        bars = storage.load_daily_bars(sym, start, end)
        if bars is not None and not bars.empty:
            bar_frames.append(bars[["date", "symbol", "close", "adj_factor"]])

    if not bar_frames:
        return pd.Series(dtype="float64", name="label")

    all_bars = pd.concat(bar_frames, ignore_index=True)
    all_bars["date"] = pd.to_datetime(all_bars["date"])
    close_wide = all_bars.pivot(index="date", columns="symbol", values="close")
    adj_wide = all_bars.pivot(index="date", columns="symbol", values="adj_factor")

    # Forward-adjusted close: eliminates ex-dividend/split price gaps
    fwd_adj_close = close_wide * adj_wide.ffill()

    # Benchmark (CSI300) forward return
    bench_df = storage.load_daily_bars("000300", start, end)
    if bench_df is None or bench_df.empty:
        raise ValueError("No CSI300 benchmark data. Run 'pangu backfill index' first.")
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_close = bench_df.set_index("date")["close"]

    # Compute weighted sum of per-horizon excess returns.
    # For single horizon this is just 1.0 × excess. For multi-horizon,
    # overlapping horizons naturally create time-decay weighting
    # (e.g. 5d+10d+20d: days 1-5 counted 3×, 6-10 counted 2×, 11-20 counted 1×).
    weights = horizon_weights if horizon_weights is not None else [1.0] * len(horizon_list)

    excess_parts = []
    for h in horizon_list:
        stock_ret = fwd_adj_close.shift(-h) / fwd_adj_close - 1
        stock_ret = stock_ret.replace([np.inf, -np.inf], np.nan)
        bench_ret = bench_close.shift(-h) / bench_close - 1
        bench_ret = bench_ret.reindex(close_wide.index)
        excess = stock_ret.sub(bench_ret, axis=0)

        if winsorize is not None:
            excess = excess.clip(-winsorize, winsorize)

        excess_parts.append(excess)

    fused = sum(w * e for w, e in zip(weights, excess_parts))

    label = fused.stack(future_stack=True)
    label.index.names = ["date", "symbol"]
    label.name = "label"

    # Cross-sectional z-score (Qlib CSZScoreNorm on labels)
    if normalize:
        unstacked = label.unstack("symbol")
        cs_mean = unstacked.mean(axis=1)
        cs_std = unstacked.std(axis=1).replace(0.0, np.nan)
        unstacked = unstacked.sub(cs_mean, axis=0).div(cs_std, axis=0)
        label = unstacked.stack(future_stack=True)
        label.index.names = ["date", "symbol"]
        label.name = "label"

    return label


# ---------------------------------------------------------------------------
# LambdaRank helpers
# ---------------------------------------------------------------------------

def discretize_labels(labels: pd.Series, n_bins: int = 10) -> pd.Series:
    """Per-day percentile discretization for LambdaRank.

    Converts continuous excess returns into integer relevance grades 0..n_bins-1.
    Each day independently: rank stocks → divide into n_bins equal groups.
    Distribution is uniform by construction (each bin ≈ 1/n_bins of stocks).

    LGBMRanker requires non-negative integer labels; continuous floats cause
    a Fatal error in the C++ core.

    Parameters
    ----------
    labels : Series with MultiIndex(date, symbol)
        Continuous labels (e.g. excess returns).
    n_bins : int
        Number of relevance grades (default 10 = decile).

    Returns
    -------
    Series with same index, integer values 0..n_bins-1, name="label".
    NaN labels remain NaN.
    """
    import numpy as np

    unstacked = labels.unstack("symbol")
    ranked = unstacked.rank(axis=1, pct=True, na_option="keep")
    # pct=True gives values in (0, 1]; multiply by n_bins and floor
    # clip to n_bins-1 to handle the exact 1.0 case
    binned = (ranked * n_bins).apply(np.floor).clip(upper=n_bins - 1)
    # Preserve NaN where original labels were NaN
    binned = binned.where(unstacked.notna())
    stacked = binned.stack(future_stack=True).astype("Int32")
    stacked.index.names = ["date", "symbol"]
    stacked.name = "label"
    # unstack/stack can introduce new (date, symbol) combinations for sparse panels;
    # restrict to the original index to keep alignment with X
    return stacked.reindex(labels.index)


def compute_groups(index: pd.MultiIndex) -> list[int]:
    """Return per-day sample counts for LambdaRank group parameter.

    LGBMRanker requires a ``group`` list where each element is the number
    of samples in one query (= one trading day). The list must be ordered
    chronologically and its sum must equal the total number of samples.

    The input index MUST be sorted by date (level 0). Raises ValueError
    if dates are not in non-decreasing order.
    """
    dates = index.get_level_values("date")
    # Verify data is sorted by date — LGBMRanker requires contiguous groups
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Index must be sorted by date for LambdaRank groups. "
            "Call .sort_index(level='date') first."
        )
    return dates.value_counts().sort_index().tolist()


# ---------------------------------------------------------------------------
# Sample weighting
# ---------------------------------------------------------------------------

def compute_time_decay_weights(
    index: pd.MultiIndex,
    halflife_days: int,
) -> pd.Series:
    """Compute exponential time-decay sample weights.

    Recent samples get higher weight; older samples decay exponentially.
    Weight formula: w(t) = 2^(-(t_max - t) / halflife_days)

    At t_max (newest sample): weight = 1.0
    At t_max - halflife: weight = 0.5
    At t_max - 2*halflife: weight = 0.25

    Parameters
    ----------
    index : MultiIndex(date, symbol)
        Sample index from training data.
    halflife_days : int
        Decay half-life in **trading days**. Larger = slower decay.
        Typical values: 40 (~2 months), 80 (~4 months), 120 (~6 months).

    Returns
    -------
    Series with same index, float64 weights in (0, 1].
    """
    dates = index.get_level_values("date")
    if halflife_days <= 0:
        raise ValueError(f"halflife_days must be positive, got {halflife_days}")
    unique_dates = np.sort(dates.unique())
    date_to_ord = {d: i for i, d in enumerate(unique_dates)}
    max_ord = len(unique_dates) - 1
    ordinals = np.array([date_to_ord[d] for d in dates])
    days_ago = (max_ord - ordinals).astype(np.float64)
    weights = np.power(2.0, -days_ago / halflife_days)
    return pd.Series(weights, index=index, name="weight")


# ---------------------------------------------------------------------------
# Window dataset builder
# ---------------------------------------------------------------------------

def build_window_datasets(
    panel: pd.DataFrame,
    labels: pd.Series,
    window: WindowSplit,
    storage: "DataStorage",
    label_horizon: int = 5,
    train_subsample_stride: int | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Build train/val/test datasets for one Walk-Forward window.

    Stock pool filtering:
    - Train: constituents_union(train_start, train_end) — maximize samples
    - Val: constituents_for_date(val_start) — simulate real trading pool
    - Test: constituents_for_date(test_start) — simulate real trading pool

    For training data, the last ``label_horizon`` trading days are excluded
    to prevent data leakage (their labels use prices from the validation period).

    Parameters
    ----------
    train_subsample_stride : int or None
        If set, subsample training dates using random block sampling to reduce
        label overlap redundancy. Dates are divided into non-overlapping blocks
        of ``train_subsample_stride`` trading days, and one date is randomly
        selected from each block. Typically set equal to ``label_horizon``
        (e.g. 5 for 5-day labels) so consecutive selected dates have no label
        overlap. Val and test splits are not affected.

    Returns
    -------
    {"train": (X, y), "val": (X, y), "test": (X, y)}
    Each X is a DataFrame (rows=samples, cols=191 factors).
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

        # Random block subsampling for training set to reduce label overlap
        if split_name == "train" and train_subsample_stride and not subset.empty:
            unique_dates = subset.index.get_level_values("date").unique().sort_values()
            rng = np.random.RandomState(42)
            sampled_dates = []
            for block_start in range(0, len(unique_dates), train_subsample_stride):
                block = unique_dates[block_start:block_start + train_subsample_stride]
                sampled_dates.append(rng.choice(block))
            subset = subset.loc[
                subset.index.get_level_values("date").isin(sampled_dates)
            ]

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
