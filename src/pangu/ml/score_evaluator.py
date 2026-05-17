"""Score matrix quality diagnostics.

Evaluates a score_matrix.parquet purely from its own data — no DB or
price data required.  Three diagnostic dimensions:

1. **Discrimination** — cross-sectional spread of scores
2. **Stability** — temporal autocorrelation of scores
3. **Rank stability** — Top-N overlap between rebalance periods
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Minimum trading days a stock must have to be included in stability analysis
MIN_OBSERVATIONS = 30
# Cap on stocks sampled for autocorrelation (avoids O(n·lags) cost on large universes)
MAX_SAMPLE_STOCKS = 200
# Standard autocorrelation lags: 1 day, 1 week, 1 month
AUTOCORR_LAGS = (1, 5, 21)
# Rebalance frequencies for rank-stability measurement
REBALANCE_FREQUENCIES = [("daily", 1), ("weekly", 5)]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_scores(
    scores: pd.DataFrame,
    top_ns: list[int] | None = None,
) -> dict:
    """Run full diagnostics on a score matrix.

    Parameters
    ----------
    scores : DataFrame (date × symbol), values are model predictions.
    top_ns : Top-N values to evaluate rank stability for.

    Returns
    -------
    dict with keys ``"discrimination"``, ``"stability"``, ``"rank_stability"``,
    each mapping to a sub-dict of metrics.
    """
    if top_ns is None:
        top_ns = [10, 30, 50]

    return {
        "discrimination": _compute_discrimination(scores, top_ns),
        "stability": _compute_stability(scores),
        "rank_stability": _compute_rank_stability(scores, top_ns),
    }


# ---------------------------------------------------------------------------
# 1. Discrimination — cross-sectional spread
# ---------------------------------------------------------------------------


def _compute_discrimination(scores: pd.DataFrame, top_ns: list[int]) -> dict:
    """Measure how well scores separate stocks on each day."""
    cs_std = scores.std(axis=1)
    p90 = scores.quantile(0.9, axis=1)
    p10 = scores.quantile(0.1, axis=1)

    result: dict = {
        "cross_sectional_std_mean": float(cs_std.mean()),
        "cross_sectional_std_median": float(cs_std.median()),
        "p90_p10_spread_mean": float((p90 - p10).mean()),
    }

    for n in top_ns:
        margins = []
        for d in scores.index:
            row = scores.loc[d].dropna().sort_values(ascending=False)
            if len(row) > n:
                margins.append(row.iloc[n - 1] - row.iloc[n])
        if margins:
            margin_arr = np.array(margins)
            result[f"top{n}_boundary_margin_mean"] = float(margin_arr.mean())
            result[f"top{n}_boundary_margin_median"] = float(np.median(margin_arr))
            # Ratio: daily score change vs boundary margin
            daily_change = scores.diff().abs().mean(axis=1).mean()
            result[f"top{n}_margin_to_daily_change"] = (
                float(margin_arr.mean() / daily_change) if daily_change > 0 else float("inf")
            )

    return result


# ---------------------------------------------------------------------------
# 2. Stability — temporal autocorrelation
# ---------------------------------------------------------------------------


def _compute_stability(scores: pd.DataFrame) -> dict:
    """Measure how stable individual stock scores are over time."""
    # Sample up to 200 stocks with enough data
    min_obs = min(MIN_OBSERVATIONS, len(scores) // 3)
    active = scores.columns[scores.notna().sum() > min_obs]
    sample = active[:MAX_SAMPLE_STOCKS] if len(active) > MAX_SAMPLE_STOCKS else active

    autocorrs: dict[int, list[float]] = {lag: [] for lag in AUTOCORR_LAGS}
    for s in sample:
        ts = scores[s].dropna()
        for lag in autocorrs:
            if len(ts) > lag + 10:
                ac = ts.autocorr(lag=lag)
                if not np.isnan(ac):
                    autocorrs[lag].append(ac)

    # Variance decomposition
    all_vals = scores.values[~np.isnan(scores.values)]
    overall_var = float(all_vals.var()) if len(all_vals) > 0 else 0.0
    cs_var = float(scores.var(axis=1).mean())  # cross-sectional (within-day)
    ts_var = float(scores.var(axis=0).mean())  # temporal (within-stock)

    result: dict = {}
    for lag in AUTOCORR_LAGS:
        vals = autocorrs[lag]
        result[f"score_autocorr_{lag}d"] = float(np.mean(vals)) if vals else float("nan")
    result.update(
        {
            "temporal_std_to_cs_std_ratio": (float(np.sqrt(ts_var) / np.sqrt(cs_var)) if cs_var > 0 else float("inf")),
            "overall_variance": overall_var,
            "cross_sectional_variance_pct": float(cs_var / overall_var * 100) if overall_var > 0 else 0.0,
            "temporal_variance_pct": float(ts_var / overall_var * 100) if overall_var > 0 else 0.0,
        }
    )
    return result


# ---------------------------------------------------------------------------
# 3. Rank stability — Top-N overlap across rebalance periods
# ---------------------------------------------------------------------------


def _compute_rank_stability(scores: pd.DataFrame, top_ns: list[int]) -> dict:
    """Measure how much the Top-N set changes between rebalance periods."""
    result: dict = {}

    for freq_name, freq_days in REBALANCE_FREQUENCIES:
        for n in top_ns:
            overlaps: list[float] = []
            jaccards: list[float] = []
            prev_top: set[str] | None = None

            for i in range(0, len(scores.index), freq_days):
                row = scores.iloc[i].dropna().sort_values(ascending=False)
                top_set = set(row.head(n).index)
                if prev_top is not None and len(top_set) == n:
                    overlap = len(top_set & prev_top)
                    union = len(top_set | prev_top)
                    overlaps.append(overlap / n)
                    jaccards.append(overlap / union if union > 0 else 0.0)
                if len(top_set) == n:
                    prev_top = top_set

            if overlaps:
                overlap_arr = np.array(overlaps)
                jaccard_arr = np.array(jaccards)
                key = f"{freq_name}_top{n}"
                result[f"{key}_overlap_mean"] = float(overlap_arr.mean())
                result[f"{key}_jaccard_mean"] = float(jaccard_arr.mean())
                result[f"{key}_low_overlap_pct"] = float((overlap_arr < 0.5).mean() * 100)

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(results: dict) -> str:
    """Format evaluation results as a terminal-friendly report."""
    lines: list[str] = []
    w = 72

    lines.append("=" * w)
    lines.append("Score Matrix Quality Report")
    lines.append("=" * w)

    # --- Discrimination ---
    d = results["discrimination"]
    lines.append("")
    lines.append("1. DISCRIMINATION (cross-sectional spread)")
    lines.append("-" * w)
    lines.append(f"  Cross-sectional std (mean):   {d['cross_sectional_std_mean']:.6f}")
    lines.append(f"  Cross-sectional std (median): {d['cross_sectional_std_median']:.6f}")
    lines.append(f"  P90-P10 spread (mean):        {d['p90_p10_spread_mean']:.6f}")

    top_ns = sorted({int(k.split("_")[0].replace("top", "")) for k in d if k.startswith("top")})
    if top_ns:
        lines.append("")
        lines.append(f"  {'Top-N':>8}  {'Boundary margin':>16}  {'Margin/DailyΔ':>14}")
        for n in top_ns:
            margin = d.get(f"top{n}_boundary_margin_mean", float("nan"))
            ratio = d.get(f"top{n}_margin_to_daily_change", float("nan"))
            flag = " ⚠" if ratio < 0.5 else ""
            lines.append(f"  {n:>8}  {margin:>16.6f}  {ratio:>13.3f}{flag}")
        lines.append("  (Margin/DailyΔ < 0.5 means boundary easily breached by noise)")

    # --- Stability ---
    s = results["stability"]
    lines.append("")
    lines.append("2. STABILITY (temporal autocorrelation)")
    lines.append("-" * w)
    for lag in AUTOCORR_LAGS:
        lines.append(f"  Score autocorr {lag}d:{s[f'score_autocorr_{lag}d']:>9.4f}")
    ratio = s["temporal_std_to_cs_std_ratio"]
    flag = " ⚠ (>1 means per-stock noise dominates cross-sectional signal)" if ratio > 1.0 else ""
    lines.append(f"  Temporal σ / CS σ:   {ratio:.4f}{flag}")
    lines.append(f"  Variance: CS={s['cross_sectional_variance_pct']:.1f}%, Temporal={s['temporal_variance_pct']:.1f}%")

    # --- Rank stability ---
    r = results["rank_stability"]
    lines.append("")
    lines.append("3. RANK STABILITY (Top-N overlap)")
    lines.append("-" * w)

    # Group by frequency
    for freq in ["daily", "weekly"]:
        freq_keys = sorted({int(k.split("_top")[1].split("_")[0]) for k in r if k.startswith(freq)})
        if not freq_keys:
            continue
        lines.append(f"  [{freq}]")
        lines.append(f"  {'Top-N':>8}  {'Overlap':>8}  {'Jaccard':>8}  {'<50% pct':>9}")
        for n in freq_keys:
            ov = r.get(f"{freq}_top{n}_overlap_mean", float("nan"))
            jc = r.get(f"{freq}_top{n}_jaccard_mean", float("nan"))
            lo = r.get(f"{freq}_top{n}_low_overlap_pct", float("nan"))
            flag = " ⚠" if ov < 0.5 else ""
            lines.append(f"  {n:>8}  {ov:>7.1%}  {jc:>8.3f}  {lo:>8.1f}%{flag}")

    lines.append("")
    lines.append("=" * w)

    # Interpretation guide
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Discrimination: Higher CS std / P90-P10 spread = better score")
    lines.append("    separation. Margin/DailyΔ < 0.5 means Top-N is unstable.")
    lines.append("  Stability: Autocorr > 0.9 is good; Temporal σ/CS σ > 1")
    lines.append("    means per-stock noise dominates the cross-sectional signal.")
    lines.append("  Rank: Weekly Top-N overlap < 50% signals excessive turnover.")

    return "\n".join(lines)
