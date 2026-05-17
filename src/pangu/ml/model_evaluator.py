"""Walk-forward model quality diagnostics.

Loads all ``wf_window_*.txt`` LightGBM model files and produces:

1. **Global feature importance** — cross-window normalised gain ranking
2. **Per-window summary** — tree counts, top features, underfitting flags
3. **Feature drift** — adjacent-window Top-K Jaccard similarity
4. **Zero-importance features** — always-zero vs frequently-zero
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)

# Threshold: windows with fewer trees are flagged as likely underfitting
MIN_TREES_THRESHOLD = 50
# Number of top features used for drift Jaccard calculation
DRIFT_TOP_K = 10
# Fraction of windows a feature must be zero in to count as "frequently zero"
FREQ_ZERO_THRESHOLD = 0.8

_WINDOW_FILE_RE = re.compile(r"wf_window_(\d+)(?:_seed(\d+))?\.txt$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_models(
    model_dir: str,
    top_n: int = 20,
) -> dict:
    """Run full diagnostics on walk-forward window models.

    Parameters
    ----------
    model_dir : Directory containing ``wf_window_*.txt`` files.
    top_n : Number of top features to include in the global ranking.

    Returns
    -------
    dict with keys ``"global_importance"``, ``"per_window"``,
    ``"feature_drift"``, ``"zero_importance"``.
    """
    windows = _load_window_boosters(model_dir)
    if not windows:
        return {
            "global_importance": {},
            "per_window": [],
            "feature_drift": {},
            "zero_importance": {},
        }

    return {
        "global_importance": _compute_global_importance(windows, top_n),
        "per_window": _compute_per_window_summary(windows),
        "feature_drift": _compute_feature_drift(windows),
        "zero_importance": _compute_zero_importance(windows),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_window_boosters(model_dir: str) -> list[tuple[int, list[lgb.Booster]]]:
    """Scan *model_dir* for ``wf_window_*.txt`` files grouped by window.

    Returns ``(window_id, boosters)`` sorted by window_id.  When multi-seed
    models exist (``wf_window_01_seed0.txt``, etc.), all seeds for a window
    are loaded so downstream functions can average their importance.
    """
    grouped: dict[int, list[lgb.Booster]] = {}
    model_path = Path(model_dir)
    if not model_path.is_dir():
        logger.warning("Model directory not found: %s", model_dir)
        return []

    for f in sorted(model_path.iterdir()):
        m = _WINDOW_FILE_RE.search(f.name)
        if m:
            window_id = int(m.group(1))
            try:
                booster = lgb.Booster(model_file=str(f))
            except Exception:
                logger.warning("Failed to load model file %s, skipping", f)
                continue
            grouped.setdefault(window_id, []).append(booster)

    results = [(wid, boosters) for wid, boosters in sorted(grouped.items())]
    n_models = sum(len(bs) for _, bs in results)
    if n_models > len(results):
        logger.info("Loaded %d windows (%d models, multi-seed) from %s", len(results), n_models, model_dir)
    else:
        logger.info("Loaded %d window models from %s", len(results), model_dir)
    return results


def _normalised_importance(booster: lgb.Booster) -> dict[str, float]:
    """Return feature importance as percentages (sum=100) for one booster."""
    raw = booster.feature_importance(importance_type="gain")
    total = raw.sum()
    if total == 0:
        return {}
    names = booster.feature_name()
    return {name: float(val / total * 100) for name, val in zip(names, raw)}


def _averaged_importance(boosters: list[lgb.Booster]) -> dict[str, float]:
    """Average normalised importance across multiple boosters (e.g. seeds).

    Returns a single importance dict whose values sum to ~100.
    """
    all_pcts = [_normalised_importance(b) for b in boosters]
    all_feats: set[str] = set()
    for p in all_pcts:
        all_feats.update(p.keys())
    if not all_feats:
        return {}
    return {f: float(np.mean([p.get(f, 0.0) for p in all_pcts])) for f in all_feats}


# ---------------------------------------------------------------------------
# 1. Global feature importance
# ---------------------------------------------------------------------------


def _compute_global_importance(
    windows: list[tuple[int, list[lgb.Booster]]],
    top_n: int,
) -> dict:
    """Aggregate normalised importance across all windows.

    When windows contain multiple seeds, importance is averaged across seeds
    within each window before aggregating across windows.
    """
    all_features: set[str] = set()
    per_window_pcts: list[dict[str, float]] = []

    for _wid, boosters in windows:
        pcts = _averaged_importance(boosters)
        per_window_pcts.append(pcts)
        all_features.update(pcts.keys())

    n_windows = len(windows)
    feature_stats: list[dict] = []
    for feat in all_features:
        vals = [pw.get(feat, 0.0) for pw in per_window_pcts]
        non_zero_count = sum(1 for v in vals if v > 0)
        feature_stats.append(
            {
                "feature": feat,
                "mean_pct": float(np.mean(vals)),
                "std_pct": float(np.std(vals)),
                "windows_used": non_zero_count,
                "windows_total": n_windows,
            }
        )

    feature_stats.sort(key=lambda x: x["mean_pct"], reverse=True)
    return {"top_features": feature_stats[:top_n], "total_features": len(all_features)}


# ---------------------------------------------------------------------------
# 2. Per-window summary
# ---------------------------------------------------------------------------


def _compute_per_window_summary(windows: list[tuple[int, list[lgb.Booster]]]) -> list[dict]:
    """Tree count + top-5 features per window.

    Multi-seed windows report tree count range (min–max) and seed-averaged
    feature importance for the top-5 ranking.
    """
    summaries: list[dict] = []
    for wid, boosters in windows:
        tree_counts = [b.num_trees() for b in boosters]
        pcts = _averaged_importance(boosters)
        top5 = sorted(pcts, key=pcts.get, reverse=True)[:5] if pcts else []  # type: ignore[arg-type]
        summaries.append(
            {
                "window_id": wid,
                "num_trees": min(tree_counts),
                "num_trees_max": max(tree_counts),
                "n_seeds": len(boosters),
                "top5_features": top5,
                "underfitting": min(tree_counts) < MIN_TREES_THRESHOLD,
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# 3. Feature drift
# ---------------------------------------------------------------------------


def _compute_feature_drift(windows: list[tuple[int, list[lgb.Booster]]]) -> dict:
    """Jaccard similarity of top-K features between adjacent windows.

    For multi-seed windows, importance is averaged across seeds before
    extracting the top-K features.
    """
    if len(windows) < 2:
        return {"pairs": [], "mean_jaccard": float("nan")}

    window_topk: list[tuple[int, set[str]]] = []
    for wid, boosters in windows:
        pcts = _averaged_importance(boosters)
        topk = set(sorted(pcts, key=pcts.get, reverse=True)[:DRIFT_TOP_K]) if pcts else set()  # type: ignore[arg-type]
        window_topk.append((wid, topk))

    pairs: list[dict] = []
    for i in range(len(window_topk) - 1):
        wid_a, top_a = window_topk[i]
        wid_b, top_b = window_topk[i + 1]
        union = len(top_a | top_b)
        jaccard = len(top_a & top_b) / union if union > 0 else 0.0
        pairs.append(
            {
                "from_window": wid_a,
                "to_window": wid_b,
                "jaccard": float(jaccard),
            }
        )

    jaccards = [p["jaccard"] for p in pairs]
    return {"pairs": pairs, "mean_jaccard": float(np.mean(jaccards))}


# ---------------------------------------------------------------------------
# 4. Zero-importance features
# ---------------------------------------------------------------------------


def _compute_zero_importance(windows: list[tuple[int, list[lgb.Booster]]]) -> dict:
    """Find features with zero importance across windows.

    For multi-seed windows, a feature is counted as zero for a window only
    if it has zero importance in ALL seeds of that window.
    """
    all_features: set[str] = set()
    zero_counts: dict[str, int] = {}
    n_windows = len(windows)

    for _wid, boosters in windows:
        per_seed_zeros: list[set[str]] = []
        for b in boosters:
            raw = b.feature_importance(importance_type="gain")
            names = b.feature_name()
            all_features.update(names)
            per_seed_zeros.append({name for name, val in zip(names, raw) if val == 0})
        window_zeros = set.intersection(*per_seed_zeros) if per_seed_zeros else set()
        for name in window_zeros:
            zero_counts[name] = zero_counts.get(name, 0) + 1

    always_zero = sorted(f for f, c in zero_counts.items() if c == n_windows)
    frequently_zero = sorted(
        f for f, c in zero_counts.items() if c >= n_windows * FREQ_ZERO_THRESHOLD and c < n_windows
    )

    return {
        "always_zero": always_zero,
        "frequently_zero": frequently_zero,
        "n_windows": n_windows,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_model_report(results: dict) -> str:
    """Format model evaluation results as a terminal-friendly report."""
    lines: list[str] = []
    w = 72

    per_window = results["per_window"]
    n_windows = len(per_window)
    lines.append("=" * w)
    lines.append(f"Model Evaluation Report ({n_windows} windows)")
    lines.append("=" * w)

    # --- 1. Global importance ---
    gi = results["global_importance"]
    top_features = gi.get("top_features", [])
    lines.append("")
    lines.append(f"1. GLOBAL FEATURE IMPORTANCE (Top-{len(top_features)})")
    lines.append("-" * w)
    if top_features:
        lines.append(f"  {'Rank':>4}  {'Feature':<20} {'Avg%':>6} {'Std%':>6} {'Windows':>8}")
        for i, f in enumerate(top_features, 1):
            lines.append(
                f"  {i:>4}  {f['feature']:<20} {f['mean_pct']:>5.1f}% {f['std_pct']:>5.1f}%"
                f" {f['windows_used']:>3}/{f['windows_total']}"
            )
    else:
        lines.append("  No features found.")

    # --- 2. Per-window summary ---
    lines.append("")
    lines.append("2. PER-WINDOW SUMMARY")
    lines.append("-" * w)
    has_multiseed = any(pw.get("n_seeds", 1) > 1 for pw in per_window)
    if has_multiseed:
        lines.append(f"  {'Window':>6}  {'Trees':>10}  {'Seeds':>5}  Top-5 Features (seed-averaged)")
    else:
        lines.append(f"  {'Window':>6}  {'Trees':>5}  Top-5 Features")
    for pw in per_window:
        flag = " ⚠" if pw["underfitting"] else ""
        top5 = ", ".join(pw["top5_features"])
        if has_multiseed:
            trees_min = pw["num_trees"]
            trees_max = pw.get("num_trees_max", trees_min)
            trees_str = f"{trees_min}~{trees_max}" if trees_min != trees_max else str(trees_min)
            lines.append(f"  {pw['window_id']:>6}  {trees_str:>10}{flag}  {pw.get('n_seeds', 1):>5}  {top5}")
        else:
            lines.append(f"  {pw['window_id']:>6}  {pw['num_trees']:>5}{flag}  {top5}")
    underfitting = [pw for pw in per_window if pw["underfitting"]]
    if underfitting:
        ids = ", ".join(str(pw["window_id"]) for pw in underfitting)
        lines.append(f"  ⚠ Underfitting (trees < {MIN_TREES_THRESHOLD}): window {ids}")

    # --- 3. Feature drift ---
    fd = results["feature_drift"]
    lines.append("")
    lines.append(f"3. FEATURE DRIFT (adjacent Top-{DRIFT_TOP_K} Jaccard)")
    lines.append("-" * w)
    pairs = fd.get("pairs", [])
    if pairs:
        lines.append(f"  {'Windows':>10}  {'Jaccard':>8}")
        for p in pairs:
            flag = " ⚠" if p["jaccard"] < 0.5 else ""
            lines.append(f"  {p['from_window']:>4}→{p['to_window']:<4}  {p['jaccard']:>8.3f}{flag}")
        lines.append(f"  Mean Jaccard: {fd['mean_jaccard']:.3f}")
    else:
        lines.append("  Not enough windows to compute drift.")

    # --- 4. Zero-importance features ---
    zi = results["zero_importance"]
    lines.append("")
    lines.append("4. ZERO-IMPORTANCE FEATURES")
    lines.append("-" * w)
    always = zi.get("always_zero", [])
    frequently = zi.get("frequently_zero", [])
    if always:
        lines.append(f"  Always zero (all {zi['n_windows']} windows): {', '.join(always)}")
    else:
        lines.append("  No features are zero across all windows.")
    if frequently:
        lines.append(f"  Frequently zero (>{FREQ_ZERO_THRESHOLD:.0%} windows): {', '.join(frequently)}")

    lines.append("")
    lines.append("=" * w)

    # Interpretation
    lines.append("")
    lines.append("Interpretation:")
    lines.append(f"  Trees < {MIN_TREES_THRESHOLD}: Early stopping too aggressive, model underfitting.")
    lines.append("  Drift Jaccard < 0.5: Model learning different patterns per period.")
    lines.append("  Always-zero features: Likely missing data (check DB).")

    return "\n".join(lines)
