---
name: train-and-backtest
description: >
  Run the full PanGu ML pipeline: compute factors, walk-forward LightGBM training,
  backtest with score matrix, and summarize results. Use when the user wants to:
  (1) retrain the model end-to-end, (2) run a full train-then-backtest cycle,
  (3) evaluate model changes with a complete pipeline run.
  Triggers: "train and backtest", "full pipeline", "retrain model",
  "run walkforward", "end-to-end training", "train then backtest".
---

# Train and Backtest — Full Pipeline Skill

## Overview

This skill runs the complete PanGu ML pipeline: factor computation → walk-forward
LightGBM training → diagnostics → backtest → results summary. All commands use
CLI defaults unless the user specifies overrides.

## Prerequisites

- Working directory must be the `trading-agent` project root
- SQLite database (`data/pangu.db`) must contain daily bars and fundamentals data
- Package manager is `uv`, not pip

## Pipeline Steps

Execute these steps sequentially. After each step, verify success before proceeding.

### Step 1: Compute Factors

Compute Alpha158 (191 factors: 159 technical + 32 fundamental) for all stocks
in the full historical pool (CSI300+CSI500 union, ~1311 stocks).

```bash
uv run pangu compute-factors --output data/factors.parquet
```

**Expected output:** `data/factors.parquet` file created. Log shows 191 factor
columns and date range. Typical size ~1.3 GB for full universe.

**If the user already has a recent `data/factors.parquet`**, skip this step and
proceed to Step 2 — ask the user whether to recompute or reuse.

### Step 2: Walk-Forward Training

Train LightGBM across all walk-forward windows (default: 18-month train +
3-month validation + 3-month test, stepped by 3 months,
`--first-train-start 2020-01-01 --last-test-end 2025-12-31`).

**Stock pool filtering per split:**
- **Train**: union of all constituents during training period — maximizes sample count
- **Val/Test**: point-in-time constituents at period start — simulates real trading pool

```bash
uv run pangu train walkforward \
  --factors data/factors.parquet \
  --model-dir models
```

**Expected output:**
- Model files saved to `models/wf_window_*_seed*.txt` (17 windows × 5 seeds = 85 files by default)
- `data/score_matrix_test.parquet` and `data/score_matrix_val.parquet` created (date × symbol, seed-averaged scores)
- Per-window IC and Rank IC metrics (computed on seed-averaged predictions) printed to console

**After training, report these metrics to the user:**
- Mean IC and Rank IC across all windows
- Worst-performing windows (IC < 0.02)
- Windows with very few trees (< 50 iterations) — may indicate underfitting
- Any signs of IC degradation in recent windows vs earlier ones

### Step 3: Model & Score Diagnostics

Run diagnostics before backtest to catch data/model issues early.

```bash
# Model quality: feature importance, dead features, per-window summary
uv run pangu evaluate-models --model-dir models

# Score quality: discrimination, stability, rank stability
uv run pangu evaluate-scores --scores data/score_matrix_val.parquet
```

**Report to user:**
- Any dead features (`none` in feature_infos) — indicates all-NaN during training
- Score discrimination: top vs bottom quintile spread
- Score stability metrics

**If dead features found:** Investigate root cause (missing data backfill?)
before proceeding to backtest. Ask user whether to continue or fix data first.

### Step 4: Backtest

Run backtest using the trained model's score matrix. Use **val scores** for
strategy tuning; test scores only for final reporting.

Val and test score matrices cover different date ranges (determined by
walk-forward window configuration). Always check the score matrix's time range
first, then pass matching `--start` and `--end`:

```bash
# Check score matrix date range
python -c "import pandas as pd; s=pd.read_parquet('data/score_matrix_val.parquet'); print(s.index.min(), s.index.max())"

# Strategy tuning (use val, pass matching dates)
uv run pangu backtest \
  --scores data/score_matrix_val.parquet \
  --start <val_start> --end <val_end>
```

**Expected output:**
- Backtest metrics: Sharpe ratio, annual return, max drawdown, win rate, turnover
- Equity curve chart saved to `data/backtest_lgb_*.png`

### Step 5: Summary

Present a metrics table to the user:

| Metric | LGB Strategy |
|--------|-------------|
| Annual Return | |
| Sharpe Ratio | |
| Max Drawdown | |
| Win Rate | |
| Annual Turnover | |

Also summarize:
- IC/Rank IC trend across windows (improving, stable, or degrading?)
- Whether the ML model produces tradeable alpha (positive excess return after costs)
- Dead features and their impact
- Any concerning patterns (high turnover, drawdown concentration)

## User Overrides

The user may specify custom parameters. Map them to CLI flags:

| User says | CLI flag |
|-----------|----------|
| "use top 20 stocks" | `--top-n 20` |
| "start from 2022" | `--first-train-start 2022-01-01` |
| "initial capital 2M" | `--capital 2000000` |
| "use custom params" | `--params-file <path>` |
| "skip factor computation" | Omit Step 1, use existing `data/factors.parquet` |
| "no plot" | `--no-plot` |
| "use 5 seeds" | `--n-seeds 5` (CLI default) |

## Error Handling

- If `data/pangu.db` is missing: tell the user to run `uv run pangu backfill bars` first
- If factor computation fails with memory error: suggest reducing date range
- If training produces NaN IC: check for missing data in specific windows
- If backtest shows 0 trades: verify score_matrix date range covers backtest period
- If dead features found: likely data backfill issue — run `pangu evaluate-models` for details
