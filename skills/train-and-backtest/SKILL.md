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
LightGBM training → backtest → results summary. All commands use CLI defaults
unless the user specifies overrides.

## Prerequisites

- Working directory must be the `trading-agent` project root
- SQLite database (`data/pangu.db`) must contain daily bars and fundamentals data
- Package manager is `uv`, not pip

## Pipeline Steps

Execute these steps sequentially. After each step, verify success before proceeding.

### Step 1: Compute Factors

Compute Alpha158 (166 factors) for all stocks in the configured pool.

```bash
uv run pangu compute-factors --output data/factors.parquet
```

**Expected output:** `data/factors.parquet` file created. Log shows factor count
and date range. Typical size ~1 GB for full CSI300+CSI500 universe.

**If the user already has a recent `data/factors.parquet`**, skip this step and
proceed to Step 2 — ask the user whether to recompute or reuse.

### Step 2: Walk-Forward Training

Train LightGBM across all walk-forward windows (default: 17 windows,
18-month train + 3-month validation + 3-month test, stepped by 3 months).

```bash
uv run pangu train walkforward \
  --factors data/factors.parquet \
  --model-dir models \
  --output data/score_matrix.parquet
```

**Expected output:**
- Model files saved to `models/wf_window_*.txt` (one per window)
- `data/score_matrix.parquet` created (date × symbol score matrix)
- Per-window IC and Rank IC metrics printed to console

**After training, report these metrics to the user:**
- Mean IC and Rank IC across all windows
- Worst-performing windows (IC < 0.02)
- Any signs of IC degradation in recent windows vs earlier ones

### Step 3: Backtest

Run backtest using the trained model's score matrix.

```bash
uv run pangu backtest \
  --strategy lgb \
  --scores data/score_matrix.parquet
```

**Expected output:**
- Backtest metrics: Sharpe ratio, annual return, max drawdown, win rate, turnover
- Equity curve chart saved to `data/backtest_lgb_*.png`

### Step 4: Baseline Comparison (Optional but Recommended)

Run the baseline (factor-only, no ML) backtest for comparison:

```bash
uv run pangu backtest --strategy baseline
```

### Step 5: Summary

Present a comparison table to the user:

| Metric | LGB Strategy | Baseline | Delta |
|--------|-------------|----------|-------|
| Annual Return | | | |
| Sharpe Ratio | | | |
| Max Drawdown | | | |
| Win Rate | | | |
| Annual Turnover | | | |

Also summarize:
- IC/Rank IC trend across windows (improving, stable, or degrading?)
- Whether the ML model adds value over baseline
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

## Error Handling

- If `data/pangu.db` is missing: tell the user to run `uv run pangu backfill bars` first
- If factor computation fails with memory error: suggest reducing date range
- If training produces NaN IC: check for missing data in specific windows
- If backtest shows 0 trades: verify score_matrix date range covers backtest period
