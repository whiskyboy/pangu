---
name: model-tuner
description: >
  Optimize LightGBM hyperparameters, fix training issues, and improve model quality.
  Use when the user wants to: (1) tune hyperparameters (early stopping, num_leaves, etc.),
  (2) switch to LambdaRank ranking mode, (3) run Optuna per-window optimization,
  (4) diagnose underfitting/overfitting windows, (5) experiment with multi-horizon labels.
  Triggers: "tune model", "hyperparameter", "early stopping", "LambdaRank",
  "Optuna", "underfitting", "overfitting", "model tuning", "optimize model".
---

# Model Tuner — Interactive Hyperparameter Optimization Skill

## Overview

This skill guides interactive model tuning for the PanGu LightGBM walk-forward
pipeline. It diagnoses training issues, proposes parameter changes, and validates
improvements — all with user review at each decision point.

## Prerequisites

- Working directory: `trading-agent` project root
- `data/factors.parquet` (191 factors, up to date)
- Existing models in `models/` and `data/score_matrix_test.parquet` (for baseline comparison)
- Package manager: `uv`
- Walk-forward defaults: `--first-train-start 2020-01-01 --last-test-end 2025-12-31`

## Workflow A: Diagnose Current Models

### Step 1: Model Quality Overview

```bash
uv run pangu evaluate-models --model-dir models
```

**Report to user:**
- Per-window metrics: IC, Rank IC, tree count, best iteration
- Windows with very few trees (< 50) → likely underfitting
- Windows with tree count = max iterations → no early stopping or needs more

### Step 2: Score Matrix Quality

```bash
uv run pangu evaluate-scores --scores data/score_matrix_val.parquet
```

**Report to user:**
- Score discrimination (top vs bottom quintile spread)
- Score stability (autocorrelation across dates)
- Rank stability (Kendall's tau between consecutive dates)

### Step 3: Feature Analysis

```bash
uv run python3 -c "
import glob
dead_features = {}
for f in sorted(glob.glob('models/wf_window_*.txt')):
    with open(f) as fh:
        names, infos = None, None
        for line in fh:
            if line.startswith('feature_names='):
                names = line.strip().split('=')[1].split()
            if line.startswith('feature_infos='):
                infos = line.strip().split('=')[1].split()
        if names and infos:
            for name, info in zip(names, infos):
                if info == 'none':
                    dead_features.setdefault(name, []).append(f.split('/')[-1])

if dead_features:
    print('Dead features (all-NaN during training):')
    for feat, windows in sorted(dead_features.items(), key=lambda x: -len(x[1])):
        print(f'  {feat:30s} dead in {len(windows)}/{len(glob.glob(\"models/wf_window_*.txt\"))} windows')
else:
    print('No dead features found.')
"
```

**Ask user:** Should we investigate dead features? (May need data backfill.)

## Workflow B: Early Stopping Fix

### Problem

Some walk-forward windows stop with very few trees (e.g., 1-16), indicating
premature early stopping — the validation loss fluctuates early and triggers
the default `early_stopping_rounds` too aggressively.

### Solution: MIN_ITERATIONS Guard

**Ask user:** Set minimum iterations to 50? (Prevents stopping before the model
has a chance to learn meaningful patterns.)

```python
# In src/pangu/ml/model.py, LGBModel.fit():
# Add: callbacks=[lgb.early_stopping(50, min_delta=0.0)]
# Or set params: min_iterations = 50
```

**Present code diff to user.** After approval:

```bash
uv run pytest tests/ -k "model" -v
uv run pangu train walkforward --factors data/factors.parquet --output data
```

**Compare:** Per-window tree counts before vs after.

## Workflow C: Hyperparameter Tuning

### Key LightGBM Parameters

| Parameter | Current | Suggested Range | Impact |
|-----------|---------|-----------------|--------|
| num_leaves | 31 | 15-63 | Model complexity |
| learning_rate | 0.05 | 0.01-0.1 | Convergence speed |
| min_child_samples | 20 | 10-100 | Overfitting control |
| feature_fraction | 0.8 | 0.5-1.0 | Feature bagging |
| bagging_fraction | 0.8 | 0.5-1.0 | Row bagging |
| reg_alpha | 0 | 0-10 | L1 regularization |
| reg_lambda | 0 | 0-10 | L2 regularization |
| num_iterations | 500 | 200-2000 | Max trees |

### Manual Tuning Flow

**Ask user which parameters to adjust.** For each change:

1. Present the proposed change and rationale
2. Apply change, retrain one representative window for quick validation
3. If promising, retrain all windows
4. Compare IC/Rank IC before and after

### Optuna Auto-Tuning (Per-Window)

For systematic optimization:

```python
# Pseudo-code for Optuna integration
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    # Train on train split, evaluate on val split
    # Return negative IC (minimize)
    ...

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

**Ask user:** How many trials? Which windows to optimize? Apply best params globally or per-window?

## Workflow D: LambdaRank Mode

### Motivation

Current objective is `regression` (MAE). For stock ranking, `lambdarank`
directly optimizes the ranking metric (NDCG), which better aligns with the
cross-sectional selection task.

### Implementation

```python
# Changes to src/pangu/ml/model.py
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10, 30],  # match top_n
    'label_gain': list(range(100)),  # relevance levels
}
# Labels need to be converted to relevance grades (e.g., quintile ranks 0-4)
```

**Ask user:** Switch to LambdaRank? This changes the loss function fundamentally.

**Steps:**
1. Present code changes for review
2. Retrain one window as quick test
3. Compare IC and score spread vs regression mode
4. If better, retrain all windows

## Workflow E: Multi-Horizon Labels

### Motivation

Current label: 5-day excess return. Short-term noise can degrade signal.
Blending multiple horizons (3d, 5d, 10d) reduces noise.

### Implementation

```python
# In src/pangu/ml/dataset.py
# Compute blended label:
# label = w_3d * ret_3d + w_5d * ret_5d + w_10d * ret_10d
# Default weights: 0.2, 0.5, 0.3 (emphasize 5d but smooth with 3d and 10d)
```

**Ask user:** What horizon weights to use? Default is 0.2/0.5/0.3.

## Validation Protocol

After any model change, always run this validation:

```bash
# 1. Retrain
uv run pangu train walkforward \
  --factors data/factors.parquet \
  --output data

# 2. Evaluate scores
uv run pangu evaluate-scores --scores data/score_matrix_val.parquet

# 3. Evaluate models
uv run pangu evaluate-models --model-dir models

# 4. Backtest comparison (check score date range first, pass matching --start/--end)
uv run pangu backtest --scores data/score_matrix_val.parquet --start <val_start> --end <val_end>
```

**Present comparison table:**

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Mean IC | | | |
| Mean Rank IC | | | |
| Min-window IC | | | |
| Sharpe Ratio | | | |
| Annual Return | | | |
| Max Drawdown | | | |

## Key Files

- `src/pangu/ml/model.py` — LGBModel (fit/predict/save/load)
- `src/pangu/ml/dataset.py` — Walk-forward windows, label computation
- `src/pangu/ml/score_evaluator.py` — Score quality diagnostics
- `src/pangu/ml/model_evaluator.py` — Model quality diagnostics
- `config/settings.toml` — Default model parameters

## Commands Reference

```bash
# Train all windows
uv run pangu train walkforward --factors data/factors.parquet --output data

# Evaluate models
uv run pangu evaluate-models --model-dir models

# Evaluate scores
uv run pangu evaluate-scores --scores data/score_matrix_val.parquet

# Backtest (check score date range first, pass matching --start/--end)
uv run pangu backtest --scores data/score_matrix_val.parquet --start <val_start> --end <val_end>

# Run model tests
uv run pytest tests/ -k "model or dataset" -v
```
