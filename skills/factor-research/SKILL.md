---
name: factor-research
description: >
  Research, analyze, and develop alpha factors for the Alpha158 engine.
  Use when the user wants to: (1) analyze factor IC/Rank IC, (2) detect
  multicollinearity or redundancy, (3) design and implement new factors,
  (4) audit factors for lookahead bias, (5) prune low-value factors.
  Triggers: "factor research", "factor IC", "factor analysis",
  "new factor", "factor importance", "redundant factors", "lookahead check".
---

# Factor Research — Interactive Analysis & Development Skill

## Overview

This skill guides interactive factor research for the PanGu Alpha158 engine.
It covers analysis (IC, correlation, importance), development (new factors),
and validation (lookahead audit, test). Each step is presented to the user
for review before proceeding.

## Prerequisites

- Working directory: `trading-agent` project root
- `data/pangu.db` with daily_bars and fundamentals
- `data/factors.parquet` (compute first if missing)
- `data/score_matrix.parquet` and `models/` (for importance analysis)
- Package manager: `uv`

## Workflow A: Factor Analysis

### Step 0: Verify factors.parquet exists and is current

```bash
python3 -c "
import os, pandas as pd
if not os.path.exists('data/factors.parquet'):
    print('MISSING: data/factors.parquet does not exist')
else:
    panel = pd.read_parquet('data/factors.parquet')
    print(f'Shape: {panel.shape} ({panel.shape[1]} factors)')
    print(f'Date range: {panel.index.get_level_values(0).min()} to {panel.index.get_level_values(0).max()}')
"
```

**If missing or stale:** Ask the user whether to compute factors first:
```bash
uv run pangu compute-factors --output data/factors.parquet
```
**If factor count ≠ 177:** Warn that factors.parquet may be outdated (current Alpha158 produces 177).

### Step 1: Factor Coverage Check

Check NaN rates and coverage of all 177 factors.

```bash
uv run python3 -c "
import pandas as pd
panel = pd.read_parquet('data/factors.parquet')
print(f'Shape: {panel.shape}')
print(f'Date range: {panel.index.get_level_values(0).min()} to {panel.index.get_level_values(0).max()}')
nan_rates = panel.isna().mean().sort_values(ascending=False)
print('\nNaN rates (top 20):')
print(nan_rates.head(20).to_string())
print(f'\nFactors with >20% NaN: {(nan_rates > 0.2).sum()}')
print(f'Factors with >50% NaN: {(nan_rates > 0.5).sum()}')
"
```

**Present results to user.** Flag any factor with >20% NaN for investigation.

### Step 2: Factor IC Analysis

Compute per-factor IC and Rank IC against 5-day forward returns.

```bash
uv run python3 -c "
import pandas as pd
import numpy as np

panel = pd.read_parquet('data/factors.parquet')
# Load labels from score computation logic
from pangu.data.storage import Database
db = Database('data/pangu.db')

# Sample: compute rank correlation of each factor with future returns
# For full analysis, use the walk-forward windows
factors = panel.columns.tolist()
print(f'Analyzing {len(factors)} factors...')
# Group by date, compute cross-sectional Spearman correlation
# (Full implementation depends on label availability)
"
```

**Ask user:** Which time period to analyze? Full range or specific window?

### Step 3: Feature Importance from Trained Models

Extract LightGBM feature importance across all walk-forward windows.

```bash
uv run pangu evaluate-models --model-dir models
```

**Present to user:**
- Top 20 most important features (by split count and gain)
- Features with zero importance across all windows (candidates for pruning)
- Importance stability across windows

### Step 4: Multicollinearity Detection

```bash
uv run python3 -c "
import pandas as pd
import numpy as np

panel = pd.read_parquet('data/factors.parquet')
# Sample 50k rows for speed
sample = panel.sample(min(50000, len(panel)), random_state=42)
corr = sample.corr(method='spearman')

# Find highly correlated pairs (|corr| > 0.9)
pairs = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if abs(corr.iloc[i, j]) > 0.9:
            pairs.append((cols[i], cols[j], round(corr.iloc[i, j], 3)))

pairs.sort(key=lambda x: -abs(x[2]))
print(f'Highly correlated pairs (|r| > 0.9): {len(pairs)}')
for f1, f2, r in pairs[:30]:
    print(f'  {f1:30s} ↔ {f2:30s}  r={r:+.3f}')
"
```

**Ask user:** Should we prune redundant factors? Which ones to keep?

## Workflow B: New Factor Development

### Step 1: Design Discussion

**Ask user:**
- What market phenomenon does this factor capture?
- What data inputs does it need? (price, volume, fundamental, alternative)
- Expected correlation with existing factors?

### Step 2: Implementation

New factors go in `src/pangu/factor/alpha158.py`. Follow these rules:

- Output: wide-format `(date × symbol)` DataFrame
- Data type: `float32`
- Rolling calls: set `min_periods` correctly
- No lookahead: `factor[T]` uses only data from dates ≤ T
- Forward-adjusted prices: `close × adj_factor` for all price-based factors
- Volume/amount: use raw (unadjusted) values

**Present code diff to user for review before applying.**

### Step 3: Lookahead Audit

For each new factor `f[T]`, systematically verify:

1. **Data inputs**: All price/volume/fundamental data used has `date ≤ T`
2. **No forward-looking ops**: No `.shift(-N)`, no future index access
3. **Rolling windows**: `rolling(d)` with `min_periods=d` uses `[T-d+1, ..., T]`
4. **Fundamental ffill**: Only fills backward in time (past → present), never future
5. **Cross-sectional ops**: rank/zscore across stocks on same date T

#### System timing reference

```
Factor computation:  factor[T] uses data from dates ≤ T
Score generation:    score[T] = model.predict(factor[T])
Rebalance decision:  on date T, engine uses score[T-1] (prev_date)
Execution price:     open of date T
Label computation:   label[T] = excess_return[T+1 : T+5]
```

#### Known safe patterns
- `open[T] / close[T]` — both known at T close ✅
- `open[T] / preclose[T]` — both known at T open ✅
- `close[T].rolling(5).mean()` — uses [T-4, ..., T] ✅
- `fund.reindex(method='ffill')` — fills from past ✅

#### Known dangerous patterns
- `close.shift(-N)` — uses future close ⛔
- `label[T] = ret[T:T+5]` — intentional for labels only, never for features ⛔
- Using T's close for T's rebalance when execution is at T's open ⛔

#### Data timeliness (PIT compliance)

For quarterly factors (ROE, REVENUE_YOY, PROFIT_YOY, GROSS_MARGIN, etc.),
the effective date (ffill start) must be the **announcement date** (`pub_date`),
NOT the report period end date. Without PIT, Q4 annual reports can leak data
up to ~94 days early.

Verify:
1. `fundamentals.pub_date IS NOT NULL` for all quarterly rows
2. `pub_date >= date` for each row (announcement is after or on period-end)
3. `load_fundamentals_filled()` delays quarterly values to `pub_date`
4. Sample check: for 5 random stocks, verify `pub_date` matches BaoStock `pubDate`

```bash
# Quick PIT audit: check pub_date coverage
uv run python -c "
from pangu.data.storage import Database
db = Database('data/pangu.db')
db.init_tables()
total = db._conn.execute('SELECT COUNT(*) FROM fundamentals WHERE roe_ttm IS NOT NULL').fetchone()[0]
has_pub = db._conn.execute('SELECT COUNT(*) FROM fundamentals WHERE roe_ttm IS NOT NULL AND pub_date IS NOT NULL').fetchone()[0]
print(f'PIT coverage: {has_pub}/{total} quarterly rows have pub_date ({has_pub/total*100:.1f}%)')
"
```

### Step 4: Test & Validate

```bash
# Run factor tests
uv run pytest tests/test_factor/ -v

# Recompute factors with new factor included
uv run pangu compute-factors --output data/factors.parquet

# Quick IC check on new factor
uv run python3 -c "
import pandas as pd
panel = pd.read_parquet('data/factors.parquet')
new_factor = 'NEW_FACTOR_NAME'  # replace
print(f'{new_factor} NaN rate: {panel[new_factor].isna().mean():.4f}')
print(f'{new_factor} stats:\n{panel[new_factor].describe()}')
"
```

**Ask user:** Results look good? Proceed to retrain models?

## Critical Price Conventions

This is the most common source of bugs:

- **Factor computation uses FORWARD-ADJUSTED prices:** `close × adj_factor`
- **Volume and amount are NOT adjusted** — use raw values from DB
- **Labels use forward-adjusted prices:**
  `label[t] = (close × adj_factor)[t+5] / (close × adj_factor)[t] - 1 - benchmark_return`
- **NEVER mix adjusted and unadjusted prices** in the same calculation

## Factor Categories in Alpha158

| Category | Count | Examples |
|----------|-------|---------|
| KBar (candlestick shape) | 9 | KMID, KLEN, KUP, KLOW, KSFT |
| Price ratios | 5 | OPEN0, HIGH0, LOW0, VWAP0, OVERNIGHT_RET |
| Rolling simple | 55 | ROC, MA, STD, MAX, MIN, RSV, VMA (× 5 windows) |
| Rolling complex | 75 | RANK, IMAX, CORR, CNTP, SUMP (× 5 windows) |
| Rolling regression | 15 | BETA, RSQR, RESI (× 5 windows) |
| Fundamental | 18 | PE, PB, PS, PCF, ROE, REVENUE_YOY, PROFIT_YOY, LN_MKTCAP, TURNOVER, GROSS_MARGIN, NET_PROFIT_MARGIN, DEBT_RATIO, ASSET_TURNOVER, CURRENT_RATIO, EQUITY_YOY, ASSET_YOY, CASHFLOW_PER_SHARE, CASHFLOW_TO_PROFIT |

Rolling windows: {5, 10, 20, 30, 60} trading days. Total: 177 factors.

## Key Files

- `src/pangu/factor/alpha158.py` — Alpha158Engine (177 factors)
- `src/pangu/factor/technical.py` — PandasTA-based technical indicators (production)
- `src/pangu/factor/fundamental.py` — Fundamental factor engine
- `src/pangu/ml/model.py` — LGBModel with feature importance
- `src/pangu/ml/dataset.py` — Walk-forward window splitting and labels
- `tests/test_factor/test_alpha158.py` — Factor unit tests

## Commands Reference

```bash
# Compute all factors
uv run pangu compute-factors --output data/factors.parquet

# Run factor tests
uv run pytest tests/ -k "factor or alpha158"

# Train to evaluate factor impact
uv run pangu train walkforward --factors data/factors.parquet --output data/score_matrix.parquet

# Evaluate model feature importance
uv run pangu evaluate-models --model-dir models

# Evaluate score quality
uv run pangu evaluate-scores --scores data/score_matrix.parquet

# Lint and format
uv run ruff check src/pangu/factor/ && uv run ruff format src/pangu/factor/
```
