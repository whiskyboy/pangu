# PanGu

[中文](README.md) | English

A personal China A-share quantitative trading signal system — weekly TopkDropout rebalance driven by LightGBM scoring + LLM bull/bear/judge debate, with automated Feishu (Lark) notifications.

> ⚠️ This system generates trading **signal suggestions** only. It does not connect to brokers or execute trades automatically. All investment decisions are your own.

## Features

- 📊 **Alpha158 factor engine + LightGBM training**: 191 factors (159 technical + 32 fundamental); monthly single-window training in production + multi-seed walk-forward for offline research (see [Offline Training & Backtest](#offline-training--backtest))
- 🤖 **LLM-TopkDropout weekly rebalance**: ML proposes BUY / SELL candidate pools; LLM bull/bear/judge picks the actual rebalance moves
- 🗃 **Virtual portfolio state**: latest holdings persisted as JSON, rebalanced on the first ISO-week trading day
- 🎯 **Executable signal enhancements**: rebalance card includes reference price + suggested share size + price-limit warning
- 🔁 **Decision replay**: `pangu replay` feeds historical `portfolio_snapshots` to the same backtest engine to estimate paper PnL
- 🚨 **Unified task monitoring**: every task wrapped by a decorator that captures exceptions → Feishu alert → `task_runs` history table
- ⏰ **APScheduler**: 6 tasks scheduled against the trading calendar; `pangu status` shows recent run history
- 🛠 **Full CLI**: data backfill / factor computation / model training / backtest / decision replay / evaluation / scheduler run

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Scheduling + Monitoring                            │
│  APScheduler │ @scheduled_task (alert+task_runs) │ CLI       │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Notification                                       │
│  Feishu Bot (notify_markdown / notify_text alert)            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Strategy & Decision                                │
│  MLScoringStrategy (LightGBM) + LLM Bull/Bear/Judge          │
│  PortfolioState (latest target JSON)                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Factor Engineering                                 │
│  Alpha158 (191) │ Technical (PandasTA) │ Fundamental         │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Data Infrastructure                                │
│  Market/News/Fundamental Provider (BaoStock + AkShare)       │
│  SQLite (daily_bars / fundamentals / trade_signals /         │
│          portfolio_snapshots / task_runs)                    │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

PanGu runs 6 scheduled tasks per trading day, forming a complete loop: data → model → signals.

> ℹ️ **Task numbering reflects logical dependencies** (data → model → signals), not scheduling order. The actual order on the 1st of each month is: T2 (hourly) → T5 (02:00) → T1 (06:00) → T4 (07:00) → T6 (08:15) → T3 (18:00).

### Data Collection

- **T1 Reference Data Sync** (06:00 on the 1st of each month): Trading calendar + index constituent snapshot
- **T2 News Polling** (hourly 00–23, runs on weekends too): Real-time financial headlines, deduplication + expiry; covers the overnight US session
- **T3 Domestic Market Sync** (18:00 trading days after close): Incremental daily K + fundamentals + valuation fields, BaoStock primary / AkShare fallback
- **T4 International Data Sync** (07:00 trading days): Fetches latest snapshots of US / HK / HSTECH / commodities into the DB; used by T6 as overnight LLM context

### Model Training

- **T5 Monthly Model Training** (02:00 on the 1st of each month): Single-window production training (full history + last N months as validation); artifact filenames remain compatible with `MLScorer.reload()`; runtime ≈ 15–20 minutes

### LLM-TopkDropout Weekly Rebalance

T6 (08:15 pre-market on every trading day) does two things:

1. **Data freshness self-check**: alerts if K-lines / global snapshot / news are stale
2. **Is-rebalance-day gate**: rebalance only on the first trading day of each ISO week; otherwise self-check only

Rebalance flow:

1. **ML scoring** — `MLScoringStrategy` runs the latest-window LightGBM model across the full pool
2. **Candidate pools** — lowest-scoring holdings → `SELL pool`, highest-scoring non-holdings → `BUY pool`
3. **LLM debate** — both pools + overnight global snapshot + last-24h news fed to `judge_rebalance`:
   - Bull: BUY / hold rationale per candidate
   - Bear: SELL / pass rationale per candidate
   - Judge: final BUY / SELL list (≤ `n_drop` each)
4. **ML fallback** — if LLM picks fewer than `n_drop`, fill the gap by ML rank; if LLM completely fails, degrade to classic ML-only TopkDropout
5. **Persist** — write `target_portfolio.json` + a `portfolio_snapshots` row tagged `is_rebalance=True`
6. **Push** — one consolidated Markdown card to Feishu containing reference price (T-1 close), suggested share size (equal-weight by `initial_capital / top_n`), estimated amount, and price-limit warning; plus a `trade_signals` audit row per decision

### Decision Replay and Monitoring

- **`pangu replay`** — Replays historical `portfolio_snapshots` through the same backtest engine to estimate paper PnL vs benchmark (replaces the old "weekly report" task). Real PnL should be tracked in your brokerage account
- **`pangu status`** — Shows DB stats + last 24h task runs + latest portfolio snapshot
- **Unified task monitoring** — every task is decorated by `@scheduled_task` (`src/pangu/tasks/_base.py`); exceptions are caught, a Feishu alert is pushed, and a `task_runs` row is persisted

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12+ |
| Package Manager | uv |
| Data Sources | BaoStock (primary) / AkShare (fallback) |
| Factor Computation | pandas + numpy + pandas-ta |
| Modeling | LightGBM (Walk-Forward ensemble) |
| LLM Interface | LiteLLM (Azure OpenAI / DeepSeek / Gemini) |
| Database | SQLite (WAL mode, raw sqlite3) |
| Notifications | lark-oapi (Feishu Bot SDK) |
| Scheduling | APScheduler |
| CLI | Click |
| Containerization | Docker |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Feishu (Lark) app credentials (App ID / App Secret, see [Feishu Bot Setup Guide](docs/feishu-bot-setup.md))
- Azure OpenAI or other LLM API key (see [LLM Provider Setup Guide](docs/litellm-setup.md))

### Installation

```bash
git clone <repo-url>
cd pangu

uv sync

cp .env.example .env
# Edit .env with your API keys and Feishu credentials
```

### CLI Usage

```bash
# Backfill (first deployment or long gap; `pangu run init` covers this in one go)
pangu backfill constituents --start 2019-01-01
pangu backfill bars --start 2019-01-01
pangu backfill fundamentals --start 2019-01-01
pangu backfill index --start 2019-01-01

# Offline training + evaluation (see "Offline Training & Backtest")
pangu compute-factors                      # Alpha158 → data/factors.parquet
pangu train                                # Single-window production training (full history)
pangu train walkforward                    # Multi-window walk-forward (research / backtest only)
pangu evaluate-scores --scores data/score_matrix_val.parquet
pangu evaluate-models --model-dir models
pangu backtest --scores data/score_matrix_val.parquet \
    --start <val_start> --end <val_end>
pangu replay --start 2026-01-01 --end 2026-05-15   # Replay historical decisions

# Runtime
pangu run init                             # One-shot cold-start (idempotent by default; --force re-runs everything)
pangu run signals [--initial-capital 200000]      # Manually trigger T6 (auto-runs T4 if today's snapshot is missing)
pangu run start [--initial-capital 200000]        # Launch the scheduler daemon
pangu status                               # DB stats + task_runs + portfolio snapshot
```

> Use `uv run pangu` if the virtual environment is not activated.

### Docker Deployment

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f worker
```

## Configuration

### Environment Variables (`.env`)

```bash
# LLM (see docs/litellm-setup.md)
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-key
AZURE_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT=your-deployment-name
DEEPSEEK_API_KEY=your-deepseek-key       # fallback
GEMINI_API_KEY=your-gemini-key           # fallback

# Feishu Bot (see docs/feishu-bot-setup.md)
FEISHU_APP_ID=cli_xxxx
FEISHU_APP_SECRET=your-secret
```

### Production Parameters (`config/settings.toml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stock_pool.indices` | `["000300","000905"]` | Index constituents used for the rebalance pool (CSI300 + CSI500) |
| `strategy.top_n` | 25 | TopkDropout target portfolio size |
| `ml.enabled` | `true` | Enable ML scoring (required for T6) |
| `ml.model_dir` | `models` | LightGBM model directory (T5 training output) |
| `ml.n_drop` | 3 | Per-rebalance turnover (max BUY/SELL count) |
| `ml.buy_candidate_size` | 10 | Top non-held candidates for the LLM BUY pool |
| `ml.sell_candidate_size` | 5 | Bottom-held candidates for the LLM SELL pool |
| `ml.val_months` | 3 | Tail validation window for T5 production training (months) |
| `portfolio.state_path` | `data/target_portfolio.json` | Latest target portfolio JSON path |
| `portfolio.initial_capital` | `100000.0` | Cold-start capital (CLI `--initial-capital` overrides) |
| `llm.provider` | `azure/$AZURE_DEPLOYMENT` | LiteLLM model identifier for `judge_rebalance` |

> Offline-training parameters (`n_seeds`, `time_decay_halflife`, …) — see [Offline Training & Backtest](#offline-training--backtest).

## Scheduled Tasks

Tasks run on the times configured under `[scheduler]` in `config/settings.toml`. Tasks marked "trading day" auto-skip non-trading days.

| Task | Time | Frequency | Description |
|------|------|-----------|-------------|
| **T1** Reference Data Sync | 06:00 | 1st of each month | Trading calendar + index constituent snapshot |
| **T2** News Polling | 00:00–23:00 hourly | Every hour (incl. weekends) | Financial headlines → dedup + expiry |
| **T3** Domestic Market Sync | 18:00 | Trading days (after close) | Daily K + fundamentals + valuation (incremental) |
| **T4** International Data Sync | 07:00 | Trading days | US / HK / commodity snapshots (overnight context) |
| **T5** Monthly Model Training | 02:00 | 1st of each month | Single-window LightGBM training (≈15–20min) |
| **T6** Signal Generation / Rebalance | 08:15 | Trading days (pre-market) | Self-check + LLM-TopkDropout rebalance on first ISO-week trading day |

> Task numbering reflects **logical dependencies** (data → model → signals), not scheduling order.

Every task is wrapped by `@scheduled_task`: exceptions auto-alert to Feishu and persist into `task_runs`. Use `pangu status` to view the last 24h of runs.

`pangu run signals` manually triggers T6 (auto-runs T4 first if today's snapshot is missing).

## Offline Training & Backtest

Offline training is separate from the production scheduler — used for first deployment, periodic review, and strategy research. Full methodology and experiment log: [`docs/ml-experiments.md`](docs/ml-experiments.md).

```bash
pangu compute-factors                          # Compute Alpha158 191 factors → data/factors.parquet

# Multi-window walk-forward (research only; takes hours)
pangu train walkforward --n-seeds 5            # → score_matrix_{val,test}.parquet + models/wf_window_*.txt

pangu evaluate-scores --scores data/score_matrix_val.parquet
pangu evaluate-models --model-dir models
pangu backtest \
    --scores data/score_matrix_val.parquet \
    --start <val_start> --end <val_end>        # Backtest / tune on the val window
pangu replay --start 2026-01-01 --end 2026-05-15   # Replay historical decisions through the engine
```

> The production T5 task uses **single-window** training (`pangu train`, full history + tail `ml.val_months` for validation), taking ≈15–20 minutes; walk-forward is research only, hence the subcommand name `pangu train walkforward`. Both produce identically-named artifacts (`wf_window_NN_seed*.txt`) that `MLScorer.reload()` can load.

| Knob | Description |
|------|-------------|
| `ml.n_seeds` | Seeds per window (default 5; use 1 for quick experiments) |
| `ml.time_decay_halflife` | Training-sample time decay half-life in trading days |
| `ml.first_train_start` | Walk-forward first training window start (default `2020-01-01`) |
| `ml.val_months` | Validation window length (months) for T5 / walk-forward |
| `scheduler.model_training_day` / `scheduler.model_training_time` | T5 retraining schedule |

> ⚠️ **Val/Test discipline**: use `score_matrix_val.parquet` for strategy selection / tuning and `score_matrix_test.parquet` for final reporting only — never select strategies based on test scores.


## Development

```bash
uv sync --extra dev

uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| LLM (Azure gpt-4o-mini, weekly rebalance) | ~$0.10 |
| VPS (2C4G) | ~$7-15 |
| Data sources (BaoStock/AkShare) | Free |
| **Total** | **< $20/month** |

## License

Private — personal use project
