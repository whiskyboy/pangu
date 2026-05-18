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

### Usage Scenarios

PanGu supports two distinct workflows. Pick your target scenario before reading further:

| Scenario | Goal | Typical Flow |
|----------|------|--------------|
| **A. Out-of-the-box (production)** | Run the daily model, generate rebalance decisions, push Feishu cards | `pangu run init` (≈9-12h one-shot bootstrap) → (optional) `pangu run signals` dry-run → `pangu run start` or `docker compose up -d` (daemon) |
| **B. Strategy research / backtest** | Recompute factors + train ML + iterate strategy logic across multiple windows | `pangu backfill *` → `pangu compute-factors` → `pangu train walkforward` → `pangu evaluate-* / pangu backtest` |

Notes to avoid common pitfalls:

- **First deployment MUST run `pangu run init` first**: it backfills historical bars + fundamentals (≈9-12h, upstream API throttled), computes factors, and trains the first model. The daemon will NOT do this for you — it only handles incremental sync and scheduled tasks.
- **When does the first Feishu signal arrive?** After the daemon starts, T6 self-checks at 08:15 every trading day, but only the **first trading day of each ISO week** (usually Monday) pushes a rebalance card. Other trading days only refresh ML rankings without notifications.
- **T6 only runs on A-share trading days** — weekends and holidays are skipped.
- **Val/Test golden rule**: use `score_matrix_val.parquet` for hyperparameter tuning / strategy selection; reserve `score_matrix_test.parquet` for the final report — **never use the test set to select strategies** (see [Offline Training & Backtest](#offline-training--backtest) below).
- **Production T5 (monthly auto-retrain) = single-window** `pangu train` (~15–20 min). **Walk-Forward = multi-window** `pangu train walkforward` (~2.5h, research only). Both produce identically named `wf_window_NN_seed*.txt` artifacts loadable by `MLScorer.reload()`.
- **`pangu backtest --scores` is a required flag** (no default): pass `score_matrix_val.parquet` to tune, `score_matrix_test.parquet` for the final report. `--start / --end` must fall within the score file's date range (the CLI auto-clamps but explicit dates are preferred).

### Prerequisites

- Python 3.12+ or Docker
- [uv](https://docs.astral.sh/uv/) package manager (non-Docker path)
- Disk ≈ 5GB (SQLite DB ~1-2GB + factors.parquet ~1.3GB + models ~50-200MB + headroom)
- Feishu (Lark) app credentials (App ID / App Secret, see [Feishu Bot Setup Guide](docs/feishu-bot-setup.md))
- An LLM API key — pick **one** of Azure OpenAI / DeepSeek / Gemini / OpenAI / etc. (see [LLM Provider Setup Guide](docs/litellm-setup.md))

### Installation

```bash
git clone <repo-url>
cd pangu

uv sync

cp .env.example .env
# Edit .env: per docs/litellm-setup.md fill credentials for ONE LLM provider; add Feishu App ID / Secret
```

### Scenario A — Out-of-the-box (production)

```bash
# 1. One-shot bootstrap: 6 years of history + first model (≈9-12h; use screen / tmux)
#    Idempotent — re-running skips completed steps; `--force` re-runs everything.
uv run pangu run init

# 2. Optional: trigger a T6 dry-run to verify the Feishu push pipeline
#    (only pushes a card on rebalance days; otherwise just runs the self-check).
uv run pangu run signals

# 3. Launch the daemon (foreground or under systemd)
uv run pangu run start
# Or Docker — see "Docker Deployment" below.
```

### Scenario B — Strategy research / backtest

```bash
# 1. Backfill historical data (skip if Scenario A already ran)
uv run pangu backfill constituents --start 2019-01-01     # <1min → config/backfill_stock_pool.yaml
uv run pangu backfill bars --start 2019-01-01             # ≈4-7h (BaoStock throttled)
uv run pangu backfill fundamentals --start 2019-01-01     # ≈4-5h (incl. pub_date for PIT)
uv run pangu backfill index --start 2019-01-01            # <1min

# 2. Compute factors + walk-forward training (details in "Offline Training & Backtest")
uv run pangu compute-factors                              # ≈10min → data/factors.parquet
uv run pangu train walkforward --n-seeds 5                # ≈2.5h → 17 windows + score_matrix_{val,test}.parquet

# 3. Evaluate + backtest
uv run pangu evaluate-scores --scores data/score_matrix_val.parquet
uv run pangu evaluate-models --model-dir models
uv run pangu backtest --scores data/score_matrix_val.parquet --start <val_start> --end <val_end>     # tune
uv run pangu backtest --scores data/score_matrix_test.parquet --start <test_start> --end <test_end>  # final report
uv run pangu replay --start 2026-01-01 --end 2026-05-15   # replay historical decisions
```

### Other common commands

```bash
uv run pangu status                       # DB stats + last 24h task_runs + portfolio snapshot
uv run pangu train                        # Single-window production training (≈15-20min; same as T5)
```

> Skip the `uv run` prefix if your virtualenv is already activated.

### Docker Deployment

```bash
cp .env.example .env
# Edit .env: fill LLM provider credentials + Feishu credentials

# 1. First deployment MUST bootstrap first (one-shot, ≈9-12h; the terminal can disconnect)
docker compose run --rm worker pangu run init

# 2. Launch the daemon
docker compose up -d
docker compose logs -f worker
```

> Skipping `pangu run init` and starting the daemon directly leaves the system with no model / factors / bars — T5 and T6 will repeatedly fail and alert on Feishu.
> The container runs in UTC by default, but APScheduler honors `config/settings.toml::[system].timezone` (default `Asia/Shanghai`) — no need to set `TZ`.

## Configuration

### Environment Variables (`.env`)

PanGu uses LiteLLM to support multiple LLM providers — pick **one** and fill its credentials (these are alternatives, not a fallback chain). Then update `config/settings.toml::[llm].provider` to match (default `azure/$AZURE_DEPLOYMENT`). See [docs/litellm-setup.md](docs/litellm-setup.md) for the full provider list and switching guide.

```bash
# === LLM Provider — pick ONE ===

# Option A: Azure OpenAI (matches settings.toml default)
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-key
AZURE_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT=your-deployment-name

# Option B: DeepSeek
# DEEPSEEK_API_KEY=your-deepseek-key

# Option C: Google Gemini
# GEMINI_API_KEY=your-gemini-key

# Option D: OpenAI
# OPENAI_API_KEY=your-openai-key

# === Feishu Bot (required; see docs/feishu-bot-setup.md) ===
# The user only has to DM the Bot once to auto-bind — no open_id config required.
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

| Task | Default Time | Default Frequency | Config Key (`[scheduler]`) | Description |
|------|--------------|-------------------|----------------------------|-------------|
| **T1** Reference Data Sync | 06:00 | 1st of each month | `reference_data_sync_time` / `reference_data_sync_day` | Trading calendar + index constituent snapshot + cninfo company profiles |
| **T2** News Polling | 00:00–23:00 hourly | Every hour (incl. weekends) | `news_poll_start_time` / `news_poll_end_time` / `news_poll_interval_minutes` | Financial headlines → dedup + expiry |
| **T3** Domestic Market Sync | 18:00 | Trading days (after close) | `domestic_kline_sync_time` | Daily K + fundamentals + valuation (incremental) |
| **T4** International Data Sync | 07:00 | Trading days | `international_data_sync_time` | US / HK / commodity snapshots (overnight context) |
| **T5** Monthly Model Training | 02:00 | 1st of each month | `model_training_time` / `model_training_day` | Single-window LightGBM training (≈15–20min) |
| **T6** Signal Generation / Rebalance | 08:15 | Trading days (pre-market) | `signal_generation_time` | Self-check + LLM-TopkDropout rebalance on first ISO-week trading day |

> Task numbering reflects **logical dependencies** (data → model → signals), not scheduling order. All time fields use `"HH:MM"` format. The timezone is controlled globally by `[system].timezone` (default `Asia/Shanghai`) — containers / VPS in other system timezones do NOT need to set `TZ`; APScheduler fires according to the settings.toml timezone.

Example: shift the rebalance push to 09:00 and the domestic sync 4 hours later:

```toml
# config/settings.toml
[scheduler]
signal_generation_time = "09:00"
domestic_kline_sync_time = "22:00"
```

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
