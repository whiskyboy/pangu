# Copilot Instructions for PanGu

## Build, Test, Lint

```bash
uv sync --extra dev          # Install dependencies
uv run pytest                # Run all tests
uv run pytest tests/test_backtest/test_engine.py -k "test_rebalance"  # Single test
uv run ruff check src/ tests/       # Lint
uv run ruff format src/ tests/      # Format
```

Python 3.12+ required. Package manager is `uv`, not pip/poetry. The `pangu` CLI is installed in both the uv venv (`.venv/bin/pangu`) and conda (`miniconda3/bin/pangu`). Use `uv run pangu <command>` or activate the venv first.

## Architecture

PanGu is an A-share quantitative trading system with two subsystems:

```
Production (daily, async, 6 tasks — numbered by logical dependency, not schedule order):
  T1 sync_reference_data   (06:00 on the 1st of each month)
  T2 poll_news             (hourly 00-23, runs on weekends too)
  T3 sync_domestic_market  (18:00 trading-day)
  T4 sync_global_market    (07:00 trading-day)
  T5 update_model          (02:00 on the 1st of each month — single-window production training)
  T6 generate_signals      (08:15 trading-day pre-market; rebalance only on ISO-week first trading day)

  Rebalance flow (T6): ML score pool → BUY pool (top-ranked non-held) +
      SELL pool (bottom-ranked held) → LLM bull/bear/judge debate →
      ML rank fallback for under-fill → update PortfolioState +
      portfolio_snapshots → push consolidated Feishu card with reference
      price (T-1 close) + suggested shares (equal-weight from initial_capital)
      + price-limit warning. NAV / weekly-report task removed.

Offline (on-demand, sync):
  BaoStock/AkShare → SQLite DB (unadjusted prices + adj_factor + turn/tradestatus/is_st/valuation)
      → Alpha158Engine (191 factors) → LightGBM (single-window for T5 prod / walk-forward for research)
      → score_matrix_val.parquet (+ score_matrix_test.parquet for walk-forward)
      → BacktestEngine (5-step rebalance, TargetProvider-based)
```

**Two subsystems:**

1. **Production signal pipeline** (daily, async):
   - `src/pangu/data/` — Data providers (BaoStock, AkShare, news) with Protocol interfaces, Composite fallback chains, SQLite caching, and incremental sync.
   - `src/pangu/factor/` — Factor engines: `technical.py` (PandasTA, 21 indicators), `fundamental.py`, `alpha158.py` (191 factors), `matrix.py` (per-symbol factor snapshot builder for the LLM evidence pack). Pure functions, no DB/API calls.
   - `src/pangu/strategy/ml/` — `MLScoringStrategy`: LightGBM ensemble scoring + BUY/SELL candidate pools + ML rank fallbacks. Module-level `pool_score_rank_maps(pool_df)` helper.
   - `src/pangu/strategy/llm/` — LLM Judge: `judge_rebalance` (bull/bear/judge pool-level debate) via LiteLLM. Single-stock LLM path removed.
   - `src/pangu/portfolio/` — `PortfolioState`: atomic JSON file holding latest virtual portfolio (date / symbols / scores / ranks).
   - `src/pangu/tasks/` — T1-T6. All wrapped by `@scheduled_task(task_id, name)` decorator from `_base.py` (log + alert + record_task_run).
   - `src/pangu/notification/` — Feishu bot: `notify_markdown` (rebalance card) and `notify_text` (alerts). Email module removed.
   - `src/pangu/scheduler.py` — APScheduler-based cron orchestration. `is_rebalance_day` (ISO-week first trading day) gates T6 rebalance.

2. **ML training & backtest pipeline** (offline, sync):
   - `src/pangu/factor/alpha158.py` — 191-factor engine (159 technical + 32 fundamental). Wide-format vectorized pandas. Outputs MultiIndex(date, symbol) × 191 columns, float32.
   - `src/pangu/ml/dataset.py` — Walk-Forward window splitting, label computation (5-day excess return using forward-adjusted prices). Train uses constituents union (all ~1311 stocks); val/test use point-in-time constituents (~800) to simulate real trading pool.
   - `src/pangu/ml/model.py` — LightGBM wrapper with fit/predict/save/load. Two entry points: `train()` (single-window for production / T5) and `train_walk_forward()` (multi-window for research). Both produce `wf_window_NN_seed*.txt` artifacts loadable by `MLScorer.reload()`.
   - `src/pangu/ml/score_evaluator.py` — Score matrix quality diagnostics (discrimination, stability, rank stability).
   - `src/pangu/ml/model_evaluator.py` — Model quality diagnostics (feature importance, per-window summary, feature drift).
   - `src/pangu/backtest/engine.py` — 5-step rebalance: classify → sell → build pool → allocate → settle. Deterministic (all iterations sorted). Consumes a `TargetProvider` (`target_provider.py`).
   - `src/pangu/backtest/target_provider.py` — `TargetProvider` Protocol + `ScoreBasedProvider` (scores → TopkDropout target) + `ReplayProvider` (historical decisions → target). Powers `pangu replay`.

**Shared infrastructure:**
- `src/pangu/cli.py` — Click-based CLI. Commands use lazy imports to avoid circular deps. `pangu status` reports DB stats + last-24h task runs + latest portfolio snapshot.
- `src/pangu/data/storage.py` — Raw sqlite3 wrapper (no ORM). Thread-safe with WAL mode. Versioned migrations via `PRAGMA user_version`. Key tables: `daily_bars`, `fundamentals`, `trade_signals` (decision audit log), `portfolio_snapshots` (date + symbols_json + is_rebalance — no NAV columns), `task_runs` (per-task run history).
- `src/pangu/config.py` — TOML settings loader with `$ENV_VAR` substitution, thread-safe singleton.
- `src/pangu/models.py` — Shared dataclasses (TradeSignal, NewsItem, Action enum, etc.).
- `src/pangu/tasks/_base.py` — `@scheduled_task` decorator: every task gets unified logging, exception handling, Feishu alert, and `task_runs` history row.

## Conventions

### Data & Storage

**Price data:** DB stores unadjusted (real market) prices. The `adj_factor` column enables forward-adjustment when needed. This is the most critical domain convention and a frequent source of bugs:
- **Factor computation** uses forward-adjusted prices: `close * adj_factor`. This ensures price continuity across ex-dividend dates.
- **Backtest execution** uses unadjusted prices directly — these are real trading prices for limit-up/down checks, order fills, and NAV calculation.
- **Label computation** uses forward-adjusted prices: `(close * adj_factor)[t+5] / (close * adj_factor)[t] - 1`. This captures dividend returns.
- **Never mix adjusted and unadjusted prices** in the same calculation.

**Fundamental data:** DB stores fundamentals **sparsely** — PE/PB/PS/PCF/market_cap are daily rows (extracted from daily bars by `composite._persist_valuation()`); ROE, revenue_yoy, profit_yoy, gross_margin etc. are quarterly (only on quarter-end dates). Two layers of forward-fill produce daily-frequency output:
1. `Database.load_fundamentals_filled()` fetches a seed row before the query range and ffills quarterly columns. **PIT-aware**: if `pub_date` is set and later than `date`, quarterly values are delayed to `pub_date` before ffill.
2. `Alpha158Engine._compute_fundamentals()` does a second ffill via `reindex(amount.index, method='ffill')` to align to trading dates.
- **Never judge data completeness by raw DB row counts.** Always check factor-level output (`factors.parquet`) or call `load_fundamentals_filled()`.

**Point-in-Time (PIT):** Quarterly financial data has a `pub_date` column (first announcement date from BaoStock `pubDate`). The ffill logic delays quarterly values so they only become effective from `pub_date`, not from the report period end. This prevents look-ahead bias (Q4 annual reports can be announced ~94 days after period-end). Backfill via `pangu backfill fundamentals --start 2019-01-01` (includes pub_dates automatically).

**Circulating market cap:** The system uses **circulating (float) market cap**, not total market cap. This is the A-share industry standard (Barra CNE5, Qlib Alpha158 all use float_mv). The derivation formula is a mathematical tautology from the exchange definition of turnover rate:
- `circ_shares = volume / (turn / 100)`, then `circ_mv = close × circ_shares`
- Computed in `composite._persist_valuation()`, stored in `fundamentals.market_cap`
- When `volume=0` or `turn=0` (suspended/no-trade days), `market_cap = NULL` — ffill carries forward the last known value
- **Never mix circulating and total market cap.** For SOE stocks, circ_mv can be 25-35% less than total_mv; this is correct behavior, not a bug.

**Data pipeline layers:** When diagnosing data issues, check the right layer — DB sparsity ≠ factor sparsity:
```
daily_bars (turn+close+volume) → composite._persist_valuation → fundamentals (PE/PB/PS/PCF/market_cap daily + quarterly sparse)
fundamentals → load_fundamentals_filled (ffill) → Alpha158 (reindex ffill) → factors.parquet (dense daily)
```

**Return calculation:** All return metrics use `initial_capital` as the denominator, not the first day's NAV. This is because `initial_capital` is the known starting amount, while `nav[0]` may already reflect Day 1 trading costs and price changes.

**Financial rounding:** Use `Decimal` with `ROUND_HALF_UP` for price limit calculations (Chinese exchange 四舍五入), not Python's built-in `round()` (banker's rounding).

**Determinism:** All set/dict iterations in backtest code must use `sorted()` to ensure reproducible results regardless of PYTHONHASHSEED.

### ML Training & Backtest

**Walk-Forward training:** Default 18-month train + 3-month val + 3-month test, stepped by 3 months, producing 17 windows (default `--first-train-start 2020-01-01 --last-test-end 2025-12-31`). Each window trains 5 models with different seeds (``--n-seeds 5``, default) and averages their predictions. Training uses all historical constituents union (~1311 stocks); val/test use point-in-time constituents (~800 stocks). Purged CV removes the last `label_horizon` days from training to prevent label leakage.

**Val/Test separation:** Val scores (`score_matrix_val.parquet`) are for strategy selection and hyperparameter tuning. Test scores (`score_matrix_test.parquet`) are for final reporting only — **never use test scores to select strategies or tune hyperparameters**. This prevents overfitting to the test period. Val and test score matrices cover different date ranges (determined by walk-forward window configuration). When backtesting, always check the score matrix's time range first and pass matching `--start` and `--end`. The CLI will auto-align if `--end` exceeds the score matrix, but explicit dates are preferred.

**LightGBM defaults:** `objective=mae, num_leaves=31, lr=0.02, subsample=0.8, colsample_bytree=0.7, min_child_samples=100, early_stopping=200, MIN_ITERATIONS=50`. The model typically stops at 50 trees — this is expected implicit regularization, not a bug.

**Backtest engine:** Weekly rebalance (first trading day of each ISO week), equal-weight, TopkDropout(top_n=30, n_drop=10). Trading costs: stamp tax 0.1% + commission 0.03% + slippage 0.1%. Excludes STAR Market (688/689 prefix).

**Multi-seed ensemble:** `--n-seeds 5` (CLI default) trains 5 models per window with seeds 0–4, averaging predictions. This reduces seed variance ~√5 and produces stable, reproducible scores. Use `--n-seeds 1` only for quick experiments. For rigorous A/B testing of strategy changes, compare paired backtest results (both runs with `--n-seeds 5`).

### Code Style & Architecture

**Protocol-driven architecture:** All layers use `typing.Protocol` for interfaces (not ABC). Data providers, strategies, LLM engine, notification — all define Protocol in `protocol.py` and implementations alongside. New implementations must satisfy the Protocol, not inherit from a base class.

**Composite + fallback chain:** Data providers use a Composite pattern: try BaoStock → fall back to AkShare. Composite handles SQLite caching and incremental sync via `data_sync_log` table. Adding a new provider means registering it in the chain.

**Components dependency injection:** All runtime dependencies are assembled into a `Components` dataclass in `main.py` and injected into `TradingScheduler` and tasks. Tests use fake implementations (`tests/fakes.py`). No global singletons for services.

**Async throughout production pipeline:** Tasks (T1-T6), LLM calls, and Feishu notifications are all `async`. The scheduler uses `AsyncIOScheduler`. Factor engines and backtest are synchronous (pure computation). Every task is wrapped by `@scheduled_task` (`src/pangu/tasks/_base.py`) which catches all exceptions, pushes a Chinese Feishu alert, and persists a `task_runs` row — never add an outer try/except in task code.

**LLM degradation guarantee:** `LLMJudgeEngineImpl.judge_rebalance` always returns a `RebalanceDecision`, never raises. On LLM failure the decision is tagged `source="llm_failed"` and T6 falls back to ML rank for SELL / BUY picks (equivalent to classic TopkDropout). Three-level JSON parsing fallback (direct → fenced block → brace-counting).

**Type hints:** Modern PEP 585+ syntax throughout — `list[str]`, `dict[str, float]`, `X | Y`. All public methods must have type annotations. Use `TYPE_CHECKING` blocks for circular dependency avoidance.

**Imports:** Absolute only (`from pangu.models import ...`). No relative imports. CLI command handlers use deferred imports inside functions to avoid circular deps.

**Logging:** Standard `logging` module with `logger = logging.getLogger(__name__)` per module.

**Code style:** Ruff with line-length=120. Minimal comments — only where clarification is needed. No over-engineering: keep code paths simple, avoid backward-compat shims, don't add parameters "just in case."

### Error Handling

Let exceptions propagate in core logic. Use explicit `raise ValueError`/`RuntimeError` for precondition failures. Broad `except` with logging is only acceptable in I/O boundaries (data providers, API calls). No defensive try/except in business logic — the user considers this "画蛇添足" (over-engineering). Data providers use `CircuitBreaker` (5 failures → 5min cooldown) and exponential backoff retry.

### Testing

Use deterministic fakes (`tests/fakes.py`), not mocks/patches. Tests are integration-style with real data structures. Use `pytest.approx()` for float comparisons. Call `reset_settings()` in tests to clear config singleton cache.

### Git Workflow

**Do NOT commit directly.** After making code changes, leave them staged/unstaged for the user to review before committing. Never run `git commit` unless the user explicitly asks to commit.

## Configuration

- `config/settings.toml` — Main config (DB path, stock pool indices, strategy params, scheduler times). Supports `$ENV_VAR` substitution.
- `.env` — Environment variables (copy from `.env.example`): `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION`, `AZURE_DEPLOYMENT` (LLM), `FEISHU_APP_ID`, `FEISHU_APP_SECRET` (notification).
- Stock pool indices are configurable: `[stock_pool].indices = ["000300", "000905"]` (CSI300 + CSI500). The pool is derived entirely from `index_constituents` — there is no manual watchlist.

## CLI Operations

Common commands and expected runtimes (800-stock pool, full date range from 2019):

| Command | Purpose | Runtime |
|---------|---------|---------|
| `pangu run init` | One-shot cold-start (backfill → compute-factors → train). Idempotent; `--force` re-runs all steps | varies |
| `pangu backfill constituents --start 2019-01-01` | Sync historical index constituents → `config/backfill_stock_pool.yaml` | <1min |
| `pangu backfill bars --start 2019-01-01 --force --pool config/backfill_stock_pool.yaml` | Re-fetch all daily bars with extended fields | ~4-7h (~18s/stock) |
| `pangu backfill fundamentals --start 2019-01-01` | Backfill quarterly fundamentals + gross_margin + pub_dates (PIT) | ~4-5h |
| `pangu backfill index --start 2019-01-01` | Backfill index daily bars (default: CSI300) | <1min |
| `pangu compute-factors` | Compute 191 Alpha158 factors → `data/factors.parquet` | ~10min |
| `pangu train` | Single-window production training (full history, no `train_months` arg) → `models/wf_window_NN_seed*.txt` | ~15-20min |
| `pangu train walkforward` | Multi-window walk-forward (research / backtest only) → score matrices + 17×5 model files | ~2.5h |
| `pangu backtest --scores data/score_matrix_val.parquet --start <val_start> --end <val_end>` | Backtest on val scores (策略调参) | <1min |
| `pangu backtest --scores data/score_matrix_test.parquet --start <test_start> --end <test_end>` | Backtest on test scores (最终报告) | <1min |
| `pangu replay --start <yyyy-mm-dd> --end <yyyy-mm-dd>` | Replay historical `portfolio_snapshots` through the backtest engine | <1min |
| `pangu evaluate-scores --scores data/score_matrix_val.parquet` | Score quality diagnostics | <10s |
| `pangu evaluate-models --model-dir models` | Model quality diagnostics (seed-averaged importance) | <10s |
| `pangu run signals [--initial-capital N]` | Manually trigger T6 (auto-runs T4 if today's snapshot is missing) | seconds |
| `pangu run start [--initial-capital N]` | Launch the APScheduler daemon (production entrypoint) | long-running |
| `pangu status` | DB stats + last 24h task runs + latest portfolio snapshot | <2s |

**`pangu train` vs `pangu train walkforward`:** Production T5 calls `train()` (single window, all history + tail `ml.val_months` for validation). `pangu train walkforward` is for offline research only — it builds 17 windows and is what generates the `score_matrix_{val,test}.parquet` consumed by `pangu backtest`. Both produce `wf_window_NN_seed*.txt` artifacts loadable by `MLScorer.reload()` (window IDs increase monotonically so the latest is auto-selected).

**Backfill operations:** Backfill is slow (bars ~4-7h, fundamentals ~2h) due to upstream API rate limiting. Use the `@backfill-manager` agent for planning, execution, monitoring, and verification. Key rules: use `screen` sessions (not Copilot `detach: true`), never kill mid-run (BaoStock TCP issues), run `constituents` first to get the full historical pool YAML.

## Deployment

Docker deployment with `docker-compose.yml`. The container runs `pangu run start` (scheduler daemon mode).

```bash
cp .env.example .env   # Fill in API keys
docker compose up -d    # Start
```

Volumes: `./data:/app/data` (SQLite DB), `./config:/app/config` (settings + index constituents cache).

## Data Files (not in git)

- `data/pangu.db` — SQLite database (daily_bars, fundamentals, index_constituents, etc.)
- `data/factors.parquet` — Pre-computed 191-factor panel (~1.3GB)
- `data/target_portfolio.json` — Current virtual portfolio state (latest holdings)
- `data/score_matrix_test.parquet` — Walk-forward model predictions on test set (date × symbol)
- `data/score_matrix_val.parquet` — Walk-forward model predictions on validation set (date × symbol)
- `models/wf_window_*_seed*.txt` — LightGBM model files. Production T5 keeps a single latest window; walk-forward research produces 17 windows × 5 seeds = 85 files.
