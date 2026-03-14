# Copilot Instructions for PanGu

## Build, Test, Lint

```bash
uv sync --extra dev          # Install dependencies
uv run pytest                # Run all tests (~480 tests)
uv run pytest tests/test_backtest/test_engine.py -k "test_rebalance"  # Single test
uv run ruff check src/ tests/       # Lint
uv run ruff format src/ tests/      # Format
```

Python 3.12+ required. Package manager is `uv`, not pip/poetry. The `pangu` CLI is installed in both the uv venv (`.venv/bin/pangu`) and conda (`miniconda3/bin/pangu`). Use `uv run pangu <command>` or activate the venv first.

## Architecture

PanGu is an A-share quantitative trading system with two subsystems:

```
Production (daily, async):
  T1 global market → T2 news → T3 domestic bars+fundamentals
      → T4: factor scoring → top-N → LLM judge (bull/bear/judge debate) → Feishu push
      → T6: verify 1/3/5d returns

Offline (on-demand, sync):
  BaoStock/AkShare → SQLite DB (unadjusted prices + adj_factor)
      → Alpha158Engine (166 factors) → LightGBM Walk-Forward (17 windows)
      → score_matrix.parquet → BacktestEngine (5-step rebalance)
```

**Two subsystems:**

1. **Production signal pipeline** (daily, async):
   - `src/pangu/data/` — Data providers (BaoStock, AkShare, news) with Protocol interfaces, Composite fallback chains, SQLite caching, and incremental sync.
   - `src/pangu/factor/` — Factor engines: `technical.py` (PandasTA-based, 21 indicators), `fundamental.py`, `macro.py`. Pure functions, no DB/API calls.
   - `src/pangu/strategy/factor/` — MultiFactorStrategy: cross-sectional z-score → weighted sum → top-N candidates.
   - `src/pangu/strategy/llm/` — LLM Judge: three-role debate (bull/bear/judge) via LiteLLM. Always returns TradeSignal (rule-based fallback on LLM failure).
   - `src/pangu/tasks/` — T1 (global market sync), T2 (news polling), T3 (domestic market + fundamentals), T4 (signal generation), T5 (reference data sync), T6 (signal verification). All async, gated by trading calendar.
   - `src/pangu/notification/` — Feishu bot notifications with card format.
   - `src/pangu/scheduler.py` — APScheduler-based cron orchestration.

2. **ML training & backtest pipeline** (offline, sync):
   - `src/pangu/factor/alpha158.py` — 166-factor engine (158 technical + 8 fundamental). Wide-format vectorized pandas. Outputs MultiIndex(date, symbol) × 166 columns, float32.
   - `src/pangu/ml/dataset.py` — Walk-Forward window splitting, label computation (5-day excess return using forward-adjusted prices).
   - `src/pangu/ml/model.py` — LightGBM wrapper with fit/predict/save/load.
   - `src/pangu/backtest/engine.py` — 5-step rebalance: classify → sell → build pool → allocate → settle. Deterministic (all iterations sorted).

**Shared infrastructure:**
- `src/pangu/cli.py` — Click-based CLI. Commands use lazy imports to avoid circular deps.
- `src/pangu/data/storage.py` — Raw sqlite3 wrapper (no ORM). Thread-safe with WAL mode.
- `src/pangu/config.py` — TOML settings loader with `$ENV_VAR` substitution, thread-safe singleton.
- `src/pangu/models.py` — Shared dataclasses (TradeSignal, NewsItem, Action enum, etc.).

## Conventions

### Data & Storage

**Price data:** DB stores unadjusted (real market) prices. The `adj_factor` column enables forward-adjustment when needed. This is the most critical domain convention and a frequent source of bugs:
- **Factor computation** uses forward-adjusted prices: `close * adj_factor`. This ensures price continuity across ex-dividend dates.
- **Backtest execution** uses unadjusted prices directly — these are real trading prices for limit-up/down checks, order fills, and NAV calculation.
- **Label computation** uses forward-adjusted prices: `(close * adj_factor)[t+5] / (close * adj_factor)[t] - 1`. This captures dividend returns.
- **Never mix adjusted and unadjusted prices** in the same calculation.

**Fundamental data:** DB stores fundamentals **sparsely** — PE/PB are daily rows; ROE, revenue_yoy, profit_yoy, gross_margin etc. are quarterly (only on quarter-end dates). Two layers of forward-fill produce daily-frequency output:
1. `Database.load_fundamentals_filled()` fetches a seed row before the query range and ffills quarterly columns.
2. `Alpha158Engine._compute_fundamentals()` does a second ffill via `reindex(amount.index, method='ffill')` to align to trading dates.
- **Never judge data completeness by raw DB row counts.** Always check factor-level output (`factors.parquet`) or call `load_fundamentals_filled()`.

**Data pipeline layers:** When diagnosing data issues, check the right layer — DB sparsity ≠ factor sparsity:
```
DB (sparse quarterly) → load_fundamentals_filled (ffill) → Alpha158 (reindex ffill) → factors.parquet (dense daily)
```

**Return calculation:** All return metrics use `initial_capital` as the denominator, not the first day's NAV. This is because `initial_capital` is the known starting amount, while `nav[0]` may already reflect Day 1 trading costs and price changes.

**Financial rounding:** Use `Decimal` with `ROUND_HALF_UP` for price limit calculations (Chinese exchange 四舍五入), not Python's built-in `round()` (banker's rounding).

**Determinism:** All set/dict iterations in backtest code must use `sorted()` to ensure reproducible results regardless of PYTHONHASHSEED.

### Code Style & Architecture

**Protocol-driven architecture:** All layers use `typing.Protocol` for interfaces (not ABC). Data providers, strategies, LLM engine, notification — all define Protocol in `protocol.py` and implementations alongside. New implementations must satisfy the Protocol, not inherit from a base class.

**Composite + fallback chain:** Data providers use a Composite pattern: try BaoStock → fall back to AkShare. Composite handles SQLite caching and incremental sync via `data_sync_log` table. Adding a new provider means registering it in the chain.

**Components dependency injection:** All runtime dependencies are assembled into a `Components` dataclass in `main.py` and injected into `TradingScheduler` and tasks. Tests use fake implementations (`tests/fakes.py`). No global singletons for services.

**Async throughout production pipeline:** Tasks (T1-T6), LLM calls, and Feishu notifications are all `async`. The scheduler uses `AsyncIOScheduler`. Factor engines and backtest are synchronous (pure computation).

**LLM degradation guarantee:** `LLMJudgeEngineImpl.judge_stock()` always returns a `TradeSignal`, never None. Three-level JSON parsing fallback (direct → fenced block → brace-counting). If LLM completely fails, rule-based keyword scoring takes over.

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
- `config/watchlist.yaml` — Stock watchlist.
- `.env` — Environment variables (copy from `.env.example`): `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION`, `AZURE_DEPLOYMENT` (LLM), `FEISHU_APP_ID`, `FEISHU_APP_SECRET` (notification).
- Stock pool indices are configurable: `[stock_pool].indices = ["000300", "000905"]` (CSI300 + CSI500).

## Deployment

Docker deployment with `docker-compose.yml`. The container runs `pangu run start` (scheduler daemon mode).

```bash
cp .env.example .env   # Fill in API keys
docker compose up -d    # Start
```

Volumes: `./data:/app/data` (SQLite DB), `./config:/app/config` (settings/watchlist).

## Data Files (not in git)

- `data/pangu.db` — SQLite database (daily_bars, fundamentals, index_constituents, etc.)
- `data/factors.parquet` — Pre-computed 166-factor panel (~1.3GB)
- `data/score_matrix.parquet` — Model predictions (date × symbol)
- `models/wf_window_*.txt` — LightGBM model files (17 Walk-Forward windows)
