# PanGu

English | [中文](README.md)

> Personal A-share quant signal system — LightGBM multi-factor scoring × LLM bull/bear debate, auto-pushing weekly rebalance suggestions to Feishu.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-600%20passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

> ⚠️ Generates trading **signal suggestions** only. No broker integration, no auto-execution. Use at your own risk.

> 📖 Detailed docs are in Chinese only — see [`docs/`](docs/).

---

## ✨ Features

- 📊 **Alpha158 × LightGBM** — 191 factors, monthly rolling training, multi-seed ensemble for backtests
- 🤖 **LLM-TopkDropout weekly rebalance** — ML picks candidate pools, LLM bull/bear/judge debate finalizes BUY/SELL
- 🎯 **Actionable signals** — rebalance pushes include reference price, suggested lots, and price-limit warnings
- 🔁 **Decision replay** — `pangu replay` reuses the backtest engine to estimate paper P&L vs benchmark
- 🚨 **Unified monitoring** — all tasks auto-alert Feishu on failure + persist to `task_runs`; `pangu status` for a quick overview
- ⏰ **APScheduler-driven** — 6 tasks scheduled by the trading calendar, fully configurable times/timezone
- 🛠 **Full CLI** — backfill, compute-factors, train, backtest, replay, evaluate

## 🏛 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Scheduling + Monitoring                            │
│  APScheduler │ @scheduled_task (alert+task_runs) │ CLI       │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Notification                                       │
│  Feishu Bot (notify_markdown / notify_text alert)            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Strategy & Decision                                │
│  MLScoringStrategy (LightGBM)  +  LLM Bull/Bear/Judge        │
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

See [Architecture](docs/architecture.md).

## 🚀 Quickstart

```bash
# 1. Clone & fill credentials
git clone <repo-url> && cd pangu
cp .env.example .env       # edit .env: pick one LLM provider + Feishu App ID/Secret

# 2. One-time bootstrap: pull history + train first model (~9-12h; run in screen)
docker compose run --rm worker pangu run init

# 3. Start the daemon
docker compose up -d
docker compose logs -f worker
```

> Not using Docker? Replace `docker compose run --rm worker` with `uv run`; daemon is `uv run pangu run start`.

For research / backtest workflow, see [Deployment Guide](docs/deployment.md).

## 📚 Documentation

| Doc | Contents |
|------|------|
| [Deployment](docs/deployment.md) | Production (A) / Research (B) / Docker full walkthrough |
| [Configuration](docs/configuration.md) | `.env` + `settings.toml` parameters |
| [Scheduling](docs/scheduling.md) | 6 tasks, time config, manual triggers |
| [Architecture](docs/architecture.md) | Layered design, task flow, project tree, conventions |
| [LLM Setup](docs/litellm-setup.md) | Switching Azure / DeepSeek / Gemini / OpenAI |
| [Feishu Bot Setup](docs/feishu-bot-setup.md) | Custom app + chat binding |
| [ML Experiments](docs/ml-experiments.md) | Walk-Forward methodology, hyperparams, Val/Test rule |
| [Copilot Agents Guide](docs/copilot-agents-guide.md) | In-repo AI collaboration |

## 🛠 Tech Stack

| Category | Technology |
|------|------|
| Language | Python 3.12+ |
| Package manager | uv |
| Data sources | AkShare (primary) / BaoStock (fallback) |
| Factor computation | pandas-ta + pandas + numpy |
| LLM | LiteLLM (Azure OpenAI / DeepSeek / Gemini / OpenAI) |
| Database | SQLite (local persistent + incremental cache) |
| Notification | lark-oapi (Feishu Bot SDK) |
| Scheduling | APScheduler |
| CLI | Click |
| Container | Docker |

## 📄 License

MIT — see [LICENSE](LICENSE).
