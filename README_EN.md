# PanGu

[中文](README.md) | English

A personal China A-share quantitative trading signal system — multi-factor strategy + LLM-powered decision engine with automated Feishu (Lark) notifications.

> ⚠️ This system generates trading **signal suggestions** only. It does not connect to brokers or execute trades automatically. All investment decisions are your own.

## Features

- 📊 **Cross-sectional factor ranking**: Technical + fundamental + macro factors, multi-factor scoring across the full stock pool
- 🤖 **LLM decision engine**: Per-stock evidence packages (factors + news + macro) → bull/bear/referee three-round debate → BUY/HOLD/SELL signals
- 🌍 **Global market integration**: US indices, Hong Kong HSI, commodities fed into macro factors; international news impact analysis on A-shares
- 🔔 **Feishu signal push**: Formatted signal cards via Feishu Bot (price, stop-loss, confidence, factor summary, news events)
- 📈 **Signal post-verification**: Automatic 1/3/5 trading day return tracking with strategy performance reports
- 🚨 **Error alerting**: Critical task failures auto-pushed to Feishu
- ⏰ **Auto scheduling**: APScheduler with trading calendar, 6 scheduled tasks, CLI single-run mode
- 🛠 **CLI management**: Command-line watchlist management, task execution, system status

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Notification & Scheduling                         │
│  Feishu Bot Push │ APScheduler │ CLI Tools                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Strategy & Decision                               │
│  Multi-Factor Ranking │ LLM Decision Engine (Bull/Bear/Ref) │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Factor Engineering                                │
│  Technical (pandas-ta) │ Fundamental │ Macro (Global Mkts)  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Data Infrastructure                               │
│  Market Provider (AkShare/BaoStock) │ News │ Stock Pool     │
│  SQLite Storage │ Trading Calendar │ Incremental Sync       │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

PanGu runs 6 scheduled tasks each trading day, forming a complete pipeline: data collection → factor computation → signal generation → post-verification.

### Data Collection

- **T1 Global Market Sync** (08:00): Fetches overnight data for US indices (S&P 500, Dow Jones, NASDAQ), Hong Kong HSI/HSTECH, VHSI volatility, and commodities (gold, oil, copper, etc.)
- **T2 News Polling** (hourly): Scrapes real-time financial headlines, deduplicates and stores, auto-cleans expired news
- **T3 Domestic Market Sync** (15:30 post-close): Incrementally pulls daily K-lines and fundamentals (PE, PB, ROE, etc.) for the full stock pool, with AkShare → BaoStock automatic fallback
- **T5 Reference Data Sync** (monthly): Updates A-share trading calendar and stock pool constituents

### Multi-Factor Ranking

The T4 signal generation task performs cross-sectional factor scoring across all stocks in the pool:

- **Technical factors** (62% weight): RSI(14), MACD histogram, Bollinger Band bias, OBV, ATR volatility, volume ratio
- **Fundamental factors** (25% weight): PE TTM, PB, ROE TTM
- **Macro factors** (5% weight): Computes weighted composites from global overnight changes — US overnight (S&P 500 × 40% + Dow × 30% + NASDAQ × 30%), HK HSI/HSTECH, VHSI volatility, and commodity price changes. A sector mapping table translates commodity movements into A-share sector adjustments (e.g., oil up → bullish for energy). A global risk composite score (risk-off assets like gold and VHSI carry negative weights) automatically raises the buy threshold when risk is elevated

All factors are Z-score normalized, weighted-summed, and min-max scaled to [0, 1]. Stocks are ranked by descending score, with the Top-N (default 10) advancing to LLM evaluation.

### LLM Decision Engine

Top-N candidates + watchlist stocks each receive an "evidence package" for a three-role LLM debate:

1. **Bull**: Presents bullish arguments based on strong technicals and positive news
2. **Bear**: Presents bearish arguments based on weak factors and negative events
3. **Referee**: Weighs both sides and renders a final BUY / SELL / HOLD verdict with confidence score

Evidence packages contain: factor scores and rankings, raw factor values (RSI, MACD, PE, etc.), stock-specific news and announcements, and global market snapshot. On LLM failure, the system automatically falls back to pure factor scoring (≥0.7 → BUY, ≤0.3 → SELL).

### Signal Push

Generated BUY/SELL signals are pushed via Feishu Bot DM, including stock info, suggested price, stop-loss level, confidence, factor summary, and key news events. Signals are persisted to the database with status tracking (new entry / sustained / exit).

### Post-Verification

T6 runs daily at 16:00, looking back at signals from 1, 3, and 5 trading days ago, comparing against actual closing prices to calculate returns. Strategy performance reports are pushed to Feishu (directional returns: BUY signals use positive returns, SELL signals use inverse returns).

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12+ |
| Package Manager | uv |
| Data Sources | AkShare (primary) / BaoStock (fallback) |
| Factor Computation | pandas-ta + pandas + numpy |
| LLM Interface | LiteLLM (Azure OpenAI / DeepSeek / Gemini) |
| Database | SQLite (local persistence + incremental cache) |
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
# Manage watchlist
pangu pool list                  # List watchlist stocks
pangu pool add 600519            # Add by symbol (auto-fetches historical data)
pangu pool add 贵州茅台           # Add by name (fuzzy match)
pangu pool remove 600519         # Remove from watchlist

# Run tasks
pangu run init                   # First-time init (sync calendar + market data)
pangu run once                   # Run all tasks once
pangu run start                  # Start scheduler (daemon mode)

# System status
pangu status                     # Database stats + strategy returns
```

> Use `uv run pangu` instead of `pangu` if the virtual environment is not activated.

### Docker Deployment

```bash
cp .env.example .env
# Edit .env with credentials

docker compose up -d             # Build and start
docker compose logs -f worker    # View logs
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

### Watchlist (`config/watchlist.yaml`)

```yaml
watchlist:
  - symbol: "600519"
    name: "贵州茅台"
    sector: "白酒"
  - symbol: "000858"
    name: "五粮液"
    sector: "白酒"
  # Recommended: 15-30 stocks across 3-5 sectors
```

### Key Parameters (`config/settings.toml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy.top_n` | 10 | Factor ranking Top-N for LLM evaluation |
| `strategy.buy_threshold` | 0.7 | Factor score > threshold → BUY signal |
| `strategy.sell_threshold` | 0.3 | Factor score < threshold → SELL signal |
| `llm.provider` | `azure/$AZURE_DEPLOYMENT` | LLM backend for decision engine |

## Scheduled Tasks

All tasks are scheduled according to the A-share trading calendar (skipped on non-trading days):

| Task | Time | Frequency | Description |
|------|------|-----------|-------------|
| T1 Global Market | 08:00 | Daily | US/HK/commodity overnight snapshot → macro factors |
| T2 News Polling | 07:00–20:00 | Hourly | Financial headlines → deduplicated storage |
| T3 Domestic Market | 15:30 | Daily | Stock pool K-lines + fundamentals (incremental) |
| T4 Signal Generation | 08:15 | Daily | Factor ranking → Top-N → evidence → LLM judge → Feishu push |
| T5 Reference Data | 1st of month | Monthly | Trading calendar + stock pool constituents |
| T6 Signal Verification | 16:00 | Daily | 1/3/5-day actual returns → performance report |

Use `pangu run once` to execute all tasks sequentially (T5→T1→T2→T3→T4→T6).

## Development

```bash
uv sync --extra dev

uv run pytest              # 409 tests
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| LLM (Azure gpt-4o-mini, ~10 calls/day) | ~$0.05 |
| VPS (2C4G) | ~$7-15 |
| Data sources (AkShare/BaoStock) | Free |
| **Total** | **< $15/month** |

## License

Private — personal use project
