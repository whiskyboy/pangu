# TradingAgent

个人 A 股量化交易信号系统 — 基于多因子策略 + LLM 综合决策，自动推送买卖信号到飞书。

> ⚠️ 本系统仅生成交易**信号建议**，不对接券商、不执行自动交易。投资决策请自行判断。

## 功能概览

- 📊 **CSI300 跨截面因子排名**：技术因子 + 基本面因子 + 宏观因子，全 A 股 300 池多因子打分排名
- 🤖 **LLM 综合决策引擎**：逐股构建证据包（因子 + 新闻 + 宏观），牛方/熊方/裁判三轮辩论，输出 BUY/HOLD/SELL 信号
- 🌍 **全球市场联动**：美股三大指数、港股恒生、大宗商品实时行情纳入宏观因子，国际新闻分析对 A 股传导影响
- 🔔 **飞书信号推送**：飞书 Bot 私聊推送，格式化信号卡片（含价格、止损止盈、置信度、因子摘要、新闻事件）
- ⏰ **自动调度**：APScheduler 按交易日历调度 5 大任务，支持 `--once` 单次执行模式

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: 推送与调度                                         │
│  飞书 Bot 推送 │ APScheduler 调度器 │ Tasks                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 策略与决策                                         │
│  多因子排名策略 │ LLM 综合决策引擎 (牛方/熊方/裁判)            │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 因子工程                                           │
│  技术因子 (pandas-ta) │ 基本面因子 │ 宏观因子 (国际行情)      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 数据基建                                           │
│  行情 Provider (AkShare/BaoStock) │ 新闻 Provider │ 股票池   │
│  SQLite 存储 │ 交易日历 │ 增量同步 + 缓存                    │
└─────────────────────────────────────────────────────────────┘
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| 包管理 | uv |
| 数据源 | AkShare (主) / BaoStock (回退) |
| 因子计算 | pandas-ta + pandas + numpy |
| LLM 接口 | LiteLLM (Azure OpenAI / DeepSeek / Gemini) |
| 数据库 | SQLite (本地持久化 + 增量缓存) |
| 推送 | lark-oapi (飞书 Bot SDK) |
| 调度 | APScheduler |
| 日志 | loguru |
| 容器化 | Docker + Docker Compose |

## 快速开始

### 前置条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器
- 飞书开放平台自建应用（获取 App ID / App Secret，详见 [飞书 Bot 配置指南](docs/feishu-bot-setup.md)）
- Azure OpenAI 或其他 LLM API Key（配置指南见 [LLM Provider 配置指南](docs/litellm-setup.md)）

### 本地运行

```bash
# 克隆项目
git clone <repo-url>
cd trading-agent

# 安装依赖
uv sync

# 复制环境变量模板并填写
cp .env.example .env
# 编辑 .env，填入 API Key、飞书凭据等

# 配置自选股
# 编辑 config/watchlist.yaml

# 单次运行（首次推荐，顺序执行全部 5 个任务）
uv run python -m trading_agent.main --once

# 调度模式（按交易日历自动运行）
uv run python -m trading_agent.main
```

### Docker 部署

```bash
# 复制环境变量模板并填写
cp .env.example .env

# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f worker
```

## 项目结构

```
trading-agent/
├── config/
│   ├── settings.toml              # 主配置文件
│   ├── watchlist.yaml             # 自选股列表
│   └── global_market_mapping.yaml # 国际行情→A股板块映射
├── src/trading_agent/
│   ├── main.py                    # 入口 (--once / scheduler 模式)
│   ├── config.py                  # 配置加载
│   ├── models.py                  # 数据模型 (TradeSignal, NewsItem 等)
│   ├── scheduler.py               # APScheduler 调度器
│   ├── tz.py                      # 交易日历 & 时区工具
│   ├── utils.py                   # 公共工具 (CircuitBreaker, retry, throttle)
│   ├── data/                      # Layer 1: 数据基建
│   │   ├── storage.py             #   SQLite 存储层
│   │   ├── market/                #   行情数据
│   │   │   ├── protocol.py        #     MarketDataProvider 协议
│   │   │   ├── akshare.py         #     AkShare 实现 (A股+国际)
│   │   │   └── baostock.py        #     BaoStock 回退实现
│   │   ├── fundamental/           #   基本面数据
│   │   │   ├── protocol.py        #     FundamentalDataProvider 协议
│   │   │   └── akshare.py         #     AkShare 实现
│   │   ├── news/                  #   新闻数据
│   │   │   ├── protocol.py        #     NewsDataProvider 协议
│   │   │   └── akshare.py         #     AkShare 实现 (财联社+东财+国际)
│   │   └── stock_pool/            #   股票池
│   │       ├── protocol.py        #     StockPool 协议
│   │       └── yaml_pool.py       #     YAML 配置实现
│   ├── factor/                    # Layer 2: 因子工程
│   │   ├── technical.py           #   技术因子 (RSI, MACD, BBANDS 等)
│   │   ├── fundamental.py         #   基本面因子 (PE, PB, ROE)
│   │   └── macro.py               #   宏观因子 (国际行情衍生)
│   ├── strategy/                  # Layer 3: 策略与决策
│   │   ├── factor/                #   因子策略
│   │   │   ├── protocol.py        #     FactorStrategy 协议
│   │   │   └── multi_factor.py    #     多因子打分排名策略
│   │   └── llm/                   #   LLM 决策引擎
│   │       ├── client.py          #     LiteLLM 客户端 (重试+回退)
│   │       ├── judge.py           #     综合决策引擎 (牛/熊/裁判)
│   │       └── prompts.py         #     Prompt 模板
│   ├── notification/              # Layer 4: 推送
│   │   ├── protocol.py            #   NotificationProvider 协议
│   │   ├── manager.py             #   NotificationManager (多通道分发)
│   │   ├── feishu.py              #   飞书 Bot 推送
│   │   └── email.py               #   邮件推送
│   └── tasks/                     # 调度任务 (独立 async 函数)
│       ├── sync_global_market.py  #   T1: 全球市场同步
│       ├── poll_news.py           #   T2: 快讯采集
│       ├── sync_domestic_market.py#   T3: 国内行情同步
│       ├── generate_signals.py    #   T4: 信号生成
│       └── sync_reference_data.py #   T5: 交易日历同步
├── tests/                         # 407 测试 (pytest)
├── data/                          # 运行时数据 (gitignored)
├── docs/                          # 文档
│   ├── PRD.md                     #   产品需求文档
│   ├── feishu-bot-setup.md        #   飞书 Bot 配置指南
│   └── litellm-setup.md           #   LLM Provider 配置指南
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example                   # 环境变量模板
```

## 配置说明

### 环境变量 (`.env`)

```bash
# LLM（详见 docs/litellm-setup.md）
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-key
AZURE_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT=your-deployment-name
DEEPSEEK_API_KEY=your-deepseek-key       # fallback
GEMINI_API_KEY=your-gemini-key           # fallback

# 飞书 Bot（配置指南见 docs/feishu-bot-setup.md）
# 用户私聊 Bot 即自动完成绑定，无需配置 open_id
FEISHU_APP_ID=cli_xxxx
FEISHU_APP_SECRET=your-secret

# 邮件（可选）
SMTP_HOST=smtp.example.com
SMTP_PORT=465
SMTP_USER=your@email.com
SMTP_PASSWORD=your-password
NOTIFY_EMAIL=target@email.com
```

### 自选股 (`config/watchlist.yaml`)

```yaml
watchlist:
  - symbol: "600519"
    name: "贵州茅台"
    sector: "白酒"
  - symbol: "000858"
    name: "五粮液"
    sector: "白酒"
  # 建议 15-30 只，覆盖 3-5 个行业
```

### 关键参数 (`config/settings.toml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `strategy.top_n` | 10 | CSI300 因子排名取 Top-N 进入 LLM 决策 |
| `strategy.buy_threshold` | 0.7 | 因子评分 > 此值发买入信号 |
| `strategy.sell_threshold` | 0.3 | 因子评分 < 此值发卖出信号 |
| `llm.provider` | `azure/$AZURE_DEPLOYMENT` | LLM 后端 (综合决策引擎) |
| `llm.fallback_providers` | `deepseek, gemini` | LLM 回退链 |
| `stock_pool.factor_pool_universe` | `csi300` | 因子选股池 (CSI300 成分股) |

## 定时任务

系统启动后自动按交易日历调度以下任务（非交易日全部跳过）：

| 任务 | 时段 | 频率 | 说明 |
|------|------|------|------|
| T1 全球市场同步 | 08:00 | 每交易日 1 次 | 美股/港股/商品隔夜快照 → 宏观因子计算 |
| T2 快讯采集 | 07:00–20:00 | 每小时 | 财联社快讯 → 去重入库 + 过期清理 |
| T3 国内行情同步 | 15:30 | 每交易日 1 次 | CSI300 + 自选股日 K 线 + 基本面数据（增量缓存） |
| T4 信号生成 | 08:15 | 每交易日 1 次 | 因子排名 → Top-N → 证据包 → LLM 综合判断 → 飞书推送 |
| T5 交易日历同步 | 每月初 | 每月 1 次 | 同步 A 股交易日历 (7 天缓存) |

使用 `--once` 模式可按 T5→T1→T2→T3→T4 顺序单次执行全部任务。

## 信号推送示例

```
🟢 买入信号 | 2026-02-14 10:32

股票: 贵州茅台 (600519)
信号状态: ⚡ 首次入选 Top-N (新机会)
因子排名: #2 / 300
建议操作: 以 ¥1,850.00 买入
止损价: ¥1,795.50 (-2.95%)
置信度: ⭐⭐⭐⭐ (0.82)

📊 因子: RSI=42.3 (超卖) | MA20金叉 ✅ | 量比=1.8
📰 事件: [财联社] 茅台一季度营收同比增长18% → 利好 (8/10)

信号来源: 因子+事件 共振
```

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试 (407 tests)
uv run pytest

# 代码检查
uv run ruff check src/ tests/

# 格式化
uv run ruff format src/ tests/
```

## 运行成本估算

| 项目 | 月成本 |
|------|--------|
| LLM (Azure gpt-4o-mini, ~10 次/日) | ~$0.05 |
| VPS (2C4G) | ~¥50-100 |
| 数据源 (AkShare/BaoStock) | 免费 |
| **合计** | **< ¥100/月** |

## 路线图

- [x] **Phase 1 (MVP)**：因子策略 + LLM 综合决策 + 飞书推送 + 自动调度
- [ ] **Phase 2**：信号后验证 + CLI 管理工具 + Streamlit 看板 + 错误告警
- [ ] **Phase 3**：回测引擎 + QLib Alpha158 因子移植 + Brave Search

## License

Private — 个人使用项目
