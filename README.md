# TradingAgent

个人 A 股量化交易信号系统 — 基于多因子策略 + LLM 事件驱动，实时推送买卖信号到飞书和邮件。

> ⚠️ 本系统仅生成交易**信号建议**，不对接券商、不执行自动交易。投资决策请自行判断。

## 功能概览

- 📊 **因子量化策略**：~20 个技术因子 + 5 个基本面因子 + 6 个宏观因子，多因子打分排名
- 🤖 **LLM 事件驱动**：新闻/公告经 LLM 牛熊辩论分析，生成事件驱动信号
- 🌍 **全球市场联动**：美股、港股、大宗商品行情纳入因子体系，国际新闻分析对 A 股传导影响
- 🔔 **信号推送**：飞书 Bot + 邮件双通道，格式化信号卡片（含价格、止损止盈、置信度）
- 📈 **实时看板**：Streamlit Web Dashboard，行情监控 + 新闻流 + 信号历史
- ⏰ **自动调度**：盘前国际扫描 → 盘中实时轮询 → 盘后全面分析，交易日历自动控制

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: 用户交互                                           │
│  Streamlit Dashboard │ 飞书 Bot 推送 │ 邮件 │ CLI            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 策略与决策                                         │
│  因子策略引擎 │ LLM 事件引擎 │ 异动检测 → 信号融合器          │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 因子工程                                           │
│  技术因子 (pandas-ta) │ 基本面因子 │ 宏观因子 (国际行情)      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 数据基建                                           │
│  行情 Provider (AkShare/BaoStock) │ 新闻 Provider │ 股票池   │
│  SQLite 存储 │ 交易日历                                      │
└─────────────────────────────────────────────────────────────┘
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 包管理 | uv |
| 数据源 | AkShare (主) / BaoStock (回退) |
| 因子计算 | pandas-ta + pandas + numpy |
| LLM 接口 | LiteLLM (Azure OpenAI / DeepSeek / Gemini) |
| 数据库 | SQLite |
| Web UI | Streamlit |
| 推送 | lark-oapi (飞书 Bot SDK) + smtplib |
| 调度 | APScheduler |
| 日志 | loguru |
| 容器化 | Docker + Docker Compose |

## 快速开始

### 前置条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 包管理器
- Docker + Docker Compose（生产部署）
- 飞书开放平台自建应用（获取 App ID / App Secret，详见 [飞书 Bot 配置指南](docs/feishu-bot-setup.md)）
- Azure OpenAI 或其他 LLM API Key（配置指南见 [LLM Provider 配置指南](docs/litellm-setup.md)）

### 本地开发

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

# 运行
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

# 访问看板
# http://<your-server-ip>:8501
```

## 项目结构

```
trading-agent/
├── config/
│   ├── settings.toml              # 主配置文件
│   ├── watchlist.yaml             # 自选股列表
│   └── global_market_mapping.yaml # 国际行情→A股板块映射
├── src/trading_agent/
│   ├── models.py                  # 数据模型 (TradeSignal, NewsItem)
│   ├── config.py                  # 配置加载
│   ├── data/                      # Layer 1: 数据基建
│   │   ├── market.py              #   行情 Provider (A股+国际)
│   │   ├── fundamental.py         #   基本面 Provider
│   │   ├── news.py                #   新闻 Provider (国内+国际)
│   │   ├── stock_pool.py          #   股票池管理
│   │   └── storage.py             #   SQLite 存储
│   ├── factor/                    # Layer 2: 因子工程
│   │   ├── technical.py           #   技术因子 (~20个)
│   │   ├── fundamental.py         #   基本面因子
│   │   └── macro.py               #   宏观因子 (国际行情衍生)
│   ├── strategy/                  # Layer 3: 策略
│   │   ├── factor_strategy.py     #   多因子打分策略
│   │   ├── llm_engine.py          #   LLM 事件驱动 (牛/熊/裁判)
│   │   ├── anomaly_detector.py    #   异动检测器
│   │   └── signal_merger.py       #   信号融合器
│   ├── notification/              # Layer 5: 推送
│   │   ├── feishu.py              #   飞书 Bot 推送
│   │   └── email.py               #   邮件推送
│   ├── dashboard/                 # Layer 5: Web UI
│   │   └── app.py                 #   Streamlit 看板
│   ├── scheduler.py               # 任务调度
│   ├── cli.py                     # CLI 管理工具
│   └── main.py                    # 入口
├── tests/                         # 测试 (pytest)
├── data/                          # 运行时数据 (gitignored)
├── docker-compose.yml             # worker + dashboard 双服务
├── Dockerfile
├── pyproject.toml
└── .env.example                   # 环境变量模板
```

## 配置说明

### 环境变量 (`.env`)

```bash
# LLM（详见 docs/litellm-setup.md）
AZURE_API_KEY=your-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
DEEPSEEK_API_KEY=your-deepseek-key       # fallback
GEMINI_API_KEY=your-gemini-key           # fallback

# 飞书 Bot (私聊推送，配置指南见 docs/feishu-bot-setup.md)
# 用户私聊 Bot 即自动完成绑定，无需配置 open_id
FEISHU_APP_ID=cli_xxxx
FEISHU_APP_SECRET=your-secret

# 邮件
SMTP_HOST=smtp.example.com
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
| `strategy.buy_threshold` | 0.7 | 综合评分 > 此值发买入信号 |
| `strategy.sell_threshold` | 0.3 | 综合评分 < 此值发卖出信号 |
| `strategy.factor_weight` | 0.6 | 因子策略权重 |
| `strategy.event_weight` | 0.4 | 事件策略权重 |
| `scheduler.market_scan_interval_minutes` | 5 | 盘中扫描间隔 |
| `llm.provider` | `azure/gpt-4o-mini` | LLM 后端 |
| `llm.news_impact_threshold` | 7 | LLM 影响度 ≥ 此值才生成信号 |

## 定时任务

系统启动后自动按交易日历调度以下任务（非交易日全部跳过）：

| 任务 | 时段 | 频率 | 说明 |
|------|------|------|------|
| T1 国内新闻采集 | 07:00–20:00 | 每 30 分钟 | 财联社快讯 + 个股新闻 → LLM 分析 → 事件信号 |
| T2 国际新闻采集 | 07:00–20:00 | 每 30 分钟 | 国际财经新闻 → LLM 分析对 A 股影响 |
| T3 国内行情同步 | 15:30 | 每日 1 次 | A 股日 K 线增量同步 + 基本面检查 |
| T4 国际行情采集 | 08:00 + 盘中每 30 分钟 | 见左 | 美股/港股/商品快照 → 宏观因子 + 异动预警 |
| T5 股票池更新 | 15:30 (T3 后) | 每日 1 次 | 全量因子排名 + 公告分析 + 事件池管理 + 信号推送 |
| T6 盘中信号生成 | 09:30–15:00 | 每 5 分钟 | 实时行情 → 因子+异动+事件 → 信号融合 → 推送 |
| T7 交易日历同步 | 每月初 | 每月 1 次 | 同步 A 股交易日历 |

## CLI 工具

```bash
# 股票池管理
trading-agent pool list
trading-agent pool add 600519        # 添加自选股 (自动拉取历史数据初始化)
trading-agent pool remove 600519

# 手动触发 (对应定时任务)
trading-agent run news --domestic    # T1: 国内新闻采集+分析
trading-agent run news --global      # T2: 国际新闻采集+分析
trading-agent run market --domestic  # T3: 国内行情同步
trading-agent run market --global    # T4: 国际行情采集
trading-agent run update-pool        # T5: 股票池更新
trading-agent run scan               # T6: 盘中信号生成

# 个股诊断 (只读查询，不触发任务)
trading-agent inspect 600519     # 查看单只股票策略详情 (因子/事件/信号)

# 系统管理
trading-agent status             # 运行状态
trading-agent db stats           # 数据库统计
```

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

# 运行测试
uv run pytest

# 代码检查
uv run ruff check src/ tests/

# 格式化
uv run ruff format src/ tests/
```

## 运行成本估算

| 项目 | 月成本 |
|------|--------|
| LLM (Azure gpt-4o-mini, ~170 次/日) | ~$0.7 |
| VPS (2C4G) | ~¥50-100 |
| 数据源 (AkShare/BaoStock) | 免费 |
| **合计** | **< ¥120/月** |

## 路线图

- [ ] **Phase 1 (MVP)**：因子策略 + LLM 事件驱动 + 信号推送 + 实时看板
- [ ] **Phase 2**：回测引擎 (backtesting.py) + QLib Alpha158 因子移植 + Brave Search
- [ ] **Phase 3**：RD-Agent 自动因子发现 + 券商接口对接

## License

Private — 个人使用项目
