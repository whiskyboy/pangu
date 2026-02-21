# PanGu

[English](README_EN.md) | 中文

个人 A 股量化交易信号系统 — 基于多因子策略 + LLM 综合决策，自动推送买卖信号到飞书。

> ⚠️ 本系统仅生成交易**信号建议**，不对接券商、不执行自动交易。投资决策请自行判断。

## 功能概览

- 📊 **CSI300 跨截面因子排名**：技术因子 + 基本面因子 + 宏观因子，全 A 股 300 池多因子打分排名
- 🤖 **LLM 综合决策引擎**：逐股构建证据包（因子 + 新闻 + 宏观），牛方/熊方/裁判三轮辩论，输出 BUY/HOLD/SELL 信号
- 🌍 **全球市场联动**：美股三大指数、港股恒生、大宗商品实时行情纳入宏观因子，国际新闻分析对 A 股传导影响
- 🔔 **飞书信号推送**：飞书 Bot 私聊推送，格式化信号卡片（含价格、止损止盈、置信度、因子摘要、新闻事件）
- 📈 **信号事后验证**：自动跟踪 1/3/5 个交易日实际收益率，推送策略回报
- 🚨 **错误告警**：关键任务失败自动推送飞书告警
- ⏰ **自动调度**：APScheduler 按交易日历调度 6 大任务，支持 CLI 单次执行模式
- 🛠 **CLI 管理工具**：命令行管理自选股、运行任务、查看系统状态

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: 推送与调度                                         │
│  飞书 Bot 推送 │ APScheduler 调度器 │ CLI 管理工具            │
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

## 工作原理

PanGu 每个交易日自动执行 6 个定时任务，形成完整的数据采集 → 因子计算 → 信号生成 → 事后验证闭环。

### 数据采集

- **T1 全球市场同步**（08:00）：采集美股三大指数（S&P 500、道琼斯、纳斯达克）、港股恒生/恒生科技、VHSI 波动率、黄金/原油/铜等大宗商品隔夜数据
- **T2 快讯采集**（每小时）：抓取财联社实时快讯，去重入库，自动清理过期新闻
- **T3 国内行情同步**（15:30 收盘后）：增量拉取股票池全量日 K 线和基本面数据（PE、PB、ROE 等），支持 AkShare → BaoStock 自动回退
- **T5 参考数据同步**（每月初）：更新 A 股交易日历和股票池成分股列表

### 多因子排名

T4 信号生成任务首先对股票池内所有股票进行跨截面因子打分：

- **技术因子**（62% 权重）：RSI(14)、MACD 柱、布林带偏离度、OBV、ATR 波动率、量比
- **基本面因子**（25% 权重）：PE TTM、PB、ROE TTM
- **宏观因子**（5% 权重）：采集全球市场隔夜涨跌幅，计算加权复合指标——美股隔夜（S&P 500 × 40% + 道指 × 30% + 纳指 × 30%）、港股恒生/恒生科技、VHSI 波动率、黄金/原油/铜等大宗商品涨跌。通过板块映射表将商品价格变动转化为对应 A 股板块的调整系数（如原油涨 → 利好能源板块），同时计算全球风险综合分（避险资产如黄金、VHSI 取负权重），当风险分过低时自动提高买入门槛

所有因子做 Z-score 标准化后加权求和，归一化到 [0, 1]，按得分降序排名取 Top-N（默认 10）。

### LLM 综合决策

进入 Top-N 的候选股票 + 自选股列表，逐只构建"证据包"送入 LLM 做三角辩论：

1. **牛方（Bull）**：基于技术面强势信号和利好新闻，陈述看涨理由
2. **熊方（Bear）**：基于因子弱势和利空事件，陈述看跌理由
3. **裁判（Judge）**：综合双方论点，给出最终 BUY / SELL / HOLD 判定和置信度

证据包包含：因子评分与排名、各因子原始值（RSI、MACD、PE 等）、个股新闻和公告、全球市场快照。LLM 失败时自动降级为纯因子评分决策（≥0.7 买入，≤0.3 卖出）。

### 信号推送

生成的 BUY/SELL 信号通过飞书 Bot 私聊推送，包含股票信息、建议价格、止损位、置信度、因子摘要和关键新闻事件。信号同时入库，追踪状态变化（首次入选 / 持续在榜 / 退出）。

### 事后验证

T6 每天 16:00 自动回看 1、3、5 个交易日前的信号，对比实际收盘价计算收益率，推送策略回报报告（方向性收益：买入信号取正收益，卖出信号取反向收益）。

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
| CLI | Click |
| 容器化 | Docker |

## 快速开始

### 前置条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器
- 飞书开放平台自建应用（获取 App ID / App Secret，详见 [飞书 Bot 配置指南](docs/feishu-bot-setup.md)）
- Azure OpenAI 或其他 LLM API Key（配置指南见 [LLM Provider 配置指南](docs/litellm-setup.md)）

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd pangu

# 安装依赖
uv sync

# 复制环境变量模板并填写
cp .env.example .env
# 编辑 .env，填入 API Key、飞书凭据等
```

### CLI 用法

```bash
# 管理自选股
pangu pool list                  # 查看自选股列表
pangu pool add 600519            # 按代码添加（自动拉取历史数据）
pangu pool add 贵州茅台           # 按名称添加（模糊匹配）
pangu pool remove 600519         # 移除自选股

# 运行任务
pangu run init                   # 首次初始化（同步日历 + 行情数据）
pangu run once                   # 单次执行全部任务
pangu run start                  # 启动调度器（daemon 模式）

# 系统状态
pangu status                     # 数据库统计 + 策略回报
```

> 如果未激活虚拟环境，使用 `uv run pangu` 代替 `pangu`。

### Docker 部署

```bash
cp .env.example .env
# 编辑 .env 填入凭据

docker compose up -d             # 构建并启动
docker compose logs -f worker    # 查看日志
```

## 项目结构

```
pangu/
├── config/
│   ├── settings.toml              # 主配置文件
│   ├── watchlist.yaml             # 自选股列表
│   └── global_market_mapping.yaml # 国际行情→A股板块映射
├── src/pangu/
│   ├── cli.py                     # CLI 入口 (pool/run/status)
│   ├── main.py                    # 组件构建 + 任务入口
│   ├── config.py                  # 配置加载
│   ├── models.py                  # 数据模型 (TradeSignal, NewsItem 等)
│   ├── scheduler.py               # APScheduler 调度器
│   ├── tz.py                      # 交易日历 & 时区工具
│   ├── utils.py                   # 公共工具 (CircuitBreaker, retry, throttle)
│   ├── data/                      # Layer 1: 数据基建
│   │   ├── storage.py             #   SQLite 存储层
│   │   ├── market/                #   行情数据 (AkShare + BaoStock 回退)
│   │   ├── fundamental/           #   基本面数据
│   │   ├── news/                  #   新闻数据 (财联社 + 东财 + 国际)
│   │   └── stock_pool/            #   股票池 (YAML 配置)
│   ├── factor/                    # Layer 2: 因子工程
│   │   ├── technical.py           #   技术因子 (RSI, MACD, BBANDS 等)
│   │   ├── fundamental.py         #   基本面因子 (PE, PB, ROE)
│   │   └── macro.py               #   宏观因子 (国际行情衍生)
│   ├── strategy/                  # Layer 3: 策略与决策
│   │   ├── factor/                #   多因子打分排名策略
│   │   └── llm/                   #   LLM 综合决策 (牛/熊/裁判)
│   ├── notification/              # Layer 4: 推送
│   │   ├── manager.py             #   多通道分发
│   │   └── feishu.py              #   飞书 Bot 推送
│   └── tasks/                     # 调度任务
│       ├── sync_global_market.py  #   T1: 全球市场同步
│       ├── poll_news.py           #   T2: 快讯采集
│       ├── sync_domestic_market.py#   T3: 国内行情同步
│       ├── generate_signals.py    #   T4: 信号生成
│       ├── sync_reference_data.py #   T5: 交易日历同步
│       └── verify_signals.py      #   T6: 信号事后验证
├── tests/                         # 409 测试 (pytest)
├── data/                          # 运行时数据 (gitignored)
├── docs/                          # 文档
├── Dockerfile
├── docker-compose.yml
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

## 定时任务

系统启动后自动按交易日历调度以下任务（非交易日全部跳过）：

| 任务 | 时段 | 频率 | 说明 |
|------|------|------|------|
| T1 全球市场同步 | 08:00 | 每交易日 1 次 | 美股/港股/商品隔夜快照 → 宏观因子计算 |
| T2 快讯采集 | 07:00–20:00 | 每小时 | 财联社快讯 → 去重入库 + 过期清理 |
| T3 国内行情同步 | 15:30 | 每交易日 1 次 | CSI300 + 自选股日 K 线 + 基本面数据（增量缓存） |
| T4 信号生成 | 08:15 | 每交易日 1 次 | 因子排名 → Top-N → 证据包 → LLM 综合判断 → 飞书推送 |
| T5 交易日历同步 | 每月初 | 每月 1 次 | 同步 A 股交易日历 + CSI300 成分股 |
| T6 信号验证 | 16:00 | 每交易日 1 次 | 1/3/5 交易日实际收益率 → 策略回报推送 |

使用 `pangu run once` 可按 T5→T1→T2→T3→T4→T6 顺序单次执行全部任务。

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
| LLM (Azure gpt-4o-mini, ~10 次/日) | ~$0.05 |
| VPS (2C4G) | ~¥50-100 |
| 数据源 (AkShare/BaoStock) | 免费 |
| **合计** | **< ¥100/月** |

## License

Private — 个人使用项目
