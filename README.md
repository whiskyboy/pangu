# PanGu

[English](README_EN.md) | 中文

个人 A 股量化交易信号系统 — 基于 LightGBM 多因子打分 + LLM 牛熊辩论的 TopkDropout 周度调仓，自动推送候选与调仓决策到飞书。

> ⚠️ 本系统仅生成交易**信号建议**，不对接券商、不执行自动交易。投资决策请自行判断。

## 功能概览

- 📊 **Alpha158 因子工程 + LightGBM 训练**：191 个因子（159 技术 + 32 基本面），生产线月度滚动训练 + 回测时多 seed 集成（详见 [离线训练与回测](#离线训练与回测)）
- 🤖 **LLM-TopkDropout 周度调仓**：ML 模型给出 BUY/SELL 候选池，LLM 牛/熊/裁判三角辩论从中精选要调仓的股票
- 🗃 **虚拟组合状态**：本地 JSON 记录最新持仓，每周 ISO 周首日按 TopkDropout 规则换仓
- 🎯 **可执行信号增强**：调仓推送包含参考价 + 建议手数 + 涨跌停告警
- 🔁 **决策回放**：`pangu replay` 用历史 `portfolio_snapshots` 喂同一回测引擎，估算生产线纸面收益
- 🚨 **统一任务监控**：所有任务通过装饰器自动捕获异常 → 飞书告警 → `task_runs` 持久化运行历史
- ⏰ **APScheduler 调度**：6 个任务按交易日历调度，`pangu status` 一览运行状态
- 🛠 **CLI 工具齐全**：数据回填 / 因子计算 / 模型训练 / 回测 / 决策回放 / 评估 / 调度运行

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: 调度 + 监控                                        │
│  APScheduler │ @scheduled_task (alert+task_runs) │ CLI       │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: 推送                                               │
│  Feishu Bot (notify_markdown / notify_text alert)            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 策略与决策                                         │
│  MLScoringStrategy (LightGBM)  +  LLM Bull/Bear/Judge        │
│  PortfolioState (latest target JSON)                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 因子工程                                           │
│  Alpha158 (191) │ Technical (PandasTA) │ Fundamental         │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 数据基建                                           │
│  Market/News/Fundamental Provider (BaoStock + AkShare)       │
│  SQLite (daily_bars / fundamentals / trade_signals /         │
│          portfolio_snapshots / task_runs)                    │
└─────────────────────────────────────────────────────────────┘
```

## 工作原理

PanGu 每个交易日按交易日历自动执行 6 个定时任务，组成"数据 → 模型 → 信号"完整闭环。

> ℹ️ **任务编号反映逻辑依赖顺序**（数据 → 模型 → 信号），不代表调度时间。每月 1 号实际执行顺序为：T2 (整点) → T5 (02:00) → T1 (06:00) → T4 (07:00) → T6 (08:15) → T3 (18:00)。

### 数据采集

- **T1 参考数据同步**（06:00 每月 1 号）：维护交易日历 + 指数成分股快照
- **T2 财经快讯轮询**（00–23 整点）：抓取财联社等快讯，去重入库 + 过期清理；非交易日也运行以覆盖隔夜美股窗口
- **T3 国内行情同步**（18:00 交易日盘后）：增量拉取股票池日 K + 基本面 + 估值字段，BaoStock 主源 / AkShare 兜底
- **T4 国际交易同步**（07:00 每交易日）：抓取美股/港股/恒生科技/商品的最新快照入库，作为 T6 LLM context 的隔夜参考

### 模型训练

- **T5 月度模型训练**（02:00 每月 1 号）：单窗口生产训练（用全部历史数据 + 最后 N 个月做验证），产物文件名兼容 `MLScorer.reload()`；月度运行 ≈15–20 分钟

### LLM-TopkDropout 周度调仓

T6（08:15 每个交易日盘前）做两件事：

1. **数据新鲜度自检**：K 线 / 全球快照 / 新闻是否到位，否则告警
2. **是否调仓日**：仅在每个 ISO 周首个交易日进入调仓流程，其它日只做自检

调仓流程：

1. **ML 打分** — `MLScoringStrategy` 用最新窗口的 LightGBM 模型对全池打分排序
2. **候选池构建** — 持仓中评分最低的若干股进入 `SELL pool`，未持仓中评分最高的若干股进入 `BUY pool`
3. **LLM 三角辩论** — 把两个候选池 + 隔夜全球行情 + 24h 快讯一起喂给 `judge_rebalance`：
   - 牛方（Bull）：列出每只候选的买入/不卖理由
   - 熊方（Bear）：列出每只候选的卖出/不买理由
   - 裁判（Judge）：综合输出最终 BUY / SELL 名单（最多 `n_drop` 只）
4. **ML 兜底** — 若 LLM 给出的 BUY/SELL 不足 `n_drop`，按 ML 排名补齐；若 LLM 整体失败则退化为经典 TopkDropout
5. **写入 PortfolioState + 持仓快照** — `target_portfolio.json` 记录新持仓，`portfolio_snapshots` 表记录调仓日的成分股
6. **推送飞书** — 一张 Markdown 卡片汇总所有 BUY / SELL，每项含参考价（T-1 close）、建议手数（按 `initial_capital / top_n` 等权重）、估算金额与涨跌停告警；同时写入 `trade_signals` 作审计

### 决策回放与监控

- **`pangu replay`**：用历史 `portfolio_snapshots` 复用同一回测引擎，估算生产线纸面收益 vs 基准（替代旧版"组合周报"任务）。实盘收益请在券商账户跟踪
- **`pangu status`**：显示 DB 统计 + 最近 24h 任务运行 + 最新持仓快照
- **统一任务监控**：所有任务被 `@scheduled_task` 装饰，异常自动写 `task_runs` 表 + 推送飞书告警

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
# 自选股管理
pangu pool list                            # 查看自选股
pangu pool add 600519 / 贵州茅台            # 添加（代码或名称）
pangu pool remove 600519                   # 移除

# 数据回填（首次部署或长时间间断后；推荐用 pangu run init 一键完成）
pangu backfill constituents --start 2019-01-01     # 同步历史成分股
pangu backfill bars --start 2019-01-01             # 拉历史日 K（耗时）
pangu backfill fundamentals --start 2019-01-01     # 拉历史基本面 (含 pub_date PIT)
pangu backfill index --start 2019-01-01            # 拉指数日 K

# 离线训练 + 评估（详见"离线训练与回测"小节）
pangu compute-factors                      # 计算 Alpha158 191 因子 → data/factors.parquet
pangu train                                # 生产单窗口训练（全部历史）→ models/wf_window_NN_seed*.txt
pangu train walkforward                    # 多窗口 Walk-Forward 训练（研究/回测用）
pangu evaluate-scores --scores data/score_matrix_val.parquet
pangu evaluate-models --model-dir models
pangu backtest --scores data/score_matrix_val.parquet \
    --start <val_start> --end <val_end>    # 调参用 val 回测
pangu replay --start 2026-01-01 --end 2026-05-15   # 用历史决策回放估算生产线收益

# 运行
pangu run init                             # 一键冷启动（默认幂等 skip if exists；--force 全量重跑）
pangu run signals [--initial-capital 200000]      # 手动触发 T6（缺 T4 快照则先跑 T4）
pangu run start [--initial-capital 200000]        # 启动调度器（daemon）

# 状态
pangu status                               # DB 统计 + 任务运行历史 + 持仓快照
```

> 未激活虚拟环境时，使用 `uv run pangu` 代替 `pangu`。

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
│   ├── settings.toml              # 主配置
│   ├── watchlist.yaml             # 自选股列表
│   └── global_market_mapping.yaml # 国际行情符号映射
├── src/pangu/
│   ├── cli.py                     # CLI 入口 (pool / run / status / backfill / train / backtest …)
│   ├── main.py                    # Components 组装 + 任务入口
│   ├── config.py                  # TOML 配置加载
│   ├── models.py                  # 数据模型 (TradeSignal, NewsItem, Action 等)
│   ├── scheduler.py               # APScheduler 调度
│   ├── tz.py / utils.py           # 交易日历 + 工具
│   ├── data/                      # Layer 1: 数据基建
│   │   ├── storage.py             #   SQLite 存储 (含 portfolio_snapshots / task_runs)
│   │   ├── market/                #   行情 (BaoStock + AkShare)
│   │   ├── fundamental/           #   基本面
│   │   ├── news/                  #   新闻
│   │   └── stock_pool/            #   股票池
│   ├── factor/                    # Layer 2: 因子工程
│   │   ├── technical.py           #   PandasTA 技术因子
│   │   ├── fundamental.py         #   基本面因子
│   │   ├── alpha158.py            #   Alpha158 191 因子引擎 (训练/打分用)
│   │   └── matrix.py              #   截面因子快照构建器 (LLM evidence pack)
│   ├── ml/                        # ML 训练 + 评估
│   │   ├── dataset.py             #   Walk-Forward 数据集
│   │   ├── model.py               #   LightGBM 包装
│   │   ├── scorer.py              #   滚动窗口集成打分
│   │   ├── score_evaluator.py     #   打分质量诊断
│   │   └── model_evaluator.py     #   模型质量诊断
│   ├── strategy/                  # Layer 3: 策略
│   │   ├── ml/ml_strategy.py      #   MLScoringStrategy + 候选池 + 兜底
│   │   └── llm/                   #   LLM 牛/熊/裁判辩论
│   │       ├── judge.py           #     judge_rebalance 主流程 + 解析 + 兜底
│   │       ├── prompts.py         #     系统/用户 prompt 模板
│   │       └── client.py          #     LiteLLM 封装（重试 + JSON 解析）
│   ├── portfolio/                 # 虚拟组合 JSON
│   ├── backtest/                  # 回测引擎
│   │   ├── engine.py              #   5 步 TopkDropout 回测
│   │   └── target_provider.py     #   TargetProvider 抽象 (ScoreBasedProvider / ReplayProvider)
│   ├── notification/              # Layer 4: 飞书推送
│   │   ├── manager.py             #   多通道分发
│   │   └── feishu.py              #   notify_markdown / notify_text
│   └── tasks/                     # 6 个调度任务
│       ├── _base.py               #   @scheduled_task (alert + task_runs)
│       ├── sync_reference_data.py #   T1
│       ├── poll_news.py           #   T2
│       ├── sync_domestic_market.py#   T3
│       ├── sync_global_market.py  #   T4
│       ├── update_model.py        #   T5 (月度训练)
│       └── generate_signals.py    #   T6 (调仓主流)
├── tests/                         # pytest 测试套件
├── data/                          # 运行时数据 (gitignored)
├── models/                        # 训练好的 LightGBM 模型 (gitignored)
├── docs/                          # 文档
├── Dockerfile / docker-compose.yml
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

### 生产环境关键参数 (`config/settings.toml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `stock_pool.indices` | `["000300","000905"]` | 调仓股票池所属指数 (CSI300 + CSI500) |
| `strategy.top_n` | 25 | TopkDropout 目标持仓数 |
| `ml.enabled` | `true` | 是否启用 ML 打分（T6 必需） |
| `ml.model_dir` | `models` | LightGBM 模型目录（T5 训练输出） |
| `ml.n_drop` | 3 | 单次调仓换仓数（最多 BUY/SELL 数） |
| `ml.buy_candidate_size` | 10 | 未持仓中 ML 排名靠前的 BUY 候选池大小 |
| `ml.sell_candidate_size` | 5 | 持仓中 ML 排名靠后的 SELL 候选池大小 |
| `ml.val_months` | 3 | T5 生产训练时切出的尾部验证窗口长度（月） |
| `portfolio.state_path` | `data/target_portfolio.json` | 当前虚拟组合 JSON 路径 |
| `portfolio.initial_capital` | `100000.0` | 冷启动总资金（CLI `--initial-capital` 可覆盖） |
| `llm.provider` | `azure/$AZURE_DEPLOYMENT` | LiteLLM 模型标识（judge_rebalance 使用） |

> 离线训练相关参数（n_seeds、`time_decay_halflife` 等）见 [离线训练与回测](#离线训练与回测)。

## 定时任务

系统启动后按 `config/settings.toml` 的 `[scheduler]` 段调度以下 6 个任务（标记"交易日"的任务在非交易日自动跳过）：

| 任务 | 时间 | 频率 | 说明 |
|------|------|------|------|
| **T1** 参考数据同步 | 06:00 | 每月 1 号 | 交易日历 + 指数成分股快照 |
| **T2** 财经快讯轮询 | 00:00–23:00 整点 | 每小时（含周末） | 财联社快讯 → 去重 + 过期清理 |
| **T3** 国内行情同步 | 18:00 | 每交易日盘后 | 日 K + 基本面 + 估值增量同步 |
| **T4** 国际交易同步 | 07:00 | 每交易日 | 美股/港股/商品快照入库（隔夜参考） |
| **T5** 月度模型训练 | 02:00 | 每月 1 号 | 单窗口 LightGBM 训练（≈15–20min） |
| **T6** 信号生成 / 调仓 | 08:15 | 每交易日盘前 | 数据自检 + ISO 周首日 LLM-TopkDropout 调仓 |

> 任务编号反映**逻辑依赖**（数据 → 模型 → 信号），不代表调度时间。

所有任务统一由 `@scheduled_task` 装饰：异常自动飞书告警 + 写 `task_runs` 历史。使用 `pangu status` 一览最近 24h 各任务运行情况。

`pangu run signals` 可手动触发 T6（若当天 T4 快照未生成会自动补跑）。

## 离线训练与回测

离线训练与生产调度分离，用于首次部署、定期复盘、策略研究。完整方法论 / 实验记录见 [`docs/ml-experiments.md`](docs/ml-experiments.md)。

```bash
pangu compute-factors                          # 计算 Alpha158 191 因子 → data/factors.parquet

# 多窗口 Walk-Forward（研究/回测专用，耗时数小时）
pangu train walkforward --n-seeds 5            # → score_matrix_{val,test}.parquet + models/wf_window_*.txt

pangu evaluate-scores --scores data/score_matrix_val.parquet
pangu evaluate-models --model-dir models
pangu backtest \
    --scores data/score_matrix_val.parquet \
    --start <val_start> --end <val_end>        # 用 val 数据回测 / 调参
pangu replay --start 2026-01-01 --end 2026-05-15   # 用历史决策回放估算生产线收益
```

> 生产线 T5 月度训练是**单窗口**训练（`pangu train`，全部历史 + 尾部 `ml.val_months` 验证），耗时 ≈15–20 分钟；Walk-Forward 仅用于离线研究，因此命名 `pangu train walkforward`。两者产物文件名一致（`wf_window_NN_seed*.txt`），均可被 `MLScorer.reload()` 加载。

| 配置项 | 说明 |
|------|------|
| `ml.n_seeds` | 每窗口集成的 seed 数（默认 5，调试可用 1 加速） |
| `ml.time_decay_halflife` | 训练样本时间衰减半衰期（交易日数） |
| `ml.first_train_start` | Walk-Forward 第一个训练窗口起点（默认 `2020-01-01`） |
| `ml.val_months` | T5 生产训练 / Walk-Forward 验证窗口长度（月） |
| `scheduler.model_training_day` / `scheduler.model_training_time` | T5 自动再训练的日期 / 时间 |

> ⚠️ **Val/Test 分离铁律**：用 `score_matrix_val.parquet` 调参，用 `score_matrix_test.parquet` 出最终报告——绝不能用 test 选策略。


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
