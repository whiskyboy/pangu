# 系统架构

[← 返回 README](../README.md)

PanGu 是一个 A 股个人量化交易信号系统，由两个子系统组成：每日异步运行的**生产信号管线**，和按需同步运行的**ML 训练与回测管线**。

## 分层架构

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

## 生产信号管线（每日异步）

PanGu 每个交易日按交易日历自动执行 6 个定时任务，组成"数据 → 模型 → 信号"完整闭环。任务编号反映**逻辑依赖顺序**（数据 → 模型 → 信号），不代表调度时间。每月 1 号实际执行顺序为：T2 (整点) → T5 (02:00) → T1 (06:00) → T4 (07:00) → T6 (08:15) → T3 (18:00)。

### 数据采集

- **T1 参考数据同步**（06:00 每月 1 号）：维护交易日历 + 指数成分股快照（含 cninfo 公司画像，用于 T6 LLM grounding）
- **T2 财经快讯轮询**（00–23 整点）：抓取财联社等快讯，去重入库 + 过期清理；非交易日也运行以覆盖隔夜美股窗口
- **T3 国内行情同步**（18:00 交易日盘后）：增量拉取股票池日 K + 基本面 + 估值字段，BaoStock 主源 / AkShare 兜底
- **T4 国际交易同步**（07:00 每交易日）：抓取美股 / 港股 / 恒生科技 / 商品的最新快照入库，作为 T6 LLM context 的隔夜参考

### 模型训练

- **T5 月度模型训练**（02:00 每月 1 号）：单窗口生产训练（用全部历史数据 + 最后 N 个月做验证），产物文件名兼容 `MLScorer.reload()`；月度运行 ≈15–20 分钟

### LLM-TopkDropout 周度调仓

T6（08:15 每个交易日盘前）做两件事：

1. **数据新鲜度自检**：K 线 / 全球快照 / 新闻是否到位，否则告警
2. **是否调仓日**：仅在每个 ISO 周首个交易日进入调仓流程，其它日只做自检

调仓流程：

1. **ML 打分** — `MLScoringStrategy` 用最新窗口的 LightGBM 模型对全池打分排序
2. **候选池构建** — 持仓中评分最低的若干股进入 `SELL pool`，未持仓中评分最高的若干股进入 `BUY pool`
3. **LLM 三角辩论** — 把两个候选池 + 隔夜全球行情 + 24h 快讯 + cninfo 公司画像一起喂给 `judge_rebalance`：
   - 牛方（Bull）：列出每只候选的买入 / 不卖理由
   - 熊方（Bear）：列出每只候选的卖出 / 不买理由
   - 裁判（Judge）：综合输出最终 BUY / SELL 名单（最多 `n_drop` 只）
4. **ML 兜底** — 若 LLM 给出的 BUY / SELL 不足 `n_drop`，按 ML 排名补齐；若 LLM 整体失败则退化为经典 TopkDropout
5. **写入 PortfolioState + 持仓快照** — `target_portfolio.json` 记录新持仓，`portfolio_snapshots` 表记录调仓日成分
6. **推送飞书** — 一张 Markdown 卡片汇总所有 BUY / SELL，每项含参考价（T-1 close）、建议手数（按 `initial_capital / top_n` 等权重）、估算金额与涨跌停告警；同时写入 `trade_signals` 作审计

### 决策回放与监控

- **`pangu replay`**：用历史 `portfolio_snapshots` 复用同一回测引擎，估算生产线纸面收益 vs 基准。实盘收益请在券商账户跟踪
- **`pangu status`**：显示 DB 统计 + 最近 24h 任务运行 + 最新持仓快照
- **统一任务监控**：所有任务被 `@scheduled_task` 装饰，异常自动写 `task_runs` 表 + 推送飞书告警（详见 [调度任务详解](scheduling.md)）

## ML 训练与回测管线（离线同步）

研究 / 回测专用，独立于生产调度。完整方法论 + 实验记录见 [`docs/ml-experiments.md`](ml-experiments.md)。

```
BaoStock/AkShare → SQLite (unadjusted prices + adj_factor + turn/tradestatus/is_st/valuation)
    → Alpha158Engine (191 factors)
    → LightGBM
        ├── 单窗口生产训练 (pangu train，T5 月度自动执行；≈15-20min)
        └── Walk-Forward 多窗口训练 (pangu train walkforward，离线研究专用；≈2.5h)
    → score_matrix_val.parquet + score_matrix_test.parquet
    → BacktestEngine (5-step rebalance, TargetProvider-based)
```

两条训练路径产物文件名一致（`wf_window_NN_seed*.txt`），均可被 `MLScorer.reload()` 加载（窗口 ID 单调递增，自动选最新）。

## 项目结构

```
pangu/
├── config/
│   ├── settings.toml              # 主配置
│   └── global_market_mapping.yaml # 国际行情符号映射
├── src/pangu/
│   ├── cli.py                     # CLI 入口 (run / status / backfill / train / backtest …)
│   ├── main.py                    # Components 组装 + 任务入口
│   ├── config.py                  # TOML 配置加载（含 $ENV_VAR 替换）
│   ├── models.py                  # 数据模型 (TradeSignal, NewsItem, Action 等)
│   ├── scheduler.py               # APScheduler 调度
│   ├── tz.py / utils.py           # 交易日历 + 工具
│   ├── data/                      # Layer 1: 数据基建
│   │   ├── storage.py             #   SQLite 存储 (含 stock_profiles / portfolio_snapshots / task_runs)
│   │   ├── market/                #   行情 (BaoStock + AkShare)
│   │   ├── fundamental/           #   基本面
│   │   ├── news/                  #   新闻
│   │   └── stock_pool/            #   股票池 + cninfo 画像
│   ├── factor/                    # Layer 2: 因子工程
│   │   ├── technical.py           #   PandasTA 技术因子
│   │   ├── fundamental.py         #   基本面因子
│   │   ├── alpha158.py            #   Alpha158 191 因子引擎 (训练 / 打分用)
│   │   └── matrix.py              #   截面因子快照构建器 (LLM evidence pack)
│   ├── ml/                        # ML 训练 + 评估
│   │   ├── dataset.py             #   Walk-Forward 数据集
│   │   ├── model.py               #   LightGBM 包装
│   │   ├── scorer.py              #   滚动窗口集成打分
│   │   ├── score_evaluator.py     #   打分质量诊断
│   │   └── model_evaluator.py     #   模型质量诊断
│   ├── strategy/                  # Layer 3: 策略
│   │   ├── ml/ml_strategy.py      #   MLScoringStrategy + 候选池 + 兜底
│   │   └── llm/                   #   LLM 牛 / 熊 / 裁判辩论
│   │       ├── judge.py           #     judge_rebalance 主流程 + 解析 + 兜底
│   │       ├── prompts.py         #     系统 / 用户 prompt 模板
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

## 设计约定

- **Protocol-driven**：所有可替换组件（数据 provider、策略、LLM、推送）通过 `typing.Protocol` 定义接口，不使用 ABC；新实现满足 Protocol 即可
- **Composite + Fallback**：数据 provider 链 (BaoStock → AkShare)，Composite 层负责 SQLite 缓存与增量同步
- **Components DI**：所有运行时依赖在 `main.py` 装配成一个 `Components` dataclass，注入 `TradingScheduler` 与各任务；无服务级单例；测试用 fake 替换
- **异步生产管线**：T1-T6 全部 `async`，APScheduler 使用 `AsyncIOScheduler`；因子计算与回测保持同步（纯计算）
- **LLM 降级保证**：`LLMJudgeEngineImpl.judge_rebalance` 永远返回 `RebalanceDecision`，从不抛异常；失败时 `source="llm_failed"`，T4 兜底为 ML 排名
- **数据持久化**：原生 `sqlite3` + WAL 模式 + 互斥锁，无 ORM；价格存未复权 + `adj_factor`，因子计算用前复权价格

详见 [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) 获取代码规范、价格惯例（unadjusted vs forward-adjusted）、PIT 合规等开发约定。
