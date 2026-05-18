# PanGu

[English](README_EN.md) | 中文

> 个人 A 股量化交易信号系统 — LightGBM 多因子打分 × LLM 牛熊辩论，每周自动把调仓建议推送到飞书。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-600%20passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

> ⚠️ 仅生成交易**信号建议**，不对接券商，不执行自动交易。投资决策请自行判断。

---

## ✨ 功能亮点

- 📊 **Alpha158 因子工程 × LightGBM**：191 个因子，生产线月度滚动训练，回测多 seed 集成
- 🤖 **LLM-TopkDropout 周度调仓**：ML 圈出候选池，LLM 牛 / 熊 / 裁判三角辩论决出 BUY/SELL
- 🎯 **可执行信号**：调仓推送含参考价、建议手数、涨跌停告警，飞书 Markdown 卡片一目了然
- 🔁 **决策回放**：`pangu replay` 复用回测引擎估算生产线纸面收益 vs 基准
- 🚨 **统一任务监控**：异常自动飞书告警 + 落 `task_runs` 历史，`pangu status` 一览全局
- ⏰ **APScheduler 调度**：6 个任务按交易日历自动调度，时区 / 时间均可配置
- 🛠 **CLI 工具齐全**：数据回填、因子计算、训练、回测、回放、评估一应俱全

## 🏛 系统架构

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

详见 [系统架构](docs/architecture.md)。

## 🚀 快速开始

```bash
# 1. 克隆并填写凭据
git clone <repo-url> && cd pangu
cp .env.example .env       # 编辑 .env：选一个 LLM provider + 飞书 App ID/Secret

# 2. 一次性 bootstrap：拉历史数据 + 训练首个模型（≈9-12h，可在 screen 中跑）
docker compose run --rm worker pangu run init

# 3. 启动 daemon
docker compose up -d
docker compose logs -f worker
```

> 不用 Docker？把 `docker compose run --rm worker` 换成 `uv run`，daemon 改为 `uv run pangu run start`。

更多场景（策略研究 / 回测）见 [部署指南](docs/deployment.md)。

## 📚 文档导航

| 文档 | 内容 |
|------|------|
| [部署指南](docs/deployment.md) | 场景 A（生产）/ 场景 B（研究） / Docker 完整流程 |
| [配置说明](docs/configuration.md) | `.env` + `settings.toml` 参数详解 |
| [调度任务详解](docs/scheduling.md) | 6 个定时任务、时间配置、手动触发 |
| [系统架构](docs/architecture.md) | 分层架构、工作原理、项目结构、设计约定 |
| [LLM Provider 配置](docs/litellm-setup.md) | Azure / DeepSeek / Gemini / OpenAI 切换 |
| [飞书 Bot 配置](docs/feishu-bot-setup.md) | 自建应用 + 私聊绑定步骤 |
| [ML 实验记录](docs/ml-experiments.md) | Walk-Forward 方法论、超参实验、Val/Test 铁律 |
| [Copilot Agents 指南](docs/copilot-agents-guide.md) | 仓库内 AI 协作流程 |

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| 包管理 | uv |
| 数据源 | AkShare (主) / BaoStock (回退) |
| 因子计算 | pandas-ta + pandas + numpy |
| LLM 接口 | LiteLLM (Azure OpenAI / DeepSeek / Gemini / OpenAI) |
| 数据库 | SQLite (本地持久化 + 增量缓存) |
| 推送 | lark-oapi (飞书 Bot SDK) |
| 调度 | APScheduler |
| CLI | Click |
| 容器化 | Docker |

## 📄 License

MIT — 详见 [LICENSE](LICENSE)。
