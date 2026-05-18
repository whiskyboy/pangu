# 部署指南

[← 返回 README](../README.md)

PanGu 支持两种使用路径，请先选择目标场景再继续。

| 场景 | 目标 | 典型流程 |
|------|------|---------|
| **A. 开箱即用（生产）** | 每日定时跑模型、生成调仓建议、飞书推送 | `pangu run init`（≈9-12h，一次性 bootstrap）→（可选）`pangu run signals` 验证 → `pangu run start` 或 `docker compose up -d`（daemon） |
| **B. 策略研究 / 回测** | 自己跑因子 + ML 训练 + 多窗口回测、迭代策略 | `pangu backfill *` → `pangu compute-factors` → `pangu train walkforward` → `pangu evaluate-* / pangu backtest` |

## 注意事项

- **首次部署务必先跑 `pangu run init`**：拉历史 K 线 + 基本面（≈9-12h，受上游 API 限流） + 计算因子 + 训练首个模型。daemon 不会替你做这些事，启动后只跑增量同步与定时任务。
- **首个调仓信号何时推送**：daemon 启动后，T6 会在每个交易日 08:15 自检；只有在 **ISO 周首个交易日**（通常周一）才推送调仓 Markdown 卡片。非调仓日只刷新 ML 排名、不会推送。
- **T6 仅在 A 股交易日运行**，节假日和周末跳过。
- **Val/Test 铁律**：`score_matrix_val.parquet` 用来调参 / 选策略；`score_matrix_test.parquet` 只用作最终报告——**绝不能用 test 选策略**。详见 [`docs/ml-experiments.md`](ml-experiments.md)。
- **生产 T5（月度自动再训练）= 单窗口** `pangu train`，约 15–20 分钟；**Walk-Forward = 多窗口** `pangu train walkforward`，约 2.5 小时，仅用于研究 / 回测。两者产物文件名一致（`wf_window_NN_seed*.txt`），都可被 `MLScorer.reload()` 加载。
- **`pangu backtest` 的 `--scores` 是必传参数**（不区分 val/test）：用 `score_matrix_val.parquet` 调参，用 `score_matrix_test.parquet` 出最终报告。`--start / --end` 必须落在该 score 文件的日期范围内（命令行会自动收敛到 score 范围，但显式指定更稳妥）。

## 前置条件

- Python 3.12+ 或 Docker（二选一）
- [uv](https://docs.astral.sh/uv/) 包管理器（非 Docker 路径）
- 磁盘空间 ≈ 5GB（SQLite DB ~1-2GB + factors.parquet ~1.3GB + models ~50-200MB + 备份余量）
- 飞书开放平台自建应用（获取 App ID / App Secret，详见 [飞书 Bot 配置指南](feishu-bot-setup.md)）
- LLM API Key（Azure OpenAI / DeepSeek / Gemini / OpenAI 等任选其一，配置指南见 [LLM Provider 配置指南](litellm-setup.md)）

## 安装

```bash
git clone <repo-url>
cd pangu

# 1. 安装依赖
uv sync

# 2. 复制并填写环境变量
cp .env.example .env
# 编辑 .env：选一个 LLM provider 填凭据；填飞书 App ID / Secret
```

详见 [配置说明](configuration.md)。

## 场景 A：开箱即用（生产）

```bash
# 1. 一键 bootstrap：拉 6 年历史数据 + 训练首个模型（≈9-12h；可在 screen / tmux 中跑）
#    步骤幂等，中断后重跑会自动跳过已完成的部分；`--force` 全量重跑
uv run pangu run init

# 2. 可选：手动触发一次 T6 验证飞书推送是否畅通（非调仓日只做自检，调仓日才推送卡片）
uv run pangu run signals

# 3. 启动 daemon（前台跑或 systemd 托管）
uv run pangu run start
```

或使用 Docker，见下文。

## 场景 B：策略研究 / 回测

```bash
# 1. 拉历史数据（如果已跑过场景 A 可跳过）
uv run pangu backfill constituents --start 2019-01-01     # <1min，得到 config/backfill_stock_pool.yaml
uv run pangu backfill bars --start 2019-01-01             # ≈4-7h，受 BaoStock 限流
uv run pangu backfill fundamentals --start 2019-01-01     # ≈4-5h，含 pub_date PIT
uv run pangu backfill index --start 2019-01-01            # <1min

# 2. 计算因子 + Walk-Forward 训练（≈2.5h；研究 / 回测专用）
uv run pangu compute-factors                              # ≈10min → data/factors.parquet
uv run pangu train walkforward --n-seeds 5                # → 17 窗口模型 + score_matrix_{val,test}.parquet

# 3. 评估 + 回测
uv run pangu evaluate-scores --scores data/score_matrix_val.parquet
uv run pangu evaluate-models --model-dir models
uv run pangu backtest --scores data/score_matrix_val.parquet --start <val_start> --end <val_end>   # 调参
uv run pangu backtest --scores data/score_matrix_test.parquet --start <test_start> --end <test_end> # 出报告
uv run pangu replay --start 2026-01-01 --end 2026-05-15   # 用历史调仓快照回放
```

完整方法论 / 实验记录见 [`docs/ml-experiments.md`](ml-experiments.md)。

## 其他常用命令

```bash
uv run pangu status                       # DB 统计 + 最近 24h 任务运行历史 + 持仓快照
uv run pangu train                        # 生产单窗口训练（≈15-20min；T5 月度任务也是它）
uv run pangu run signals                  # 手动触发 T6（若当天 T4 快照未生成会自动补跑）
```

> 已激活虚拟环境（如 `source .venv/bin/activate`）则可省略 `uv run` 前缀。

## Docker 部署

```bash
cp .env.example .env
# 编辑 .env 填入 LLM provider 凭据 + 飞书凭据

# 1. 首次部署必须先 bootstrap（在容器内一次性跑完，约 9-12h；终端可断开）
docker compose run --rm worker pangu run init

# 2. 启动 daemon
docker compose up -d
docker compose logs -f worker            # 查看日志
```

> 不先跑 `pangu run init` 直接 `docker compose up -d` 的话，daemon 启动后 T5/T6 会因为缺少历史数据 / 模型反复告警飞书。
> 容器内默认 UTC 时区，但 APScheduler 调度遵循 `config/settings.toml::[system].timezone`（默认 `Asia/Shanghai`），不需要额外设 `TZ` 环境变量。

挂载卷：

- `./data:/app/data` — SQLite DB + factors.parquet + 持仓 JSON
- `./config:/app/config` — settings.toml + watchlist + 备份 stock pool yaml
- `./models:/app/models` — 训练好的 LightGBM 模型

## 监控 daemon

启动 daemon 后：

```bash
uv run pangu status        # DB 统计 + 最近 24h 任务运行历史 + 最新持仓快照
# 异常的任务会自动推飞书告警，并在 task_runs 表留下完整 traceback
```

详见 [调度任务详解](scheduling.md)。
