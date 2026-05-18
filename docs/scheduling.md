# 调度任务详解

[← 返回 README](../README.md)

PanGu 用 APScheduler 在交易日历驱动下运行 6 个任务。所有时间均由 `config/settings.toml` 控制。

## 任务清单

| 任务 | 默认时间 | 默认频率 | 配置键（`[scheduler]`） | 说明 |
|------|---------|---------|------------------------|------|
| **T1** 参考数据同步 | 06:00 | 每月 1 号 | `reference_data_sync_time` / `reference_data_sync_day` | 交易日历 + 指数成分股快照 + cninfo 公司画像 |
| **T2** 财经快讯轮询 | 00:00–23:00 整点 | 每小时（含周末） | `news_poll_start_time` / `news_poll_end_time` / `news_poll_interval_minutes` | 财联社快讯 → 去重 + 过期清理 |
| **T3** 国内行情同步 | 18:00 | 每交易日盘后 | `domestic_kline_sync_time` | 日 K + 基本面 + 估值增量同步 |
| **T4** 国际交易同步 | 07:00 | 每交易日 | `international_data_sync_time` | 美股 / 港股 / 商品快照入库（隔夜参考） |
| **T5** 月度模型训练 | 02:00 | 每月 1 号 | `model_training_time` / `model_training_day` | 单窗口 LightGBM 训练（≈15–20min） |
| **T6** 信号生成 / 调仓 | 08:15 | 每交易日盘前 | `signal_generation_time` | 数据自检 + LLM-TopkDropout 调仓（节奏由 `[rebalance]` 控制） |

> 任务编号反映**逻辑依赖**（数据 → 模型 → 信号），不代表调度时间。标记"交易日"的任务在非交易日自动跳过。

## 调仓节奏

T6 每个交易日的 `signal_generation_time` 都会被 cron 唤起，但 **是否真正调仓** 由 `config/settings.toml::[rebalance]` 段决定：

```toml
[rebalance]
mode = "weekly"        # "weekly" 或 "monthly"
weekly_day = 1         # 1..5 (Mon..Fri)
monthly_day = 1        # 1..28
```

| 模式 | 语义 |
|------|------|
| `weekly:N` | 每周第 N 天（1=Mon..5=Fri）调仓 |
| `monthly:N` | 每月第 N 天（1..28，避免 2 月边界）调仓 |

**非交易日处理**：如果排定的那一天不是交易日，系统**顺延**到下一个交易日触发；与多数 ETF 定投策略一致。算法描述：找到当前日所属周/月最近的目标日 T（T ≤ 今天），从 T 起向后扫描首个交易日 T*，今天是调仓日 iff 今天 == T*。

**不去重 — 设计上接受跨周/跨月双触发**：

- `weekly:5` 周五是节假日 → 顺延到下周一；下周五正常交易 → **该 ISO 周内 2 次调仓**。
- `monthly:28` 遇到 2 月 28 号是周末并整周非交易 → 顺延到下月 1 号；下月 28 号正常交易 → **该月内 2 次调仓**。

这是有意保留的简单语义：不引入"上次调仓距今 ≥ X 日"的去重逻辑，便于回测 / 生产一致和算法可预测性。如果某次双触发产生的换手让你不满意，请在该次调仓后手动调整持仓或选择不会导致跨边界的目标日（如 `monthly:15`）。

**默认值**：`weekly:1` ≈ 历史 ISO-周首交易日行为，老部署升级零变化。

**回测端独立配置**：

```bash
# 回测使用与生产相同的节奏（默认）
pangu backtest --rebalance weekly:1 ...

# 回测尝试更激进的每周五调仓
pangu backtest --rebalance weekly:5 ...

# 回测尝试每月调仓
pangu backtest --rebalance monthly:15 ...
```

注意：`pangu replay` 也接受 `--rebalance`，**必须** 与生产历史快照所用的节奏一致，否则 replay 触发日与快照日错位、PnL 不准确。

## 时区

所有调度时间使用 `config/settings.toml::[system].timezone` 统一控制，默认 `Asia/Shanghai`。APScheduler 会按 settings.toml 时区触发，**容器或海外 VPS 处于其他系统时区不需要额外设 `TZ`**。

## 修改时间

所有时间字段格式 `"HH:MM"`，day 字段为月份 1-28 的整数。示例：把调仓信号推送到 09:00、国内行情同步推迟 4 小时：

```toml
# config/settings.toml
[scheduler]
signal_generation_time = "09:00"
domestic_kline_sync_time = "22:00"
```

修改后重启 daemon (`pangu run start` 或 `docker compose restart worker`) 即生效。

## 任务监控

所有任务统一由 `@scheduled_task` 装饰（`src/pangu/tasks/_base.py`），自动捕获异常 → 写 `task_runs` 表 → 推送飞书告警。

```bash
pangu status                         # DB 统计 + 最近 24h 任务运行历史 + 最新持仓快照
pangu run signals                    # 手动触发 T6（若当天 T4 快照未生成会自动补跑）
pangu run signals --initial-capital 200000   # 临时改资金规模（不写回 settings.toml）
```

## misfire grace time

T1 设置了 `misfire_grace_time`，daemon 在调度时间窗口内启动会立刻补跑，避免错过月度刷新。其他任务依赖每日重复，自然 self-heal，无需 grace time。

## 手动触发

```bash
pangu run init       # 一键冷启动 bootstrap（详见 deployment.md）
pangu run signals    # 手动触发 T6
```

T2-T5 没有专门的手动入口（设计上无需手动）：
- T2 新闻可直接通过 `pangu` CLI 子命令或 backfill 触发
- T3/T4 增量同步会被 T6 自检触发
- T5 训练用 `pangu train`（生产单窗口）或 `pangu train walkforward`（离线多窗口）
