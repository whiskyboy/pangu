# Copilot Agent & Skill 使用指南

本文档介绍 PanGu 项目中配置的 Copilot Agent 和 Skill，以及它们的使用场景和最佳实践。

## 架构总览

```
Agents（后台自主执行，输出报告）
├── data-quality        数据完整性审计
├── backtest-analyst    回测结果分析
└── backfill-manager    数据回填执行与监控

Skills（主对话中交互，需用户决策）
├── factor-research     因子分析与开发
├── train-and-backtest  全 pipeline（因子→训练→回测）
└── model-tuner         超参优化
```

**Agent vs Skill 选型原则：**

| 特征 | Agent | Skill |
|------|-------|-------|
| 执行模式 | 后台自主完成 | 在主对话中交互 |
| 上下文需求 | 不需要对话历史 | 可利用之前的对话上下文 |
| 用户参与 | 产出报告即可 | 需要中途做决策 |
| 典型场景 | 审计、诊断、code review | 多步 pipeline、引导式工作流 |

## Agent

### data-quality

**用途：** DB + 因子数据完整性审计、覆盖率检查、数据异常检测。

**触发方式：**
- 自然语言：`"检查数据完整性"` / `"run data quality audit"` / `"DB 健康检查"`
- 手动选择：`/agent` → data-quality
- 命令行：`copilot --agent data-quality --prompt "run full audit"`

**检查项：**
- daily_bars 行数、日期范围、NULL 率
- fundamentals 各列覆盖率（daily vs quarterly）
- adj_factor 异常检测（常量 1.0 = 静默刷新失败）
- factors.parquet NaN 率（尤其关注 fundamental 因子）
- score_matrix_test.parquet / score_matrix_val.parquet 时效性
- 模型 dead features 检测

**使用时机：**
- 每次 backfill 或 DB migration 后
- 重新训练前，确认数据无异常
- 定期健康检查

### backfill-manager

**用途：** 规划、执行、监控数据回填操作，诊断失败，验证完成度。

**触发方式：**
- 自然语言：`"backfill bars"` / `"回填数据"` / `"监控 backfill 进度"` / `"re-fetch data"`
- 手动选择：`/agent` → backfill-manager
- 命令行：`copilot --agent backfill-manager --prompt "backfill bars for full pool"`

**职责：**
- 规划回填顺序（constituents → bars → fundamentals → index）
- 使用 `screen` 启动长时间回填任务
- 监控进度（tail 日志、检查 ok/fail 计数）
- 诊断失败（BaoStock TCP 断连、登录过期、特定股票失败）
- 回填后验证完成度（行数、覆盖率、adj_factor 异常检测）

**使用时机：**
- 初始化数据库
- 需要补充历史数据
- 发现数据缺失或异常需要重新抓取

### backtest-analyst

**用途：** 分析回测结果、诊断回撤、比较不同策略/参数的回测。

**触发方式：**
- 自然语言：`"分析回测结果"` / `"为什么回撤这么大"` / `"比较 LGB 和 baseline"`
- 手动选择：`/agent` → backtest-analyst
- 命令行：`copilot --agent backtest-analyst --prompt "analyze latest backtest"`

**分析内容：**
- Sharpe ratio / max drawdown / 年化收益 / 胜率 / 换手率
- 回撤归因（时段、行业集中度）
- Score quality 诊断（区分度、稳定性）
- 多策略对比（LGB vs baseline）

**使用时机：**
- 训练完成后获取完整分析报告
- 调参后对比两次回测，量化改进

## Skill

### factor-research

**用途：** 因子分析（IC、冗余检测）、新因子开发、lookahead 审计。

**触发方式：**
- 自然语言：`"分析因子 IC"` / `"factor research"` / `"检测冗余因子"` / `"设计新因子"`
- 手动选择：`/skills` → factor-research

**工作流：**
1. **因子覆盖检查** — NaN 率、数据源追溯
2. **IC 分析** — 按时段计算每个因子的 IC / Rank IC
3. **Feature importance** — 从训练好的模型提取特征重要性
4. **冗余检测** — Spearman 相关性 > 0.9 的因子对
5. **新因子开发** — 设计 → 实现 → lookahead 审计 → 测试

**使用时机：**
- 添加新因子前，先分析现有因子
- 训练后发现 IC 下降，需要诊断因子质量

### train-and-backtest

**用途：** 完整 ML pipeline — compute-factors → train → evaluate → backtest。

**触发方式：**
- 自然语言：`"train and backtest"` / `"重新训练"` / `"跑完整 pipeline"` / `"retrain model"`
- 手动选择：`/skills` → train-and-backtest

**Pipeline 步骤：**
1. **Compute Factors** — Alpha158 (191 因子)
2. **Walk-Forward Training** — LightGBM 滚动训练
3. **Model & Score Diagnostics** — 诊断 dead features、score 质量
4. **Backtest** — 回测 LGB 策略
5. **Baseline Comparison** — 对比基线策略
6. **Summary** — 输出对比表

**使用时机：**
- 数据/因子/模型有变动后，端到端重跑
- 首次部署或大版本升级后

### model-tuner

**用途：** 超参优化、训练问题诊断、模型改进实验。

**触发方式：**
- 自然语言：`"调优模型"` / `"fix early stopping"` / `"tune hyperparameters"` / `"try LambdaRank"`
- 手动选择：`/skills` → model-tuner

**支持的优化场景：**
- **A. 诊断** — 发现 underfitting/overfitting 窗口、dead features
- **B. 早停修复** — MIN_ITERATIONS guard 防止过早停止
- **C. 超参调优** — num_leaves / learning_rate / regularization
- **D. LambdaRank** — 从回归切换到排序优化
- **E. 多周期标签** — 3d/5d/10d 混合标签减少噪声

**使用时机：**
- 回测结果不理想，需要系统性调优
- 发现特定窗口 IC 异常低

## 典型工作流

```
1. @backfill-manager → 回填数据（constituents → bars → fundamentals → index）
   ↓
2. @data-quality → 验证数据完整性
   ↓
3. @factor-research → 分析 / 添加因子
   ↓
4. @train-and-backtest → 训练 + 回测
   ↓
5. @backtest-analyst → 分析回测结果
   ↓
6. @model-tuner → 根据分析调优
   ↓
7. 回到 4，迭代直到满意
```

## 配置说明

### Agent 配置文件

Agent 配置位于 `.github/agents/`，使用 `.agent.md` 扩展名：

```yaml
---
name: agent-name
description: Agent description
model: claude-opus-4.6
---
Agent prompt and instructions...
```

**注意事项：**
- 不设置 `tools` 字段 = 默认拥有全部工具权限（推荐）
- 通过 prompt 中的 "You must NOT" 来约束行为
- 修改配置后**需要重启 CLI** 才能生效

### Skill 配置文件

Skill 配置位于 `skills/<skill-name>/SKILL.md`：

```yaml
---
name: skill-name
description: >
  Skill description with trigger words...
---
Skill instructions and workflow steps...
```

### 可用模型

Agent/Skill 的 `model` 字段支持以下模型（按推荐顺序）：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| `claude-opus-4.6` | 最强推理能力 | 复杂分析、审计（默认） |
| `claude-sonnet-4` | 平衡性能与速度 | 快速任务 |
| `claude-haiku-4.5` | 最快速度 | 简单查询 |
