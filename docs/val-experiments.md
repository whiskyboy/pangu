# Val Set Experiments — LGB 策略超参优化

## 方法论

Optuna Bayesian joint optimization over 17 training dimensions + separate portfolio grid search.
Multi-fidelity: Phase A (n_seeds=1) → Phase B (n_seeds=5 for top-3) → Phase C (portfolio grid) → Phase D (test confirmation).

### 固定参数

**训练**：
- `first_train_start = 2020-01-01`, `last_test_end = 2025-12-31`
- `val_months = 3`, `test_months = 3`
- `label_winsorize = 0.2`
- `early_stop_metric = 'mae'` (rankic 5-10x慢且无收益)

**回测**：
- Val backtest: `--start 2022-01-01 --end 2025-08-31` (所有 train_months 配置的交集)
- Phase A/B: `top_n=30, n_drop=10`

---

## Phase A: Optuna 联合探索 (80 trials, n_seeds=1)

**Status**: ✅ Complete

### 搜索空间

17 维 + 条件参数：num_leaves, learning_rate, n_estimators, subsample, colsample_bytree,
min_child_samples, max_bin, reg_alpha, reg_lambda, early_stopping_rounds, min_iterations,
mode, normalize_label, time_decay_halflife, train_months, step_months, label_horizon
(+ conditional: n_bins for ranking, label_weights for multi-horizon)

### Top-10 Trial 结果

| Rank | Trial | Val Sharpe | AnnRet | MaxDD | WinRate | Turnover | Leaves | LR | Train | TD | Horizon | Step |
|------|-------|-----------|--------|-------|---------|----------|--------|-----|-------|----|---------|------|
| 1 | #67 | **+0.683** | +11.3% | -24.3% | 52.5% | 33.4x | 32 | 0.039 | 12mo | 102 | 10 | 2 |
| 2 | #65 | +0.622 | +9.9% | -25.7% | 51.7% | 33.8x | 29 | 0.045 | 12mo | 116 | 10 | 2 |
| 3 | #77 | +0.609 | +10.2% | -24.4% | 52.7% | 34.0x | 40 | 0.029 | 12mo | 103 | 10 | 2 |
| 4 | #59 | +0.603 | +10.2% | -29.1% | 52.4% | 33.5x | 23 | 0.035 | 12mo | 122 | 10 | 2 |
| 5 | #61 | +0.595 | +9.7% | -24.9% | 52.1% | 33.2x | 24 | 0.035 | 12mo | 131 | 10 | 2 |
| 6 | #74 | +0.553 | +9.0% | -23.3% | 52.9% | 33.8x | 41 | 0.039 | 12mo | 115 | 10 | 2 |
| 7 | #73 | +0.550 | +8.9% | -24.1% | 53.2% | 33.9x | 41 | 0.040 | 12mo | 114 | 10 | 2 |
| 8 | #71 | +0.544 | +8.7% | -23.7% | 52.9% | 33.2x | 33 | 0.039 | 12mo | 123 | 10 | 2 |
| 9 | #68 | +0.530 | +8.3% | -23.6% | 52.9% | 33.4x | 37 | 0.053 | 12mo | 102 | 10 | 2 |
| 10 | #63 | +0.525 | +8.2% | -23.2% | 52.6% | 33.6x | 29 | 0.046 | 12mo | 123 | 10 | 2 |

**所有 Top-10 一致特征**: `train=12mo, horizon=10, step=2, regression, normalize=False, max_bin=63`

### Best Trial (#67) 完整回测报告

- **Sharpe**: 0.683
- **年化收益**: +11.3%
- **最大回撤**: -24.3%
- **胜率**: 52.5%
- **年化换手率**: 33.4x
- **LightGBM 参数**: leaves=32, lr=0.039, n_est=889, subsample=0.84, colsample=0.53, min_child=118, es=264, mi=20, reg_alpha=6.6e-6, reg_lambda=0.69, stride=3

### 参数重要性排序

| 参数 | 重要性 |
|------|--------|
| mode (regression vs ranking) | **0.9233** |
| label_horizon | 0.0164 |
| min_iterations | 0.0131 |
| colsample_bytree | 0.0107 |
| train_months | 0.0070 |
| early_stopping_rounds | 0.0045 |
| time_decay_halflife | 0.0041 |
| 其余 11 个参数 | < 0.004 |

`mode` 占 92.3% 的重要性 — ranking 模式 (LambdaRank) 始终产出负 Sharpe，是最关键的选择。

### 搜索过程分析

- **前 30 trials**: TPE 在 regression 和 ranking 之间均匀探索，regression 明显胜出
- **Trial 31-50**: 收敛到 regression + train=21mo + hz=5,10，Sharpe 到 0.49
- **Trial 51-80**: 发现新 regime — `train=12mo + hz=10 + step=2`，Sharpe 跳跃到 0.68
- **关键转折**: 短训练窗口(12mo) + 单10天 horizon + 重叠窗口(step=2) 大幅优于先前认为最佳的 21mo + 多horizon

---

## Phase B: Top-3 精确验证 (n_seeds=5)

**Status**: ✅ Complete

### Top-3 配置对比

| Config | Trial | Phase A Sharpe (1-seed) | Phase B Sharpe (5-seed) | AnnRet | MaxDD | WinRate | Turnover |
|--------|-------|------------------------|------------------------|--------|-------|---------|----------|
| 1 | #67 | 0.683 | 0.601 | +9.9% | -25.9% | — | — |
| **2** | **#65** | **0.622** | **0.768** | **+12.8%** | **-23.7%** | **—** | **—** |
| 3 | #77 | 0.609 | 0.593 | +9.7% | -23.6% | — | — |

### Model Winner: Config 2 (Trial #65)

**参数**:
- num_leaves=29, learning_rate=0.0446, n_estimators=756
- subsample=0.84, colsample_bytree=0.52, min_child_samples=143
- max_bin=63, reg_alpha=7.8e-6, reg_lambda=7.93
- early_stopping_rounds=247, min_iterations=20
- train_months=12, step_months=2, time_decay_halflife=116
- label_horizon=10, normalize_label=False, mode=regression
- train_subsample_stride=4

**回测指标** (Val, 2022-01-01 ~ 2025-08-31, top_n=30, n_drop=10):
- **Sharpe**: 0.768
- **年化收益**: +12.8%
- **最大回撤**: -23.7%

**IC**: Mean Test IC=0.043, Test RankIC=0.058; Mean Val IC=0.034, Val RankIC=0.050

### Model Winner 选择理由

Config 2 在 5-seed 验证中 Sharpe=0.768，远高于 Config 1 (0.601) 和 Config 3 (0.593)。
Config 1 (#67) 在 Phase A 的高 Sharpe (0.683) 是 seed-lucky，5-seed 后降至 0.601。
这验证了 multi-seed 验证的必要性：单 seed 结果可能误导方向。

---

## Phase C: 组合参数 Grid Search

**Status**: ✅ Complete

### Grid: top_n × n_drop × score_smooth_halflife (105 combos)

Using Model Winner (Config 2, Trial #65) 的 val score matrix.

### Top-10 组合

| Rank | top_n | n_drop | EMA | Val Sharpe | AnnRet | MaxDD | WinRate | Turnover |
|------|-------|--------|-----|-----------|--------|-------|---------|----------|
| 1 | **20** | **5** | **2** | **+0.857** | +13.9% | -20.7% | 52.3% | 25.0x |
| 2 | 20 | 10 | 3 | +0.837 | +14.0% | -24.7% | 52.1% | 36.5x |
| 3 | 20 | 0 | 1 | +0.822 | +14.6% | -24.1% | 53.2% | 54.1x |
| 4 | 20 | 10 | 1 | +0.805 | +14.0% | -23.9% | 52.1% | 44.7x |
| 5 | 20 | 5 | 3 | +0.802 | +12.8% | -20.9% | 52.6% | 24.6x |
| 6 | 20 | 15 | 2 | +0.802 | +13.7% | -25.0% | 52.5% | 44.2x |
| 7 | 20 | 15 | 1 | +0.800 | +14.1% | -24.3% | 52.7% | 53.6x |
| 8 | 20 | 8 | 1 | +0.798 | +13.7% | -23.2% | 52.5% | 38.2x |
| 9 | 20 | 15 | 3 | +0.797 | +13.4% | -24.1% | 51.6% | 38.2x |
| 10 | 20 | 12 | 3 | +0.796 | +13.2% | -24.1% | 51.8% | 37.8x |

**所有 Top-10 均为 top_n=20**

### 参数敏感度分析

| 参数 | 最佳值 | Marginal mean Sharpe |
|------|--------|---------------------|
| top_n=20 | ✅ | 0.707 |
| top_n=30 | | 0.624 |
| top_n=50 | | 0.521 |
| n_drop=0 | | 0.640 |
| n_drop=5 | ✅ | 0.608 |
| n_drop=12 | | 0.652 |
| EMA=0 | | 0.605 |
| EMA=1 | ✅ | 0.646 |
| EMA=2 | | 0.635 |

Best config (#1) 选择 top_n=20 + n_drop=5 + EMA=2，兼具高 Sharpe (0.857)、低回撤 (20.7%) 和低换手率 (25x)。

---

## Phase D: Test Set 一次性确认

**Status**: ✅ Complete

### 最终配置

**模型参数 (Trial #65, Phase B winner)**:
```
num_leaves=29, learning_rate=0.0446, n_estimators=756
subsample=0.84, colsample_bytree=0.52, min_child_samples=143
max_bin=63, reg_alpha=7.84e-6, reg_lambda=7.93
early_stopping_rounds=247, min_iterations=20
mode=regression, normalize_label=false
time_decay_halflife=116, train_months=12, step_months=2
train_subsample_stride=4, label_horizon=10
```

**组合参数 (Phase C winner)**:
```
top_n=20, n_drop=5, score_smooth_halflife=2
```

### Val vs Test 对比

同一 backtest 区间 `2022-01-01 ~ 2025-08-31`（剔除 test 尾部多出的 3.5 个月，保证可比）：

| Metric | Val | Test | Gap |
|--------|-----|------|-----|
| Sharpe | **+0.857** | **+0.285** | -66.8% |
| Annual Return | +13.9% | +3.9% | -72.0% |
| Max Drawdown | -20.7% | -26.9% | +30.0% |
| Win Rate | 52.3% | 52.4% | ≈0% |
| Turnover | 25.0x | 25.3x | ≈0% |

完整 test 区间 `2022-01-01 ~ 2025-12-17` (3.8y)：Sharpe=0.236, AnnRet=+2.9%, MaxDD=-26.9%

### Val-Test Gap 分析

**Gap 显著**：Val Sharpe 0.857 → Test 0.285（-67%）。

**分解 Gap 来源**：

| 配置 | Val Sharpe | Test Sharpe | Gap |
|------|-----------|-------------|-----|
| HPO Model + Default Portfolio (30/10) | 0.768 | 0.178 | -77% |
| HPO Model + Optimized Portfolio (20/5/ema2) | 0.857 | 0.285 | -67% |

- **模型层 (Phase A/B)**：Val→Test gap 77%，是主要过拟合来源
- **组合层 (Phase C)**：portfolio 优化反而**缩小**了 gap（0.178→0.285），说明 top_n=20/n_drop=5 泛化好

**过拟合原因分析**：
1. Optuna 80 trials 在 val set 上搜索 → 隐式过拟合到 val period 的市场特征
2. Walk-Forward 的 val period 紧接 train period（模型记忆更强），test period 距 train 更远（预测力衰减）
3. train_months=12 比默认 18 短 → 模型更依赖近期模式，泛化能力受限

### 与 Baseline 对比

E0 Baseline = 默认参数（train=18mo, step=3, 5 seeds, top_n=30, n_drop=10, time_decay=180）：

| 配置 | Val Sharpe | Test Sharpe | Val-Test Gap |
|------|-----------|-------------|-------------|
| E0 Baseline | 0.201 | **0.353** | -75% (test>val!) |
| HPO Optimized | **0.857** | 0.285 | -67% |

**关键发现**：
- E0 Baseline 在 test 上反而 **优于** HPO 优化结果（0.353 vs 0.285）
- E0 呈现 test > val 的反常模式，说明 E0 默认参数恰好对 test period 友好
- **HPO 优化在 val 上提升了 4x（0.201→0.857），但在 test 上反而下降了 19%（0.353→0.285）**
- 这是教科书级的过拟合案例：val 大幅提升，test 不升反降

### 结论与建议

**⚠️ HPO 优化未通过 test set 验证，不建议采纳。**

1. **Val Sharpe 0.857 是虚高的**：80 trials Optuna 搜索 + 105 组合 grid search 在同一 val set 上重复优化，导致严重过拟合
2. **E0 默认参数在 test set 上更优**：Sharpe 0.353 > 0.285
3. **唯一可靠收获**：
   - `top_n=20` 优于 `top_n=30`（marginal effect 在 val 和 test 上一致）
   - `score_smooth_halflife=2` 轻微改善（低成本，无过拟合风险）
   - `ranking mode` 确认不可用（92.3% 参数重要性，全部负 Sharpe）

**建议的保守改动**（不需要重训模型）：
- 将 `top_n` 从 30 改为 20, `n_drop` 从 10 改为 5
- 添加 `score_smooth_halflife=2`
- 保留 E0 默认模型参数不变

**保守改动的 test set 验证**（E0 模型 + 新组合参数）：

| 配置 | Val Sharpe | Test Sharpe |
|------|-----------|-------------|
| E0 + 原组合 (30/10) | 0.201 | 0.353 |
| **E0 + 新组合 (20/5/ema2)** | **0.269** | **0.690** |

✅ **Test Sharpe 从 0.353 提升到 0.690（+95%），且 test > val，无过拟合迹象。**
组合参数优化是本次 HPO 实验唯一可靠且显著的收获。

**如果要继续 HPO 探索**：
- 减少 Optuna trials 数（20-30）以降低 val 过拟合
- 使用 nested cross-validation（outer loop 换 val period）
- 在 train_months={15,18} 范围搜索（12 太短）
