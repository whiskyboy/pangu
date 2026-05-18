# 配置说明

[← 返回 README](../README.md)

PanGu 有两个配置入口：

- **环境变量** (`.env`)：LLM / 飞书凭据等敏感信息，由 `python-dotenv` 自动加载
- **`config/settings.toml`**：业务参数（股票池、策略阈值、调度时间、ML 超参等）；支持 `$ENV_VAR` 占位符

## 环境变量 (`.env`)

PanGu 通过 LiteLLM 支持多家 LLM provider，**选一个**填凭据即可（不是 fallback 关系）。同时需要修改 `config/settings.toml::[llm].provider` 与之对应（默认 `azure/$AZURE_DEPLOYMENT`）。完整 provider 列表与切换方法见 [LLM Provider 配置指南](litellm-setup.md)。

```bash
# === LLM Provider（任选一个填）===

# 方案 A：Azure OpenAI（settings.toml 默认）
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-key
AZURE_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT=your-deployment-name

# 方案 B：DeepSeek
# DEEPSEEK_API_KEY=your-deepseek-key

# 方案 C：Google Gemini
# GEMINI_API_KEY=your-gemini-key

# 方案 D：OpenAI
# OPENAI_API_KEY=your-openai-key

# === 飞书 Bot（必需；详见 docs/feishu-bot-setup.md）===
# 用户私聊 Bot 即自动完成绑定，无需配置 open_id
FEISHU_APP_ID=cli_xxxx
FEISHU_APP_SECRET=your-secret
```

## 业务参数 (`config/settings.toml`)

### 策略 / 组合关键参数

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

### 离线训练参数

离线 Walk-Forward 训练相关参数（`n_seeds`、`time_decay_halflife`、`first_train_start`、`last_test_end` 等）单独维护在 `[ml]` 段；完整方法论与默认值见 [`docs/ml-experiments.md`](ml-experiments.md)。

### 调度时间

所有 6 个任务的触发时间都在 `[scheduler]` 段，时区由 `[system].timezone` 统一控制。修改时间不需要改代码。详见 [调度任务详解](scheduling.md)。

### `watchlist.yaml`

调仓股票池由 `[stock_pool].indices`（指数成分股）和 `config/watchlist.yaml`（手动 watchlist）联合构成。生产环境通常只用前者；watchlist 用于强制覆盖少量个股。

## 配置加载机制

- `.env` 由 `python-dotenv` 在进程启动时加载到 `os.environ`
- `config/settings.toml` 用 `tomllib` 解析；任何形如 `"$VAR"` 的值会被替换为 `os.environ["VAR"]`，未定义则原样保留
- 配置是线程安全单例，测试中可通过 `reset_settings()` 清空缓存

## 推荐做法

- 不要把 `.env` 提交进 git（已在 `.gitignore` 中）
- 生产环境建议把 `.env` 路径加固（chmod 600）
- 用 `pangu status` 快速验证配置是否生效（DB 路径、调度时间、最近任务状态）
