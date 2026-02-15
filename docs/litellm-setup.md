# LLM Provider 配置指南

TradingAgent 通过 [LiteLLM](https://docs.litellm.ai/) 统一接口调用各种 LLM 后端。你只需更改 `config/settings.toml` 中的 `[llm].provider` 和相应的环境变量，即可在不同 Provider 之间无缝切换，**无需修改任何业务代码**。

---

## 工作原理

```
TradingAgent 业务代码
        │
        ▼
  litellm.completion(model="azure/gpt-4o-mini", messages=[...])
        │
        ▼
   LiteLLM SDK (自动路由)
        │
   ┌────┼────┬──────────┬───────────┐
   ▼    ▼    ▼          ▼           ▼
 Azure  DeepSeek  Gemini API  Ollama(本地)  ...
```

LiteLLM 通过 model 字符串的 **前缀** 自动识别 Provider：

| 前缀 | Provider | 示例 |
|------|----------|------|
| `azure/` | Azure OpenAI | `azure/gpt-4o-mini` |
| `deepseek/` | DeepSeek | `deepseek/deepseek-chat` |
| `gemini/` | Google Gemini (API Key) | `gemini/gemini-2.5-flash` |
| `openai/` | OpenAI 官方 | `openai/gpt-4o` |
| `anthropic/` | Anthropic Claude | `anthropic/claude-sonnet-4-20250514` |
| `github_copilot/` | GitHub Copilot | `github_copilot/gpt-4o` |
| `ollama/` | Ollama (本地部署) | `ollama/qwen2.5` |

---

## Provider 配置详解

### 1. Azure OpenAI（默认推荐）

**推荐理由**：企业级稳定性、国内网络可直连、合规、gpt-4o-mini 性价比高。

**第 1 步：创建 Azure OpenAI 资源**

1. 登录 [Azure Portal](https://portal.azure.com/)
2. 搜索并创建 **Azure OpenAI** 资源
3. 进入资源 → **Keys and Endpoint**，获取：
   - **Endpoint**（如 `https://your-resource.openai.azure.com/`）
   - **API Key**

**第 2 步：部署模型**

1. 打开 [Azure AI Foundry](https://ai.azure.com/)（原 Azure OpenAI Studio）
2. 进入 **Deployments** → **Create deployment**
3. 选择模型（推荐 `gpt-4o-mini`），设置部署名（如 `gpt-4o-mini`）

> ⚠️ LiteLLM 的 model 字符串使用的是**部署名**，不是模型名。例如部署名为 `gpt-4o-mini`，则 model 为 `azure/gpt-4o-mini`。

**第 3 步：配置环境变量**

```bash
# .env
AZURE_API_KEY=your-azure-api-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
```

**第 4 步：配置 settings.toml**

```toml
[llm]
provider = "azure/gpt-4o-mini"
azure_api_base = "$AZURE_API_BASE"
azure_api_version = "2024-08-01-preview"
```

---

### 2. DeepSeek

**推荐理由**：价格极低（约为 GPT-4o-mini 的 1/10），中文能力强，适合降低日常运营成本。

**第 1 步：获取 API Key**

1. 注册 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 进入 **API Keys** 页面，创建新的 API Key

**第 2 步：配置环境变量**

```bash
# .env
DEEPSEEK_API_KEY=your-deepseek-api-key
```

**第 3 步：配置 settings.toml**

```toml
[llm]
provider = "deepseek/deepseek-chat"
```

> 不需要配置 `api_base`，LiteLLM 会自动使用 DeepSeek 官方 API 地址。

**可用模型：**

| LiteLLM Model 字符串 | 说明 |
|----------------------|------|
| `deepseek/deepseek-chat` | 通用对话模型（推荐） |
| `deepseek/deepseek-reasoner` | 深度推理模型（Thinking 模式，更慢但更强） |

---

### 3. Google Gemini（API Key 模式）

**推荐理由**：免费额度较高，适合作为 fallback。

**第 1 步：获取 API Key**

1. 打开 [Google AI Studio](https://aistudio.google.com/)
2. 点击 **Get API Key** → **Create API key**
3. 选择或创建一个 Google Cloud 项目

**第 2 步：配置环境变量**

```bash
# .env
GEMINI_API_KEY=your-gemini-api-key
```

**第 3 步：配置 settings.toml**

```toml
[llm]
provider = "gemini/gemini-2.5-flash"
```

> 使用 `gemini/` 前缀表示通过 API Key 直连 Google，无需 GCP 项目配置。如果使用 Vertex AI（通过 GCP 服务账号认证），前缀应为 `vertex_ai/`。

**可用模型：**

| LiteLLM Model 字符串 | 说明 |
|----------------------|------|
| `gemini/gemini-2.5-flash` | 快速推理模型（推荐，性价比高） |
| `gemini/gemini-2.5-pro` | 高性能模型 |

---

### 4. OpenAI 官方

**第 1 步：获取 API Key**

1. 登录 [OpenAI Platform](https://platform.openai.com/)
2. 进入 **API Keys** 页面创建 Key

**第 2 步：配置环境变量**

```bash
# .env
OPENAI_API_KEY=your-openai-api-key
```

**第 3 步：配置 settings.toml**

```toml
[llm]
provider = "openai/gpt-4o-mini"
```

**可用模型：**

| LiteLLM Model 字符串 | 说明 |
|----------------------|------|
| `openai/gpt-4o-mini` | 快速经济模型（推荐） |
| `openai/gpt-4o` | 高性能模型 |

---

### 5. Anthropic Claude

**第 1 步：获取 API Key**

1. 登录 [Anthropic Console](https://console.anthropic.com/)
2. 进入 **API Keys** 页面创建 Key

**第 2 步：配置环境变量**

```bash
# .env
ANTHROPIC_API_KEY=your-anthropic-api-key
```

**第 3 步：配置 settings.toml**

```toml
[llm]
provider = "anthropic/claude-sonnet-4-20250514"
```

**可用模型：**

| LiteLLM Model 字符串 | 说明 |
|----------------------|------|
| `anthropic/claude-sonnet-4-20250514` | 平衡型模型（推荐） |
| `anthropic/claude-haiku-3-5-20241022` | 快速经济模型 |

---

### 6. GitHub Copilot

**推荐理由**：如果你已有 GitHub Copilot 订阅（Individual / Business / Enterprise），可以直接复用订阅额度调用多种模型（GPT-4o、Claude Sonnet 等），无需额外注册其他 Provider 或管理 API Key。

**前提条件**：需要有效的 [GitHub Copilot 订阅](https://github.com/features/copilot)。

**第 1 步：首次认证（OAuth Device Flow）**

GitHub Copilot 使用 OAuth 设备授权流程，**无需手动配置 API Key**。首次调用时 LiteLLM 会自动引导你完成认证：

1. 启动系统后，LiteLLM 会在终端输出一个验证 URL 和设备码，例如：
   ```
   Please visit: https://github.com/login/device
   Enter code: ABCD-1234
   ```
2. 在浏览器中打开该 URL，登录 GitHub 账号，输入设备码
3. 授权完成后，凭证会自动缓存到本地，后续调用无需重复认证

> 💡 认证凭证存储在本地，重启系统后通常无需重新认证。

**第 2 步：配置 settings.toml**

```toml
[llm]
provider = "github_copilot/gpt-4o"
```

> 不需要在 `.env` 中配置任何 API Key 或环境变量。

**可用模型：**

GitHub Copilot 通过统一入口提供多家厂商的模型，具体可用模型取决于你的订阅类型：

| LiteLLM Model 字符串 | 原始模型 | 说明 |
|----------------------|---------|------|
| `github_copilot/gpt-4o` | OpenAI GPT-4o | 高性能通用模型 |
| `github_copilot/gpt-4o-mini` | OpenAI GPT-4o-mini | 快速经济模型（推荐） |
| `github_copilot/gpt-4.1` | OpenAI GPT-4.1 | 最新 OpenAI 模型 |
| `github_copilot/claude-sonnet-4.5` | Anthropic Claude Sonnet 4.5 | 平衡型模型 |

> 可用模型列表随 GitHub Copilot 更新，最新列表参见 [GitHub Copilot 支持的模型](https://docs.github.com/en/copilot/reference/ai-models/supported-models)。

**注意事项：**

- ⚠️ GitHub Copilot 的 API 调用有速率限制，高频调用场景（如每 5 分钟全量扫描）可能触发限流
- ⚠️ 首次认证需要交互式终端（浏览器 + 手动输入设备码），不适合纯无头服务器首次部署。建议先在本地完成认证，再部署到服务器
- 适合作为已有 Copilot 订阅用户的便捷选项，生产环境建议搭配其他 Provider 作为 fallback

---

### 7. Ollama（本地部署）

**推荐理由**：完全免费、数据不出本地、无网络依赖。适合对隐私敏感或网络不稳定的场景。

**第 1 步：安装 Ollama**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**第 2 步：拉取模型**

```bash
ollama pull qwen2.5:7b    # 推荐：阿里通义千问，中文能力强
# 或
ollama pull llama3.1:8b   # Meta Llama 3.1
```

**第 3 步：配置环境变量**

```bash
# .env（仅当 Ollama 不在默认地址时需要配置）
OLLAMA_API_BASE=http://localhost:11434
```

**第 4 步：配置 settings.toml**

```toml
[llm]
provider = "ollama/qwen2.5:7b"
```

> ⚠️ 本地小模型（7B-8B）的分析质量可能不如 GPT-4o-mini 或 DeepSeek，建议仅在 API 不可用时作为应急 fallback。

---

## Fallback 链配置

TradingAgent 支持多 Provider fallback。当主 Provider 调用失败时，自动尝试下一个：

```toml
# config/settings.toml
[llm]
provider = "azure/gpt-4o-mini"                                       # 主 Provider
fallback_providers = ["deepseek/deepseek-chat", "gemini/gemini-2.5-flash"]  # 降级链
```

降级顺序：`azure/gpt-4o-mini` → `deepseek/deepseek-chat` → `gemini/gemini-2.5-flash` → 规则打分降级（关键词匹配）。

要使用 fallback，需要在 `.env` 中配置**所有** fallback Provider 的 API Key：

```bash
# .env
AZURE_API_KEY=your-azure-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
DEEPSEEK_API_KEY=your-deepseek-key
GEMINI_API_KEY=your-gemini-key
```

---

## 完整 .env 示例

```bash
# === LLM Provider API Keys ===
# 至少配置一个 Provider 的 Key，推荐配置多个以启用 fallback

# Azure OpenAI（默认主 Provider）
AZURE_API_KEY=your-azure-api-key
AZURE_API_BASE=https://your-resource.openai.azure.com/

# DeepSeek（推荐 fallback）
DEEPSEEK_API_KEY=your-deepseek-api-key

# Google Gemini（推荐 fallback）
GEMINI_API_KEY=your-gemini-api-key

# OpenAI 官方（可选）
# OPENAI_API_KEY=your-openai-api-key

# Anthropic Claude（可选）
# ANTHROPIC_API_KEY=your-anthropic-api-key

# GitHub Copilot（无需 API Key，通过 OAuth 设备授权认证）
```

---

## 完整 settings.toml [llm] 配置

```toml
[llm]
provider = "azure/gpt-4o-mini"        # LiteLLM model string
fallback_providers = ["deepseek/deepseek-chat", "gemini/gemini-2.5-flash"]
azure_api_base = "$AZURE_API_BASE"
azure_api_version = "2024-08-01-preview"
temperature = 0.1                     # 低温度保证输出稳定性
max_tokens = 800                      # 需容纳牛/熊/裁判三角色输出
news_impact_threshold = 7             # 影响程度 ≥ 7 才生成信号
max_api_calls_per_hour = 60           # 成本控制
```

---

## 各 Provider 对比

| Provider | 成本 (per 1M tokens) | 中文能力 | 稳定性 | 国内可用性 | 适合场景 |
|----------|---------------------|---------|--------|-----------|---------|
| Azure OpenAI (gpt-4o-mini) | ~$0.15 in / $0.60 out | ★★★★ | ★★★★★ | ★★★★★ | 默认主力 |
| DeepSeek (deepseek-chat) | ~$0.014 in / $0.028 out | ★★★★★ | ★★★★ | ★★★★★ | 低成本替代 |
| Gemini (2.5-flash) | 免费额度较高 | ★★★★ | ★★★★ | ★★★ (需代理) | 免费额度 / fallback |
| GitHub Copilot | 订阅包含 | ★★★★ | ★★★★ | ★★★ (需代理) | 已有订阅用户 |
| OpenAI (gpt-4o-mini) | ~$0.15 in / $0.60 out | ★★★★ | ★★★★★ | ★★ (需代理) | 海外部署 |
| Anthropic (claude-sonnet) | ~$3 in / $15 out | ★★★★ | ★★★★★ | ★★ (需代理) | 高质量分析 |
| Ollama (本地 7B) | 免费 | ★★★ | ★★★★★ | ★★★★★ | 离线应急 |

---

## 常见问题

**Q: 切换 Provider 需要改代码吗？**

不需要。只修改 `settings.toml` 的 `provider` 字段和 `.env` 中对应的 API Key 即可。LiteLLM 统一了所有 Provider 的调用接口。

**Q: 可以同时使用多个 Provider 吗？**

主 Provider 只能配置一个，但可以通过 `fallback_providers` 配置降级链。主 Provider 失败时自动切换到下一个。

**Q: DeepSeek 为什么这么便宜？**

DeepSeek 是国产大模型，定价策略激进。质量与 GPT-4o-mini 接近，中文场景甚至更优。适合用作默认 Provider 以降低成本。

**Q: 如何验证 LLM 配置是否正确？**

```bash
# 启动系统后手动触发一次新闻采集（包含 LLM 分析）
trading-agent run news --domestic
```

如果 LLM 调用成功，日志中会显示新闻的情感分析结果。

**Q: 国内网络无法访问某些 Provider？**

Azure OpenAI 和 DeepSeek 国内可直连。Gemini、OpenAI、Anthropic 需要代理。可在 `.env` 中设置：

```bash
HTTPS_PROXY=http://your-proxy:port
```
