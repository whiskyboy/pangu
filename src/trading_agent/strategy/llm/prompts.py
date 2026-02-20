"""Prompt templates for LLM comprehensive judge — PRD §4.3.2.

Pure functions: no DB, no provider calls.  All data passed in as arguments.
"""

from __future__ import annotations

import math

import pandas as pd

from trading_agent.models import NewsItem

# ---------------------------------------------------------------------------
# JSON output schema expected from LLM
# ---------------------------------------------------------------------------

LLM_OUTPUT_SCHEMA: dict[str, str] = {
    "action": "BUY | SELL | HOLD",
    "confidence": "0.0-1.0",
    "bull_reason": "看多理由 (1-3 句)",
    "bear_reason": "看空理由 (1-3 句)",
    "judge_conclusion": "裁判综合结论 (2-4 句)",
    "short_term_outlook": "短期展望 (1-2 周)",
    "mid_term_outlook": "中期展望 (1-3 月)",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TRADING_JUDGE_SYSTEM_PROMPT: str = """\
你是一位专业的 A 股投资分析师。你的任务是综合所有证据，对给定的股票给出交易建议。

## 重要背景

这些股票已经过多因子量化模型的筛选，进入候选池的股票在因子排名和得分上表现突出。\
因子策略的预判（BUY/EXIT/WATCHLIST）已在因子数据中标注。\
你的核心职责是：基于全部证据，判断因子策略的预判是否可靠，并给出最终操作建议。

## 分析方法

请依次从三个角色的视角进行分析：
1. **牛方 (Bull)**：寻找所有支持买入的理由——因子得分高、利好新闻、行业趋势向好等
2. **熊方 (Bear)**：寻找所有支持卖出/回避的理由——因子走弱、利空消息、估值过高等
3. **裁判 (Judge)**：综合牛方和熊方的论点，结合当前市场环境，给出最终判断

## 决策原则

- 当因子策略预判为 BUY 时，你应判断"在什么条件下应该买入"以及"有哪些风险需要注意"，\
而不是简单地否定因子信号。如果没有发现明确的利空证据推翻因子预判，应倾向 BUY。
- 当因子策略预判为 EXIT 时，你应判断"是否确实应该退出"以及"是否有理由继续持有"。
- HOLD 仅在多空证据完全均衡、无法做出明确判断时使用。不要把 HOLD 当作默认选项。
- confidence 应反映证据的充分程度，而非你对市场走势的确定性。\
例如：因子排名靠前 + 利好新闻 → confidence 0.75-0.85；因子信号明确但新闻中性 → confidence 0.65-0.75。

## 输入说明

你会收到以下信息：
- **因子数据**：综合得分 (0-1)、截面排名（X/N 名，N为全市场股票数）、各因子值、因子策略预判
- **个股新闻**：最近 24 小时相关新闻
- **公司公告**：最近的公司公告
- **市场快讯**：最近 24 小时财经快讯
- **全球市场**：隔夜美股/港股/商品行情

## 输出要求

请严格以 JSON 格式输出，不要包含其他内容：

```json
{
  "action": "BUY",
  "confidence": 0.75,
  "bull_reason": "看多理由",
  "bear_reason": "看空理由",
  "judge_conclusion": "裁判综合结论",
  "short_term_outlook": "短期 1-2 周展望",
  "mid_term_outlook": "中期 1-3 月展望"
}
```

- action 只能是 BUY / SELL / HOLD 之一
- confidence 范围 0.0-1.0，表示对 action 的信心
- 所有文本字段使用中文

## 示例

假设某股票因子得分 0.82 (排名第 2/300)，因子策略预判 BUY，RSI=55，MACD 金叉，\
PE_TTM=15 (行业偏低)，近期有"获大额订单"新闻，隔夜美股小幅上涨：

```json
{
  "action": "BUY",
  "confidence": 0.80,
  "bull_reason": "因子综合得分靠前(2/300)，MACD 金叉确认上行趋势，PE 估值低于行业均值，且获大额订单利好基本面",
  "bear_reason": "RSI 接近 60，短期有一定涨幅，需注意追高风险",
  "judge_conclusion": "技术面与基本面共振向好，事件催化明确，短期回调风险可控。建议买入，设置合理止损",
  "short_term_outlook": "订单利好尚未充分反映，预计 1-2 周内有上行空间",
  "mid_term_outlook": "低估值+订单增长支撑，中期趋势偏多，关注后续业绩兑现"
}
```
"""

# ---------------------------------------------------------------------------
# Prompt truncation limits
# ---------------------------------------------------------------------------

_MAX_STOCK_NEWS = 10
_MAX_ANNOUNCEMENTS = 5
_MAX_TELEGRAPH = 15

# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def build_stock_prompt(
    symbol: str,
    name: str,
    factor_score: float,
    factor_rank: int,
    factor_details: dict[str, float],
    stock_news: list[NewsItem],
    announcements: list[NewsItem],
    telegraph: list[NewsItem],
    global_market: pd.DataFrame,
    *,
    factor_signal: str = "",
    universe_size: int = 0,
) -> str:
    """Build per-stock "evidence package" user prompt.

    All data must be pre-fetched; this function only formats text.

    Parameters
    ----------
    factor_signal : factor strategy's pre-decision (e.g. "BUY", "EXIT")
    universe_size : total stocks in factor universe (for rank context)
    """
    sections: list[str] = [
        f"# {symbol} {name}\n",
        _format_factor_section(factor_score, factor_rank, factor_details,
                               universe_size=universe_size,
                               factor_signal=factor_signal),
        _format_news_section("📰 个股新闻", stock_news, _MAX_STOCK_NEWS),
        _format_news_section("📋 公司公告", announcements, _MAX_ANNOUNCEMENTS),
        _format_news_section("📡 市场快讯", telegraph, _MAX_TELEGRAPH),
        _format_global_market_section(global_market),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------

FACTOR_LABELS: dict[str, str] = {
    "rsi_14": "RSI(14)",
    "macd_hist": "MACD 柱",
    "bias_20": "BIAS(20)",
    "obv": "OBV",
    "atr_14": "ATR(14)",
    "volume_ratio": "量比",
    "pe_ttm": "PE(TTM)",
    "pb": "PB",
    "roe_ttm": "ROE(TTM)",
    "macro_adj": "宏观调整",
}


def _format_factor_section(
    score: float, rank: int, details: dict[str, float],
    *, universe_size: int = 0, factor_signal: str = "",
) -> str:
    rank_str = f"第 {rank} / {universe_size} 名" if universe_size else f"第 {rank} 名"
    lines = [
        "## 📊 因子数据",
        f"- 综合得分: {score:.4f}",
        f"- 截面排名: {rank_str}",
    ]
    if factor_signal:
        lines.append(f"- 因子策略预判: {factor_signal}")
    if details:
        lines.append("- 因子细项:")
        for key, val in details.items():
            label = FACTOR_LABELS.get(key, key)
            if isinstance(val, float) and math.isnan(val):
                lines.append(f"  - {label}: 数据缺失")
            else:
                lines.append(f"  - {label}: {val:.4f}")
    return "\n".join(lines) + "\n"


def _format_news_section(
    heading: str, items: list[NewsItem], limit: int
) -> str:
    lines = [f"## {heading}"]
    if not items:
        lines.append("无\n")
        return "\n".join(lines)

    shown = items[:limit]
    for item in shown:
        ts = item.timestamp.strftime("%m-%d %H:%M")
        title = item.title
        # Include content snippet if available (first 100 chars)
        if item.content and item.content != item.title:
            snippet = item.content[:100]
            if len(item.content) > 100:
                snippet += "…"
            lines.append(f"- [{ts}] {title} — {snippet}")
        else:
            lines.append(f"- [{ts}] {title}")

    if len(items) > limit:
        lines.append(f"  (共 {len(items)} 条, 仅展示前 {limit} 条)")
    lines.append("")
    return "\n".join(lines)


def _format_global_market_section(df: pd.DataFrame) -> str:
    lines = ["## 🌍 全球市场 (隔夜)"]
    if df is None or df.empty:
        lines.append("无\n")
        return "\n".join(lines)

    for _, row in df.iterrows():
        name = row.get("name", row.get("symbol", ""))
        close = row.get("close", float("nan"))
        change_pct = row.get("change_pct", float("nan"))
        if math.isnan(close) or math.isnan(change_pct):
            lines.append(f"- {name}: 数据缺失")
            continue
        sign = "+" if change_pct >= 0 else ""
        lines.append(f"- {name}: {close:.2f} ({sign}{change_pct:.2f}%)")
    lines.append("")
    return "\n".join(lines)
