"""Prompt templates for LLM pool-level rebalance debate.

Pure functions: no DB, no provider calls.  All data passed in as arguments.
"""

from __future__ import annotations

import math

import pandas as pd

from pangu.models import NewsItem

# ---------------------------------------------------------------------------
# Prompt truncation limits
# ---------------------------------------------------------------------------

# Per-candidate truncation for rebalance prompt (tighter to bound total length)
_REBAL_STOCK_NEWS = 5
_REBAL_ANNOUNCEMENTS = 3
_REBAL_TELEGRAPH = 10

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
}

# ---------------------------------------------------------------------------
# Shared section formatters
# ---------------------------------------------------------------------------


def _format_news_section(heading: str, items: list[NewsItem], limit: int) -> str:
    lines = [f"## {heading}"]
    if not items:
        lines.append("无\n")
        return "\n".join(lines)

    shown = items[:limit]
    for item in shown:
        ts = item.timestamp.strftime("%m-%d %H:%M")
        title = item.title
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


# ---------------------------------------------------------------------------
# Rebalance prompt — pool-level Bull/Bear/Judge debate
# ---------------------------------------------------------------------------

REBALANCE_OUTPUT_SCHEMA: dict[str, str] = {
    "sell_debate": '{"bull": "...", "bear": "..."}',
    "buy_debate": '{"bull": "...", "bear": "..."}',
    "sells": '[{"symbol": "...", "reason": "...", "evidence": "..."}, ...]',
    "buys": '[{"symbol": "...", "reason": "...", "evidence": "..."}, ...]',
}


REBALANCE_SYSTEM_PROMPT_TEMPLATE: str = """\
你是一位 A 股基金经理。今天是周首交易日，你需要做调仓决策。

## 重要背景

- 当前持仓 {top_n} 只股票；每次最多换 {n_drop} 只（n_drop={n_drop}）。
- 量化 ML 模型已圈定两个候选池：
  - **SELL 候选池**：当前持仓中 ML 评分最差的 {sell_pool_size} 只
  - **BUY 候选池**：非持仓中 ML 评分最好的 {buy_pool_size} 只
- 你的任务：在 ML 圈定的候选池内做事件级精选（两阶段 TopkDropout 的第二阶段）。

## 分析方法

依次从三个角色的视角对**整个候选池**进行分析：
1. **牛方 (Bull)**：寻找池内股票的利好/支撑论据
2. **熊方 (Bear)**：寻找池内股票的利空/风险论据
3. **裁判 (Judge)**：综合两方论辩，挑出最该卖/最该买的 ≤{n_drop} 只

## 决策原则

- **强证据优先**：监管/重大事件 > 业绩变化 > 技术形态 > 行业趋势
- **SELL 比 BUY 更严格**：卖错 = 错过反弹，机会成本高
  - 持仓股若有强烈利好预期（新订单、解禁完成、并购、业绩超预期）→ 不选（LLM 否决 ML SELL 信号）
  - 持仓股若仅是 ML 排名靠后但无明显利空 → 让兜底逻辑按 ML 排名补齐
- **BUY 重事件催化**：候选股若有重大隐藏风险（监管处罚、业绩雷、ST 化）→ 不选
- **找不到充分理由 → 不选**：兜底逻辑会用 ML 排名补齐缺口，无需为了凑满 {n_drop} 而强选
- **必填字段**：sells/buys 每项的 `symbol` 与 `reason`；`evidence` 建议引用具体新闻/公告/技术指标

## 输入数据

每只候选股都附带：
- ML 综合得分 (0-1, 归一化), 当前 ML 截面排名
- 持仓股额外附带：上次记录的 ML 排名、排名变化（持续走弱 vs 短期波动）
- 因子细项（RSI/MACD/PE/PB/ROE 等可用项）
- 个股近期新闻/公司公告
- 全市场快讯、隔夜全球市场环境

## 输出 JSON 格式（严格遵守）

```json
{{
  "sell_debate": {{
    "bull": "看多论据：为什么这些 ML 想卖的股票里有些可能不该卖（1-3 句）",
    "bear": "看空论据：为什么这些股票整体确实该卖（1-3 句）"
  }},
  "buy_debate": {{
    "bull": "看多论据：为什么候选股有买入价值（1-3 句）",
    "bear": "看空论据：为什么有些候选股需要谨慎（1-3 句）"
  }},
  "sells": [
    {{"symbol": "600000", "reason": "1-2 句具体理由", "evidence": "新闻/公告/指标引用"}}
  ],
  "buys": [
    {{"symbol": "601899", "reason": "1-2 句具体理由", "evidence": "新闻/公告/指标引用"}}
  ]
}}
```

- `sells` / `buys` 最多各 {n_drop} 只
- `symbol` 必须来自对应候选池；池外的 symbol 会被丢弃
- 所有中文字段使用中文
- 若实在选不出 → 数组为空，由系统按 ML 排名兜底
"""


def build_rebalance_system_prompt(
    *,
    top_n: int,
    n_drop: int,
    sell_pool_size: int,
    buy_pool_size: int,
) -> str:
    """Return the system prompt for ``judge_rebalance`` with pool sizes filled in."""
    return REBALANCE_SYSTEM_PROMPT_TEMPLATE.format(
        top_n=top_n,
        n_drop=n_drop,
        sell_pool_size=sell_pool_size,
        buy_pool_size=buy_pool_size,
    )


def build_rebalance_prompt(
    *,
    today: str,
    sell_candidates: list[dict],
    buy_candidates: list[dict],
    telegraph: list[NewsItem],
    global_market: pd.DataFrame,
    top_n: int,
    n_drop: int,
    universe_size: int = 0,
) -> str:
    """Build the user message for ``judge_rebalance``.

    Parameters
    ----------
    today : trading date string (YYYY-MM-DD).
    sell_candidates / buy_candidates : list of dicts. Required keys per item:
        ``symbol`` (str), ``name`` (str), ``ml_score`` (float 0-1),
        ``ml_rank`` (int).
      Optional keys:
        ``prev_ml_rank`` (int|None, sell side only),
        ``rank_delta`` (int|None, sell side only),
        ``factor_details`` (dict[str, float], filtered to known keys),
        ``stock_news`` (list[NewsItem]),
        ``announcements`` (list[NewsItem]).
    telegraph / global_market : market-level context (shared).
    top_n / n_drop : passed for prompt clarity (not used for logic here).
    universe_size : total stocks scored today (for rank display).
    """
    parts: list[str] = [
        f"# 调仓决策 ({today})\n",
        _format_pool_summary(top_n, n_drop, len(sell_candidates), len(buy_candidates)),
        _format_candidate_section(
            "🔴 SELL 候选池（持仓中 ML 评分最差）",
            sell_candidates,
            universe_size=universe_size,
            include_rank_history=True,
        ),
        _format_candidate_section(
            "🟢 BUY 候选池（非持仓中 ML 评分最好）",
            buy_candidates,
            universe_size=universe_size,
            include_rank_history=False,
        ),
        _format_news_section("📡 市场快讯", telegraph, _REBAL_TELEGRAPH),
        _format_global_market_section(global_market),
    ]
    return "\n".join(parts)


def _format_pool_summary(top_n: int, n_drop: int, n_sell: int, n_buy: int) -> str:
    return (
        f"## 任务概要\n"
        f"- 目标持仓: {top_n} 只\n"
        f"- 每次最多换: {n_drop} 只（n_drop={n_drop}）\n"
        f"- SELL 候选池规模: {n_sell}\n"
        f"- BUY  候选池规模: {n_buy}\n"
    )


def _format_candidate_section(
    heading: str,
    candidates: list[dict],
    *,
    universe_size: int,
    include_rank_history: bool,
) -> str:
    lines = [f"## {heading}"]
    if not candidates:
        lines.append("（空）\n")
        return "\n".join(lines)

    for cand in candidates:
        sym = cand.get("symbol", "")
        name = cand.get("name", sym)
        score = cand.get("ml_score", 0.0)
        rank = cand.get("ml_rank", 0)
        rank_str = f"{rank}/{universe_size}" if universe_size else f"{rank}"
        header = f"### {sym} {name}".rstrip()
        lines.append(header)
        lines.append(f"- ML 综合得分: {float(score):.4f}, 当前排名: {rank_str}")

        if include_rank_history:
            prev = cand.get("prev_ml_rank")
            delta = cand.get("rank_delta")
            if prev is None and delta is None:
                lines.append("- ML 排名历史: 首次进入持仓（无上次记录）")
            else:
                prev_s = str(prev) if prev is not None else "—"
                delta_s = _format_rank_delta(delta)
                lines.append(f"- 上次 ML 排名: {prev_s}（变化: {delta_s}）")

        # Factor details (technical indicators + fundamentals)
        details = cand.get("factor_details") or {}
        if details:
            lines.append("- 因子细项:")
            for key, val in details.items():
                label = FACTOR_LABELS.get(key, key)
                if isinstance(val, float) and math.isnan(val):
                    lines.append(f"  - {label}: 数据缺失")
                elif isinstance(val, (int, float)):
                    lines.append(f"  - {label}: {float(val):.4f}")
                else:
                    lines.append(f"  - {label}: {val}")

        # News / announcements
        s_news = cand.get("stock_news") or []
        anns = cand.get("announcements") or []
        if s_news:
            lines.append("- 个股新闻:")
            for item in s_news[:_REBAL_STOCK_NEWS]:
                ts = item.timestamp.strftime("%m-%d %H:%M")
                title = item.title
                snippet = ""
                if item.content and item.content != item.title:
                    snippet = item.content[:80]
                    if len(item.content) > 80:
                        snippet += "…"
                    snippet = f" — {snippet}"
                lines.append(f"  - [{ts}] {title}{snippet}")
            if len(s_news) > _REBAL_STOCK_NEWS:
                lines.append(f"  - (共 {len(s_news)} 条, 仅展示前 {_REBAL_STOCK_NEWS} 条)")
        else:
            lines.append("- 个股新闻: 无")

        if anns:
            lines.append("- 公司公告:")
            for item in anns[:_REBAL_ANNOUNCEMENTS]:
                ts = item.timestamp.strftime("%m-%d")
                lines.append(f"  - [{ts}] {item.title}")
            if len(anns) > _REBAL_ANNOUNCEMENTS:
                lines.append(f"  - (共 {len(anns)} 条, 仅展示前 {_REBAL_ANNOUNCEMENTS} 条)")
        else:
            lines.append("- 公司公告: 无")
        lines.append("")
    return "\n".join(lines)


def _format_rank_delta(delta: int | None) -> str:
    if delta is None:
        return "—"
    if delta > 0:
        return f"↓{delta}（变差）"
    if delta < 0:
        return f"↑{abs(delta)}（变好）"
    return "持平"
