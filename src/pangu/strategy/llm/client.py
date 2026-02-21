"""LLM client — LiteLLM wrapper with retry and rule-based degradation.

LLMClient: LiteLLM wrapper with retry + rule degradation.
LLMJudgeEngine: per-stock comprehensive judge (evidence package → BUY/SELL/HOLD).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

import pandas as pd

from pangu.models import Action, NewsItem, SignalStatus, TradeSignal
from pangu.strategy.llm.prompts import (
    TRADING_JUDGE_SYSTEM_PROMPT,
    build_stock_prompt,
)
from pangu.tz import now as _now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Client — LiteLLM wrapper with retry + rule degradation
# ---------------------------------------------------------------------------

# Default structured output schema expected from LLM (judge mode)
_DEFAULT_RESPONSE: dict = {
    "action": "HOLD",
    "confidence": 0.0,
    "bull_reason": "",
    "bear_reason": "",
    "judge_conclusion": "",
    "short_term_outlook": "",
    "mid_term_outlook": "",
}

_BULLISH_KEYWORDS: list[str] = [
    "利好", "增持", "回购", "业绩预增", "业绩大增", "超预期", "突破",
    "获批", "中标", "战略合作", "订单", "扩产", "涨停", "新高",
    "重大合同", "并购", "重组", "分红", "买入", "增长",
]

_BEARISH_KEYWORDS: list[str] = [
    "利空", "减持", "业绩预减", "业绩下滑", "亏损", "退市", "违规",
    "处罚", "诉讼", "跌停", "暴跌", "下调", "风险", "警示",
    "ST", "质押", "爆雷", "卖出", "下降", "萎缩",
]


class LLMClient:
    """LiteLLM wrapper with retry and rule-based degradation.

    Fallback order:
    1. Primary model (retry with exponential backoff)
    2. Rule-based keyword scoring (no API call)
    """

    _RETRY_BASE_DELAY = 1.0   # seconds
    _RETRY_ATTEMPTS = 2

    def __init__(
        self,
        *,
        model: str = "azure/gpt-4o-mini",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def call(self, system_prompt: str, user_prompt: str) -> dict:
        """Call LLM and return parsed JSON dict.

        Tries primary model with retry → rule-based degradation.
        """
        result = await self._try_model(self._model, system_prompt, user_prompt)
        if result is not None:
            return result

        logger.warning("LLM provider failed, using rule-based fallback")
        return self._rule_based_fallback(user_prompt)

    async def _try_model(
        self, model: str, system_prompt: str, user_prompt: str
    ) -> dict | None:
        """Try a single model with exponential backoff. Returns parsed dict or None."""
        import litellm

        for attempt in range(self._RETRY_ATTEMPTS):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    timeout=self._timeout,
                    response_format={"type": "json_object"},
                )
                self._call_count += 1
                text = response.choices[0].message.content or ""
                parsed = self._parse_json_response(text)
                if parsed is not None:
                    return parsed
                logger.warning(
                    "Model %s returned unparseable response (attempt %d)",
                    model, attempt + 1,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Model %s failed (attempt %d)",
                    model, attempt + 1, exc_info=True,
                )
            # Exponential backoff before retry (skip after last attempt)
            if attempt < self._RETRY_ATTEMPTS - 1:
                delay = self._RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
        return None

    def _parse_json_response(self, text: str) -> dict | None:
        """Extract JSON dict from LLM response text.

        Tries in order:
        1. Direct json.loads (pure JSON)
        2. Fenced ```json ... ``` block
        3. First {...} substring (JSON embedded in text)
        """
        text = text.strip()
        if not text:
            return None

        # 1. Direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # 2. Fenced JSON block
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1).strip())
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # 3. First valid JSON object in text (brace-counting for arbitrary nesting)
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
            start = text.find("{", start + 1)

        return None

    def _rule_based_fallback(self, user_prompt: str) -> dict:
        """Keyword-based scoring when all LLM providers fail."""
        bull_count = sum(1 for kw in _BULLISH_KEYWORDS if kw in user_prompt)
        bear_count = sum(1 for kw in _BEARISH_KEYWORDS if kw in user_prompt)

        if bull_count > bear_count:
            action = "BUY"
            confidence = min(0.5 + bull_count * 0.05, 0.8)
            bull_reason = f"关键词命中: {bull_count}个利好词"
            bear_reason = f"关键词命中: {bear_count}个利空词"
        elif bear_count > bull_count:
            action = "SELL"
            confidence = min(0.5 + bear_count * 0.05, 0.8)
            bull_reason = f"关键词命中: {bull_count}个利好词"
            bear_reason = f"关键词命中: {bear_count}个利空词"
        else:
            action = "HOLD"
            confidence = 0.3
            bull_reason = "无明显利好信号"
            bear_reason = "无明显利空信号"

        return {
            "action": action,
            "confidence": confidence,
            "bull_reason": bull_reason,
            "bear_reason": bear_reason,
            "judge_conclusion": f"规则降级 (bull={bull_count}, bear={bear_count})",
            "short_term_outlook": "规则降级, 无法判断",
            "mid_term_outlook": "规则降级, 无法判断",
        }

# ---------------------------------------------------------------------------
# LLMJudgeEngine — per-stock comprehensive judge (方案C)
# ---------------------------------------------------------------------------

class LLMJudgeEngine(Protocol):
    """Interface for per-stock LLM comprehensive judge."""

    async def judge_stock(
        self,
        symbol: str,
        name: str,
        factor_score: float,
        factor_rank: int,
        factor_details: dict[str, float],
        stock_news: list[NewsItem],
        announcements: list[NewsItem],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        price: float,
        *,
        factor_signal: str = "",
        universe_size: int = 0,
    ) -> TradeSignal:
        """Judge a single stock and return a TradeSignal."""
        ...

    async def judge_pool(
        self,
        candidates: list[dict[str, Any]],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        *,
        universe_size: int = 0,
    ) -> list[TradeSignal]:
        """Judge a pool of candidates and return signals."""
        ...

