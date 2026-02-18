"""LLMEventEngine Protocol + LLMClient + FakeLLMEventEngine — PRD §4.3.2 / §6."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from trading_agent.models import Action, NewsItem, SignalStatus, TradeSignal

logger = logging.getLogger(__name__)


class LLMEventEngine(Protocol):
    """Interface for LLM-powered event-driven signal generation."""

    async def analyze_news(
        self, news: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        """Analyze news items and produce event-driven signals."""
        ...

    async def analyze_announcement(
        self, announcements: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        """Analyze company announcements and produce event-driven signals."""
        ...


# ---------------------------------------------------------------------------
# LLM Client — LiteLLM wrapper with retry + fallback + rule degradation
# ---------------------------------------------------------------------------

# Default structured output schema expected from LLM
_DEFAULT_RESPONSE: dict = {
    "direction": "neutral",
    "impact_score": 1,
    "bull_reason": "",
    "bear_reason": "",
    "judge_conclusion": "",
    "affected_symbols": [],
    "affected_sectors": [],
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
    """LiteLLM wrapper with retry, fallback chain, and rule-based degradation.

    Fallback order:
    1. Primary model (retry once on failure)
    2. Each fallback model (sequential, retry once each)
    3. Rule-based keyword scoring (no API call)
    """

    def __init__(
        self,
        *,
        model: str = "azure/gpt-4o-mini",
        fallback_models: list[str] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._fallback_models = fallback_models or []
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def call(self, system_prompt: str, user_prompt: str) -> dict:
        """Call LLM and return parsed JSON dict.

        Tries primary → fallbacks → rule-based degradation.
        """
        models = [self._model, *self._fallback_models]

        for model in models:
            result = await self._try_model(model, system_prompt, user_prompt)
            if result is not None:
                return result

        # All providers failed — rule-based fallback
        logger.warning("All LLM providers failed, using rule-based fallback")
        return self._rule_based_fallback(user_prompt)

    async def _try_model(
        self, model: str, system_prompt: str, user_prompt: str
    ) -> dict | None:
        """Try a single model with one retry. Returns parsed dict or None."""
        import litellm

        for attempt in range(2):  # 1 initial + 1 retry
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
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
            direction = "bullish"
            score = min(5 + bull_count, 10)
            bull_reason = f"关键词命中: {bull_count}个利好词"
            bear_reason = f"关键词命中: {bear_count}个利空词"
        elif bear_count > bull_count:
            direction = "bearish"
            score = min(5 + bear_count, 10)
            bull_reason = f"关键词命中: {bull_count}个利好词"
            bear_reason = f"关键词命中: {bear_count}个利空词"
        else:
            direction = "neutral"
            score = 3
            bull_reason = "无明显利好信号"
            bear_reason = "无明显利空信号"

        return {
            "direction": direction,
            "impact_score": score,
            "bull_reason": bull_reason,
            "bear_reason": bear_reason,
            "judge_conclusion": f"规则降级: {direction} (bull={bull_count}, bear={bear_count})",
            "affected_symbols": [],
            "affected_sectors": [],
        }


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------


class FakeLLMEventEngine:
    """Returns deterministic event signals for news containing known symbols."""

    async def analyze_news(
        self, news: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        from trading_agent.tz import now as _now
        now = _now()
        signals: list[TradeSignal] = []
        for item in news:
            for symbol in item.symbols:
                if symbol not in watchlist:
                    continue
                signals.append(TradeSignal(
                    timestamp=now,
                    symbol=symbol,
                    name=symbol,
                    action=Action.BUY,
                    signal_status=SignalStatus.NEW_ENTRY,
                    days_in_top_n=0,
                    price=100.0,
                    confidence=0.8,
                    source="llm_event",
                    reason=f"fake LLM: bullish on '{item.title}'",
                    metadata={"news_title": item.title},
                ))
        return signals

    async def analyze_announcement(
        self, announcements: list[NewsItem], watchlist: list[str]
    ) -> list[TradeSignal]:
        return await self.analyze_news(announcements, watchlist)
