"""LLM engines — PRD §4.3.2 / §6.

LLMClient: LiteLLM wrapper with retry + fallback + rule degradation.
LLMJudgeEngine: per-stock comprehensive judge (evidence package → BUY/SELL/HOLD).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

import pandas as pd

from trading_agent.models import Action, NewsItem, SignalStatus, TradeSignal
from trading_agent.strategy.prompts import (
    TRADING_JUDGE_SYSTEM_PROMPT,
    build_stock_prompt,
)
from trading_agent.tz import now as _now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Client — LiteLLM wrapper with retry + fallback + rule degradation
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
    """LiteLLM wrapper with retry, fallback chain, and rule-based degradation.

    Fallback order:
    1. Primary model (retry with exponential backoff)
    2. Each fallback model (sequential, retry with backoff each)
    3. Rule-based keyword scoring (no API call)
    """

    # Retry backoff settings
    _RETRY_BASE_DELAY = 1.0   # seconds
    _RETRY_ATTEMPTS = 2       # per model
    _FALLBACK_DELAY = 0.5     # seconds between models

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

        Tries primary → fallbacks (with delay) → rule-based degradation.
        """
        models = [self._model, *self._fallback_models]

        for i, model in enumerate(models):
            if i > 0:
                await asyncio.sleep(self._FALLBACK_DELAY)
            result = await self._try_model(model, system_prompt, user_prompt)
            if result is not None:
                return result

        # All providers failed — rule-based fallback
        logger.warning("All LLM providers failed, using rule-based fallback")
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

# Factor keys that should appear in prompts (whitelist).
# Anything outside this set (e.g. amount, adj_factor, ma5) is filtered out.
_KNOWN_FACTORS: frozenset[str] = frozenset({
    "rsi_14", "macd_hist", "bias_20", "obv", "atr_14",
    "volume_ratio", "pe_ttm", "pb", "roe_ttm", "macro_adj",
})


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


class LLMJudgeEngineImpl:
    """LLM comprehensive judge: evidence package → BUY/SELL/HOLD.

    Uses bull/bear/judge three-role debate (inspired by TradingAgents).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

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
        """Judge a single stock. Always returns a TradeSignal (never None)."""
        try:
            filtered_details = {
                k: v for k, v in factor_details.items() if k in _KNOWN_FACTORS
            }

            user_prompt = build_stock_prompt(
                symbol=symbol,
                name=name,
                factor_score=factor_score,
                factor_rank=factor_rank,
                factor_details=filtered_details,
                stock_news=stock_news,
                announcements=announcements,
                telegraph=telegraph,
                global_market=global_market,
                factor_signal=factor_signal,
                universe_size=universe_size,
            )

            result = await asyncio.wait_for(
                self._client.call(TRADING_JUDGE_SYSTEM_PROMPT, user_prompt),
                timeout=60,
            )
            return self._parse_signal(
                result, symbol, name, factor_score, factor_rank,
                filtered_details, price, _now(),
            )
        except Exception:  # noqa: BLE001
            logger.warning("LLM judge failed for %s, using factor fallback",
                           symbol, exc_info=True)
            return self._factor_fallback(
                symbol, name, factor_score, price, _now(),
            )

    async def judge_pool(
        self,
        candidates: list[dict[str, Any]],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        *,
        universe_size: int = 0,
    ) -> list[TradeSignal]:
        """Judge a pool of candidates sequentially.

        Each candidate dict must contain:
            symbol, name, factor_score, factor_rank, factor_details,
            stock_news, announcements, price

        Parameters
        ----------
        universe_size : total number of stocks in the factor universe (for rank display)
        """
        signals: list[TradeSignal] = []
        ok, fail = 0, 0
        pool_size = universe_size or len(candidates)

        for c in candidates:
            try:
                signal = await self.judge_stock(
                    symbol=c["symbol"],
                    name=c["name"],
                    factor_score=c["factor_score"],
                    factor_rank=c["factor_rank"],
                    factor_details=c["factor_details"],
                    stock_news=c.get("stock_news", []),
                    announcements=c.get("announcements", []),
                    telegraph=telegraph,
                    global_market=global_market,
                    price=c["price"],
                    factor_signal=c.get("factor_signal", ""),
                    universe_size=universe_size,
                )
                if signal.metadata is not None:
                    signal.metadata["pool_size"] = pool_size
                signals.append(signal)
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
                logger.warning("judge_pool: %s failed", c.get("symbol", "UNKNOWN"),
                               exc_info=True)

        logger.info("judge_pool: %d/%d succeeded, %d failed",
                    ok, len(candidates), fail)
        return signals

    def build_evidence_pool(
        self,
        candidate_symbols: list[str],
        pool_df: pd.DataFrame,
        factor_matrix: pd.DataFrame,
        status_map: dict[str, tuple[SignalStatus, int, float | None]],
        tech_df: dict[str, pd.DataFrame],
        name_map: dict[str, str],
        stock_news_map: dict[str, tuple[list, list]],
        *,
        factor_signal_map: dict[str, str] | None = None,
    ) -> list[dict]:
        """Build evidence packages for judge_pool. Pure data in, pure data out.

        Parameters
        ----------
        candidate_symbols : ordered list of symbols to evaluate
        pool_df : factor pool DataFrame (symbol, score, rank)
        factor_matrix : full factor matrix indexed by symbol
        status_map : symbol → (SignalStatus, days_in_top_n, prev_factor_score)
        tech_df : symbol → computed tech DataFrame (with 'close' column)
        name_map : symbol → display name
        stock_news_map : symbol → (stock_news, announcements)
        factor_signal_map : symbol → factor signal label (BUY/EXIT/WATCHLIST)
        """
        _fsm = factor_signal_map or {}
        evidence_pool: list[dict] = []
        for sym in candidate_symbols:
            row = pool_df[pool_df["symbol"] == sym] if not pool_df.empty else pd.DataFrame()
            f_score = float(row["score"].iloc[0]) if not row.empty else 0.5
            f_rank = int(row["rank"].iloc[0]) if not row.empty else len(pool_df) + 1
            f_details = (
                factor_matrix.loc[sym].to_dict()
                if sym in factor_matrix.index else {}
            )
            sig_status, days_in_top, prev_score = status_map.get(
                sym, (SignalStatus.NEW_ENTRY, 0, None),
            )
            s_news, s_anns = stock_news_map.get(sym, ([], []))

            bars = tech_df.get(sym)
            price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else 0.0

            evidence_pool.append({
                "symbol": sym,
                "name": name_map.get(sym, sym),
                "factor_score": f_score,
                "factor_rank": f_rank,
                "factor_details": f_details,
                "factor_signal": _fsm.get(sym, ""),
                "signal_status": sig_status.value,
                "days_in_top_n": days_in_top,
                "prev_factor_score": prev_score,
                "stock_news": s_news,
                "announcements": s_anns,
                "price": price,
            })
        return evidence_pool

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_signal(
        self,
        result: dict,
        symbol: str,
        name: str,
        factor_score: float,
        factor_rank: int,
        factor_details: dict[str, float],
        price: float,
        now: Any,
    ) -> TradeSignal:
        """Parse LLM JSON output into a TradeSignal."""
        raw_action = str(result.get("action", "HOLD")).upper().strip()
        try:
            action = Action(raw_action)
        except ValueError:
            logger.warning("Invalid action '%s' for %s, defaulting to HOLD",
                           raw_action, symbol)
            action = Action.HOLD

        confidence = float(result.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        bull = result.get("bull_reason", "")
        bear = result.get("bear_reason", "")
        conclusion = result.get("judge_conclusion", "")
        reason = conclusion

        return TradeSignal(
            timestamp=now,
            symbol=symbol,
            name=name,
            action=action,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=price,
            confidence=confidence,
            source="llm_judge",
            reason=reason,
            factor_score=factor_score,
            metadata={
                "bull_reason": bull,
                "bear_reason": bear,
                "judge_conclusion": conclusion,
                "short_term_outlook": result.get("short_term_outlook", ""),
                "mid_term_outlook": result.get("mid_term_outlook", ""),
                "factor_rank": factor_rank,
                "factor_details": factor_details,
            },
        )

    def _factor_fallback(
        self,
        symbol: str,
        name: str,
        factor_score: float,
        price: float,
        now: Any,
    ) -> TradeSignal:
        """Fallback when LLM is unavailable: use pure factor score."""
        if factor_score >= 0.7:
            action = Action.BUY
        elif factor_score <= 0.3:
            action = Action.SELL
        else:
            action = Action.HOLD

        return TradeSignal(
            timestamp=now,
            symbol=symbol,
            name=name,
            action=action,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=price,
            confidence=factor_score,
            source="factor_fallback",
            reason=f"LLM 不可用, 纯因子降级 (score={factor_score:.4f})",
            factor_score=factor_score,
            metadata={"fallback": True},
        )



# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------


class FakeLLMJudgeEngine:
    """Deterministic judge based on factor_score. For testing only.

    Unlike LLMJudgeEngineImpl, no error handling — expects well-formed input.
    """

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
    ) -> TradeSignal:

        if factor_score >= 0.7:
            action = Action.BUY
        elif factor_score <= 0.3:
            action = Action.SELL
        else:
            action = Action.HOLD

        return TradeSignal(
            timestamp=_now(),
            symbol=symbol,
            name=name,
            action=action,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=price,
            confidence=factor_score,
            source="fake_llm_judge",
            reason=f"fake judge: score={factor_score:.2f}",
            factor_score=factor_score,
            metadata={},
        )

    async def judge_pool(
        self,
        candidates: list[dict[str, Any]],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
    ) -> list[TradeSignal]:
        signals = []
        for c in candidates:
            signal = await self.judge_stock(
                symbol=c["symbol"],
                name=c["name"],
                factor_score=c["factor_score"],
                factor_rank=c["factor_rank"],
                factor_details=c.get("factor_details", {}),
                stock_news=c.get("stock_news", []),
                announcements=c.get("announcements", []),
                telegraph=telegraph,
                global_market=global_market,
                price=c["price"],
            )
            signals.append(signal)
        return signals
