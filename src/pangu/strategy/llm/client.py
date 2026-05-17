"""LLM client and Protocol — LiteLLM wrapper for pool-level rebalance debate."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol

import pandas as pd

from pangu.models import NewsItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Client — LiteLLM wrapper with retry
# ---------------------------------------------------------------------------


class LLMClient:
    """LiteLLM wrapper with retry and structured JSON output.

    Raises RuntimeError if the model fails to return a parseable JSON object
    after all retries — callers (e.g. judge_rebalance) wrap this in their own
    try/except and fall back to deterministic ML ranking when the LLM is down.
    """

    _RETRY_BASE_DELAY = 1.0  # seconds
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
        """Call LLM and return parsed JSON dict, or raise RuntimeError."""
        result = await self._try_model(self._model, system_prompt, user_prompt)
        if result is not None:
            return result
        raise RuntimeError(f"LLM model {self._model} failed after retries")

    # ---------------------------------------------------------------------------
    # LLMJudgeEngine Protocol — pool-level rebalance debate
    # ---------------------------------------------------------------------------

    async def _try_model(self, model: str, system_prompt: str, user_prompt: str) -> dict | None:
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
                    model,
                    attempt + 1,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Model %s failed (attempt %d)",
                    model,
                    attempt + 1,
                    exc_info=True,
                )
            # Exponential backoff before retry (skip after last attempt)
            if attempt < self._RETRY_ATTEMPTS - 1:
                delay = self._RETRY_BASE_DELAY * (2**attempt)
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


# ---------------------------------------------------------------------------
# LLMJudgeEngine Protocol — pool-level rebalance debate
# ---------------------------------------------------------------------------


class LLMJudgeEngine(Protocol):
    """Interface for pool-level LLM rebalance judge (Bull/Bear/Judge debate)."""

    async def judge_rebalance(
        self,
        *,
        today: str,
        sell_candidates: list[dict[str, Any]],
        buy_candidates: list[dict[str, Any]],
        telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        top_n: int,
        n_drop: int,
        universe_size: int = 0,
        timeout: float = 120.0,
    ) -> Any:
        """Judge SELL + BUY candidate pools and return a RebalanceDecision."""
        ...
