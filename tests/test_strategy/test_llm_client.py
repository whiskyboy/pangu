"""Tests for LLMClient — M4.1."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pangu.strategy.llm.client import LLMClient


@pytest.fixture(autouse=True)
def _no_sleep():
    """Mock asyncio.sleep to avoid real delays in tests."""
    with patch("pangu.strategy.llm.client.asyncio.sleep", new_callable=AsyncMock):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RESPONSE = {
    "action": "BUY",
    "confidence": 0.8,
    "bull_reason": "重大合同签订",
    "bear_reason": "短期估值偏高",
    "judge_conclusion": "利好为主",
    "short_term_outlook": "短期看涨",
    "mid_term_outlook": "中期震荡",
}


def _make_completion(content: str) -> SimpleNamespace:
    """Build a mock litellm completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def setup_method(self) -> None:
        self.client = LLMClient(model="test/model")

    def test_pure_json(self) -> None:
        text = json.dumps(_VALID_RESPONSE)
        result = self.client._parse_json_response(text)
        assert result == _VALID_RESPONSE

    def test_fenced_json(self) -> None:
        text = f"Here is the analysis:\n```json\n{json.dumps(_VALID_RESPONSE)}\n```"
        result = self.client._parse_json_response(text)
        assert result == _VALID_RESPONSE

    def test_fenced_no_lang(self) -> None:
        text = f"Result:\n```\n{json.dumps(_VALID_RESPONSE)}\n```"
        result = self.client._parse_json_response(text)
        assert result == _VALID_RESPONSE

    def test_embedded_json(self) -> None:
        text = f"Based on my analysis, {json.dumps(_VALID_RESPONSE)} is the result."
        result = self.client._parse_json_response(text)
        assert result is not None
        assert result["action"] == "BUY"

    def test_empty_text(self) -> None:
        assert self.client._parse_json_response("") is None
        assert self.client._parse_json_response("  ") is None

    def test_invalid_json(self) -> None:
        assert self.client._parse_json_response("not json at all") is None

    def test_json_array_rejected(self) -> None:
        """Only dicts are accepted, not arrays."""
        assert self.client._parse_json_response("[1, 2, 3]") is None

    def test_deeply_nested_json(self) -> None:
        """3+ levels of nesting should parse correctly."""
        nested = {
            "action": "BUY",
            "confidence": 0.8,
            "bull_reason": "strong",
            "bear_reason": "weak",
            "judge_conclusion": "buy",
            "short_term_outlook": "up",
            "mid_term_outlook": "stable",
            "metadata": {"confidence": "high", "factors": {"fundamental": "strong"}},
        }
        text = f"Analysis result: {json.dumps(nested)} -- end"
        result = self.client._parse_json_response(text)
        assert result is not None
        assert result["action"] == "BUY"
        assert result["metadata"]["factors"]["fundamental"] == "strong"


# ---------------------------------------------------------------------------
# LLM call tests (mock litellm)
#
# Note: After the LLM-TopkDropout refactor, ``_rule_based_fallback`` was
# removed. ``LLMClient.call`` now raises RuntimeError after all retries fail —
# the caller (``judge_rebalance``) wraps this in its own try/except and falls
# back to deterministic ML ranking.
# ---------------------------------------------------------------------------


class TestLLMCall:
    @pytest.mark.asyncio
    async def test_primary_success(self) -> None:
        client = LLMClient(model="test/primary")
        mock_resp = _make_completion(json.dumps(_VALID_RESPONSE))

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.call("system", "user prompt")

        assert result["action"] == "BUY"
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_primary_fails_raises_runtime_error(self) -> None:
        """All connection errors after retries → RuntimeError."""
        client = LLMClient(model="test/primary")

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=ConnectionError("all down"),
        ):
            with pytest.raises(RuntimeError, match="failed after retries"):
                await client.call("system", "user prompt")

        assert client.call_count == 0  # no successful LLM calls

    @pytest.mark.asyncio
    async def test_unparseable_response_retries_then_raises(self) -> None:
        """Non-JSON content after retries → RuntimeError."""
        client = LLMClient(model="test/primary")
        bad_resp = _make_completion("I cannot provide JSON")

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=bad_resp):
            with pytest.raises(RuntimeError, match="failed after retries"):
                await client.call("system", "user prompt")

    @pytest.mark.asyncio
    async def test_empty_response_content_raises(self) -> None:
        """Empty content from LLM after retries → RuntimeError."""
        client = LLMClient(model="test/primary")
        empty_resp = _make_completion("")

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=empty_resp):
            with pytest.raises(RuntimeError, match="failed after retries"):
                await client.call("system", "user prompt")

    @pytest.mark.asyncio
    async def test_call_count_tracks_successful_only(self) -> None:
        client = LLMClient(model="test/primary")
        mock_resp = _make_completion(json.dumps(_VALID_RESPONSE))

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            await client.call("s1", "u1")
            await client.call("s2", "u2")

        assert client.call_count == 2
