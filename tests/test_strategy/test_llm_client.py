"""Tests for LLMClient — M4.1."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from trading_agent.strategy.llm_engine import LLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RESPONSE = {
    "direction": "bullish",
    "impact_score": 8,
    "bull_reason": "重大合同签订",
    "bear_reason": "短期估值偏高",
    "judge_conclusion": "利好为主",
    "affected_symbols": ["601899"],
    "affected_sectors": ["有色金属"],
}


def _make_completion(content: str) -> SimpleNamespace:
    """Build a mock litellm completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


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
        assert result["direction"] == "bullish"

    def test_empty_text(self) -> None:
        assert self.client._parse_json_response("") is None
        assert self.client._parse_json_response("  ") is None

    def test_invalid_json(self) -> None:
        assert self.client._parse_json_response("not json at all") is None

    def test_json_array_rejected(self) -> None:
        """Only dicts are accepted, not arrays."""
        assert self.client._parse_json_response('[1, 2, 3]') is None

    def test_deeply_nested_json(self) -> None:
        """3+ levels of nesting should parse correctly."""
        nested = {
            "direction": "bullish",
            "impact_score": 8,
            "bull_reason": "strong",
            "bear_reason": "weak",
            "judge_conclusion": "buy",
            "affected_symbols": ["601899"],
            "affected_sectors": [],
            "metadata": {"confidence": "high", "factors": {"fundamental": "strong"}},
        }
        text = f"Analysis result: {json.dumps(nested)} -- end"
        result = self.client._parse_json_response(text)
        assert result is not None
        assert result["direction"] == "bullish"
        assert result["metadata"]["factors"]["fundamental"] == "strong"


# ---------------------------------------------------------------------------
# Rule-based fallback tests
# ---------------------------------------------------------------------------


class TestRuleBasedFallback:
    def setup_method(self) -> None:
        self.client = LLMClient(model="test/model")

    def test_bullish_keywords(self) -> None:
        text = "该公司获批新药，业绩预增超预期，增持计划公布"
        result = self.client._rule_based_fallback(text)
        assert result["direction"] == "bullish"
        assert result["impact_score"] >= 5

    def test_bearish_keywords(self) -> None:
        text = "业绩下滑，大股东减持，面临处罚风险"
        result = self.client._rule_based_fallback(text)
        assert result["direction"] == "bearish"
        assert result["impact_score"] >= 5

    def test_neutral_no_keywords(self) -> None:
        text = "公司召开年度股东大会，审议日常议案"
        result = self.client._rule_based_fallback(text)
        assert result["direction"] == "neutral"
        assert result["impact_score"] == 3

    def test_structure_complete(self) -> None:
        result = self.client._rule_based_fallback("test")
        required_keys = [
            "direction", "impact_score", "bull_reason",
            "bear_reason", "judge_conclusion",
            "affected_symbols", "affected_sectors",
        ]
        for key in required_keys:
            assert key in result

    def test_impact_score_capped(self) -> None:
        """Score should never exceed 10."""
        text = " ".join(["利好 增持 回购 业绩预增 超预期 突破 获批 中标"] * 3)
        result = self.client._rule_based_fallback(text)
        assert result["impact_score"] <= 10


# ---------------------------------------------------------------------------
# LLM call tests (mock litellm)
# ---------------------------------------------------------------------------


class TestLLMCall:
    @pytest.mark.asyncio
    async def test_primary_success(self) -> None:
        client = LLMClient(model="test/primary")
        mock_resp = _make_completion(json.dumps(_VALID_RESPONSE))

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.call("system", "user prompt")

        assert result["direction"] == "bullish"
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self) -> None:
        client = LLMClient(
            model="test/primary",
            fallback_models=["test/fallback1"],
        )
        mock_resp = _make_completion(json.dumps(_VALID_RESPONSE))

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model") == "test/primary":
                raise ConnectionError("primary down")
            return mock_resp

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=side_effect):
            result = await client.call("system", "user prompt")

        assert result["direction"] == "bullish"
        # Primary tried 2 times (initial + retry), then fallback 1 time
        assert client.call_count == 1  # only successful calls counted

    @pytest.mark.asyncio
    async def test_all_providers_fail_rule_fallback(self) -> None:
        client = LLMClient(
            model="test/primary",
            fallback_models=["test/fallback1"],
        )

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=ConnectionError("all down"),
        ):
            result = await client.call("system", "重大合同签订，利好明显")

        assert result["direction"] == "bullish"
        assert "规则降级" in result["judge_conclusion"]
        assert client.call_count == 0  # no successful LLM calls

    @pytest.mark.asyncio
    async def test_unparseable_response_retries(self) -> None:
        """If LLM returns non-JSON, retry then try fallback."""
        client = LLMClient(
            model="test/primary",
            fallback_models=["test/fallback1"],
        )
        bad_resp = _make_completion("I cannot provide JSON")
        good_resp = _make_completion(json.dumps(_VALID_RESPONSE))

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model") == "test/primary":
                return bad_resp
            return good_resp

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=side_effect):
            result = await client.call("system", "user prompt")

        assert result["direction"] == "bullish"

    @pytest.mark.asyncio
    async def test_empty_response_content(self) -> None:
        """Empty content from LLM should trigger fallback."""
        client = LLMClient(model="test/primary")
        empty_resp = _make_completion("")

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=empty_resp):
            result = await client.call("system", "利空消息，减持风险")

        assert result["direction"] == "bearish"
        assert "规则降级" in result["judge_conclusion"]

    @pytest.mark.asyncio
    async def test_call_count_tracks_successful_only(self) -> None:
        client = LLMClient(model="test/primary")
        mock_resp = _make_completion(json.dumps(_VALID_RESPONSE))

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            await client.call("s1", "u1")
            await client.call("s2", "u2")

        assert client.call_count == 2
