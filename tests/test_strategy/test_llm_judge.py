"""Tests for LLMJudgeEngine — M4.4."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_sleep():
    """Mock asyncio.sleep to avoid real delays in tests."""
    with patch("trading_agent.strategy.llm_engine.asyncio.sleep", new_callable=AsyncMock):
        yield


from trading_agent.models import Action, NewsCategory, NewsItem, Region, SignalStatus
from trading_agent.strategy.llm_engine import (
    FakeLLMJudgeEngine,
    LLMClient,
    LLMJudgeEngineImpl,
    _KNOWN_FACTORS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(
    action: str = "BUY",
    confidence: float = 0.8,
    bull: str = "利好明确",
    bear: str = "短期有压力",
    conclusion: str = "综合看多",
) -> dict:
    return {
        "action": action,
        "confidence": confidence,
        "bull_reason": bull,
        "bear_reason": bear,
        "judge_conclusion": conclusion,
        "short_term_outlook": "短期看涨",
        "mid_term_outlook": "中期震荡",
    }


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_news(title: str = "测试新闻") -> NewsItem:
    return NewsItem(
        timestamp=datetime(2026, 2, 19, 8, 0),
        title=title,
        content="正文",
        source="财联社",
        region=Region.DOMESTIC,
    )


_GLOBAL_DF = pd.DataFrame([
    {"name": "标普500", "close": 5200.0, "change_pct": 0.35},
])

_FACTOR_DETAILS = {
    "rsi_14": 55.0,
    "macd_hist": 0.12,
    "pe_ttm": 15.3,
    "amount": 1e10,       # should be filtered
    "adj_factor": 1.0,    # should be filtered
    "ma5": 38.0,          # should be filtered
}


# ---------------------------------------------------------------------------
# LLMJudgeEngineImpl tests
# ---------------------------------------------------------------------------

class TestLLMJudgeEngineImpl:

    @pytest.fixture()
    def engine(self) -> LLMJudgeEngineImpl:
        client = LLMClient(model="test/model")
        return LLMJudgeEngineImpl(client)

    @pytest.mark.asyncio
    async def test_judge_stock_buy_signal(self, engine):
        resp = _make_llm_response(action="BUY", confidence=0.85)
        content = json.dumps(resp)

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            signal = await engine.judge_stock(
                symbol="601899", name="紫金矿业",
                factor_score=0.82, factor_rank=2,
                factor_details=_FACTOR_DETAILS,
                stock_news=[_make_news()], announcements=[],
                telegraph=[_make_news("快讯")],
                global_market=_GLOBAL_DF, price=39.5,
            )

        assert signal.action == Action.BUY
        assert signal.confidence == 0.85
        assert signal.source == "llm_judge"
        assert signal.symbol == "601899"
        assert signal.price == 39.5
        assert signal.factor_score == 0.82
        assert signal.reason == "综合看多"
        assert signal.metadata["bull_reason"] == "利好明确"
        assert signal.metadata["factor_rank"] == 2

    @pytest.mark.asyncio
    async def test_judge_stock_sell_signal(self, engine):
        resp = _make_llm_response(action="SELL", confidence=0.7)
        content = json.dumps(resp)

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            signal = await engine.judge_stock(
                symbol="000001", name="平安银行",
                factor_score=0.25, factor_rank=50,
                factor_details={"rsi_14": 30.0},
                stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=10.0,
            )

        assert signal.action == Action.SELL
        assert signal.confidence == 0.7
        assert signal.source == "llm_judge"

    @pytest.mark.asyncio
    async def test_judge_stock_hold_signal(self, engine):
        resp = _make_llm_response(action="HOLD", confidence=0.5)
        content = json.dumps(resp)

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            signal = await engine.judge_stock(
                symbol="000001", name="平安银行",
                factor_score=0.5, factor_rank=10,
                factor_details={}, stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=10.0,
            )

        assert signal.action == Action.HOLD
        assert signal.source == "llm_judge"

    @pytest.mark.asyncio
    async def test_judge_stock_invalid_action_defaults_hold(self, engine):
        resp = _make_llm_response(action="INVALID")
        content = json.dumps(resp)

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            signal = await engine.judge_stock(
                symbol="000001", name="平安银行",
                factor_score=0.5, factor_rank=10,
                factor_details={}, stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=10.0,
            )

        assert signal.action == Action.HOLD

    @pytest.mark.asyncio
    async def test_judge_stock_all_llm_fail_uses_rule_fallback(self, engine):
        """When all LLM providers fail, LLMClient returns rule-based result."""
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   side_effect=Exception("API down")):
            signal = await engine.judge_stock(
                symbol="601899", name="紫金矿业",
                factor_score=0.82, factor_rank=2,
                factor_details={"rsi_14": 55.0},
                stock_news=[_make_news("利好消息，增持回购")],
                announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=39.5,
            )

        # Rule-based fallback still produces a valid signal via _parse_signal
        assert signal.source == "llm_judge"
        assert "规则降级" in signal.metadata["judge_conclusion"]

    @pytest.mark.asyncio
    async def test_judge_stock_factor_fallback_on_internal_error(self, engine):
        """Factor fallback triggers when build_stock_prompt or _parse_signal fails."""
        # patching at llm_engine module level
        with patch("trading_agent.strategy.llm_engine.build_stock_prompt",
                          side_effect=TypeError("unexpected error")):
            signal = await engine.judge_stock(
                symbol="601899", name="紫金矿业",
                factor_score=0.82, factor_rank=2,
                factor_details={"rsi_14": 55.0},
                stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=39.5,
            )

        assert signal.action == Action.BUY  # factor_score=0.82 >= 0.7
        assert signal.source == "factor_fallback"
        assert signal.metadata["fallback"] is True

    @pytest.mark.asyncio
    async def test_judge_stock_factor_fallback_sell(self, engine):
        # patching at llm_engine module level
        with patch("trading_agent.strategy.llm_engine.build_stock_prompt",
                          side_effect=TypeError("unexpected error")):
            signal = await engine.judge_stock(
                symbol="000001", name="平安银行",
                factor_score=0.2, factor_rank=50,
                factor_details={}, stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=10.0,
            )

        assert signal.action == Action.SELL
        assert signal.source == "factor_fallback"

    @pytest.mark.asyncio
    async def test_judge_stock_factor_fallback_hold(self, engine):
        # patching at llm_engine module level
        with patch("trading_agent.strategy.llm_engine.build_stock_prompt",
                          side_effect=TypeError("unexpected error")):
            signal = await engine.judge_stock(
                symbol="000001", name="平安银行",
                factor_score=0.5, factor_rank=10,
                factor_details={}, stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=10.0,
            )

        assert signal.action == Action.HOLD
        assert signal.source == "factor_fallback"

    @pytest.mark.asyncio
    async def test_judge_stock_filters_non_factor_columns(self, engine):
        """Verify amount/adj_factor/ma5 are NOT passed to build_stock_prompt."""
        resp = _make_llm_response()
        content = json.dumps(resp)

        from trading_agent.strategy.prompts import build_stock_prompt as real_build
        captured_details: list[dict] = []

        def capture_build(*args, **kwargs):
            details = kwargs.get(
                "factor_details",
                args[4] if len(args) > 4 else {},
            )
            captured_details.append(dict(details))
            return real_build(*args, **kwargs)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock,
                  return_value=_make_completion(content)),
            patch("trading_agent.strategy.llm_engine.build_stock_prompt",
                  side_effect=capture_build),
        ):
            await engine.judge_stock(
                symbol="601899", name="紫金矿业",
                factor_score=0.82, factor_rank=2,
                factor_details=_FACTOR_DETAILS,
                stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=39.5,
            )

        assert len(captured_details) == 1
        passed = captured_details[0]
        assert "rsi_14" in passed
        assert "macd_hist" in passed
        assert "pe_ttm" in passed
        assert "amount" not in passed
        assert "adj_factor" not in passed
        assert "ma5" not in passed

    @pytest.mark.asyncio
    async def test_judge_stock_confidence_clamped(self, engine):
        resp = _make_llm_response(confidence=1.5)
        content = json.dumps(resp)

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            signal = await engine.judge_stock(
                symbol="000001", name="平安银行",
                factor_score=0.5, factor_rank=10,
                factor_details={}, stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=10.0,
            )

        assert signal.confidence == 1.0

    @pytest.mark.asyncio
    async def test_judge_stock_metadata_has_outlooks(self, engine):
        resp = _make_llm_response()
        content = json.dumps(resp)

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            signal = await engine.judge_stock(
                symbol="000001", name="平安银行",
                factor_score=0.5, factor_rank=10,
                factor_details={}, stock_news=[], announcements=[],
                telegraph=[], global_market=_GLOBAL_DF, price=10.0,
            )

        assert "short_term_outlook" in signal.metadata
        assert "mid_term_outlook" in signal.metadata
        assert signal.metadata["judge_conclusion"] == "综合看多"


# ---------------------------------------------------------------------------
# judge_pool tests
# ---------------------------------------------------------------------------

class TestJudgePool:

    @pytest.fixture()
    def engine(self) -> LLMJudgeEngineImpl:
        client = LLMClient(model="test/model")
        return LLMJudgeEngineImpl(client)

    def _make_candidate(self, symbol: str, name: str,
                        score: float, rank: int, price: float) -> dict:
        return {
            "symbol": symbol, "name": name,
            "factor_score": score, "factor_rank": rank,
            "factor_details": {"rsi_14": 50.0},
            "stock_news": [], "announcements": [],
            "price": price,
        }

    @pytest.mark.asyncio
    async def test_judge_pool_collects_signals(self, engine):
        resp = _make_llm_response()
        content = json.dumps(resp)

        candidates = [
            self._make_candidate("601899", "紫金矿业", 0.82, 1, 39.5),
            self._make_candidate("000001", "平安银行", 0.65, 2, 10.0),
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            signals = await engine.judge_pool(
                candidates, telegraph=[], global_market=_GLOBAL_DF,
            )

        assert len(signals) == 2
        symbols = {s.symbol for s in signals}
        assert symbols == {"601899", "000001"}

    @pytest.mark.asyncio
    async def test_judge_pool_partial_failure(self, engine):
        """When one stock's LLM fails, it gets rule-based fallback, not factor fallback."""
        resp = _make_llm_response()
        content = json.dumps(resp)

        async def flaky_completion(**kwargs):
            msgs = kwargs.get("messages", [])
            user_msg = msgs[1]["content"] if len(msgs) > 1 else ""
            if "000001" in user_msg:
                raise Exception("API error for 000001")
            return _make_completion(content)

        candidates = [
            self._make_candidate("601899", "紫金矿业", 0.82, 1, 39.5),
            self._make_candidate("000001", "平安银行", 0.65, 2, 10.0),
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock,
                   side_effect=flaky_completion):
            signals = await engine.judge_pool(
                candidates, telegraph=[], global_market=_GLOBAL_DF,
            )

        # Both return signals: one from LLM, one from rule-based fallback
        assert len(signals) == 2
        by_sym = {s.symbol: s for s in signals}
        assert by_sym["601899"].source == "llm_judge"
        # 000001 gets rule-based fallback (still source="llm_judge")
        assert by_sym["000001"].source == "llm_judge"
        assert "规则降级" in by_sym["000001"].metadata["judge_conclusion"]

    @pytest.mark.asyncio
    async def test_judge_pool_empty_candidates(self, engine):
        signals = await engine.judge_pool(
            candidates=[], telegraph=[], global_market=_GLOBAL_DF,
        )
        assert signals == []


# ---------------------------------------------------------------------------
# FakeLLMJudgeEngine tests
# ---------------------------------------------------------------------------

class TestFakeLLMJudgeEngine:

    @pytest.mark.asyncio
    async def test_high_score_buy(self):
        engine = FakeLLMJudgeEngine()
        signal = await engine.judge_stock(
            symbol="601899", name="紫金矿业",
            factor_score=0.8, factor_rank=1,
            factor_details={}, stock_news=[], announcements=[],
            telegraph=[], global_market=_GLOBAL_DF, price=39.5,
        )
        assert signal.action == Action.BUY
        assert signal.source == "fake_llm_judge"

    @pytest.mark.asyncio
    async def test_low_score_sell(self):
        engine = FakeLLMJudgeEngine()
        signal = await engine.judge_stock(
            symbol="000001", name="平安银行",
            factor_score=0.2, factor_rank=50,
            factor_details={}, stock_news=[], announcements=[],
            telegraph=[], global_market=_GLOBAL_DF, price=10.0,
        )
        assert signal.action == Action.SELL

    @pytest.mark.asyncio
    async def test_mid_score_hold(self):
        engine = FakeLLMJudgeEngine()
        signal = await engine.judge_stock(
            symbol="000001", name="平安银行",
            factor_score=0.5, factor_rank=10,
            factor_details={}, stock_news=[], announcements=[],
            telegraph=[], global_market=_GLOBAL_DF, price=10.0,
        )
        assert signal.action == Action.HOLD

    @pytest.mark.asyncio
    async def test_judge_pool(self):
        engine = FakeLLMJudgeEngine()
        candidates = [
            {"symbol": "601899", "name": "紫金矿业",
             "factor_score": 0.8, "factor_rank": 1, "price": 39.5},
            {"symbol": "000001", "name": "平安银行",
             "factor_score": 0.2, "factor_rank": 2, "price": 10.0},
        ]
        signals = await engine.judge_pool(
            candidates, telegraph=[], global_market=_GLOBAL_DF,
        )
        assert len(signals) == 2
        assert signals[0].action == Action.BUY
        assert signals[1].action == Action.SELL


# ---------------------------------------------------------------------------
# build_evidence_pool
# ---------------------------------------------------------------------------


class TestBuildEvidencePool:
    def test_basic_build(self):
        from trading_agent.strategy.llm_engine import LLMClient

        engine = LLMJudgeEngineImpl(LLMClient())
        pool_df = pd.DataFrame({
            "symbol": ["600519", "000858"],
            "score": [0.9, 0.7],
            "rank": [1, 2],
        })
        factor_matrix = pd.DataFrame(
            {"rsi_14": [55.0, 60.0], "macd_hist": [0.1, -0.1]},
            index=["600519", "000858"],
        )
        status_map = {
            "600519": (SignalStatus.NEW_ENTRY, 0, None),
            "000858": (SignalStatus.SUSTAINED, 2, 0.65),
        }
        tech_df = {
            "600519": pd.DataFrame({"close": [1850.0]}),
            "000858": pd.DataFrame({"close": [150.0]}),
        }
        name_map = {"600519": "贵州茅台", "000858": "五粮液"}
        stock_news_map = {"600519": ([], []), "000858": ([], [])}

        result = engine.build_evidence_pool(
            ["600519", "000858"], pool_df, factor_matrix,
            status_map, tech_df, name_map, stock_news_map,
        )
        assert len(result) == 2
        assert result[0]["symbol"] == "600519"
        assert result[0]["factor_score"] == pytest.approx(0.9)
        assert result[0]["factor_rank"] == 1
        assert result[0]["signal_status"] == "new_entry"
        assert result[0]["price"] == pytest.approx(1850.0)
        assert result[1]["prev_factor_score"] == pytest.approx(0.65)

    def test_missing_symbol_gets_defaults(self):
        from trading_agent.strategy.llm_engine import LLMClient

        engine = LLMJudgeEngineImpl(LLMClient())
        pool_df = pd.DataFrame(columns=["symbol", "score", "rank"])
        factor_matrix = pd.DataFrame()
        result = engine.build_evidence_pool(
            ["999999"], pool_df, factor_matrix,
            {}, {}, {}, {},
        )
        assert len(result) == 1
        assert result[0]["factor_score"] == 0.5
        assert result[0]["price"] == 0.0
        assert result[0]["name"] == "999999"  # fallback to symbol
