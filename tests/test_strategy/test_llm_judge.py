"""Tests for LLMJudgeEngineImpl.judge_rebalance — pool-level debate."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from pangu.models import NewsItem, Region
from pangu.strategy.llm import RebalanceDecision
from pangu.strategy.llm.client import LLMClient
from pangu.strategy.llm.judge import LLMJudgeEngineImpl
from tests.fakes import FakeLLMJudgeEngine


@pytest.fixture(autouse=True)
def _no_sleep():
    """Mock asyncio.sleep to avoid real delays in retry backoff."""
    with patch("pangu.strategy.llm.client.asyncio.sleep", new_callable=AsyncMock):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GLOBAL_DF = pd.DataFrame(
    [
        {"name": "标普500", "close": 5200.0, "change_pct": 0.35},
    ]
)


def _make_news(title: str = "测试新闻") -> NewsItem:
    return NewsItem(
        timestamp=datetime(2026, 2, 19, 8, 0),
        title=title,
        content="正文",
        source="财联社",
        region=Region.DOMESTIC,
    )


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_candidate(
    symbol: str,
    name: str,
    ml_score: float,
    ml_rank: int,
    *,
    prev_ml_rank: int | None = None,
) -> dict:
    cand = {
        "symbol": symbol,
        "name": name,
        "ml_score": ml_score,
        "ml_rank": ml_rank,
        "factor_details": {"rsi_14": 55.0, "macd_hist": 0.1},
        "stock_news": [],
        "announcements": [],
    }
    if prev_ml_rank is not None:
        cand["prev_ml_rank"] = prev_ml_rank
        cand["rank_delta"] = ml_rank - prev_ml_rank
    return cand


def _llm_response(
    *,
    sells: list[dict] | None = None,
    buys: list[dict] | None = None,
    sell_debate: dict | None = None,
    buy_debate: dict | None = None,
) -> dict:
    return {
        "sells": sells or [],
        "buys": buys or [],
        "sell_debate": sell_debate or {"bull": "看多陈述", "bear": "看空陈述"},
        "buy_debate": buy_debate or {"bull": "看多陈述", "bear": "看空陈述"},
    }


# ---------------------------------------------------------------------------
# LLMJudgeEngineImpl.judge_rebalance
# ---------------------------------------------------------------------------


class TestJudgeRebalance:
    @pytest.fixture()
    def engine(self) -> LLMJudgeEngineImpl:
        return LLMJudgeEngineImpl(LLMClient(model="test/model"))

    @pytest.mark.asyncio
    async def test_happy_path(self, engine):
        sell_pool = [_make_candidate("600015", "S15", 0.3, 18, prev_ml_rank=12)]
        buy_pool = [
            _make_candidate("600000", "B00", 0.9, 1),
            _make_candidate("600001", "B01", 0.85, 2),
        ]
        resp = _llm_response(
            sells=[{"symbol": "600015", "reason": "估值偏高", "evidence": "ev1"}],
            buys=[{"symbol": "600000", "reason": "突破上行", "evidence": "ev2"}],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_completion(json.dumps(resp))):
            decision = await engine.judge_rebalance(
                today="2026-02-16",
                sell_candidates=sell_pool,
                buy_candidates=buy_pool,
                telegraph=[_make_news()],
                global_market=_GLOBAL_DF,
                top_n=25,
                n_drop=2,
            )

        assert isinstance(decision, RebalanceDecision)
        assert decision.source == "llm_judge"
        assert [p.symbol for p in decision.sells] == ["600015"]
        assert [p.symbol for p in decision.buys] == ["600000"]
        assert decision.sells[0].evidence == "ev1"
        assert decision.buys[0].evidence == "ev2"
        assert decision.sell_debate.bull == "看多陈述"
        assert decision.buy_debate.bear == "看空陈述"

    @pytest.mark.asyncio
    async def test_filters_out_of_pool(self, engine):
        """LLM picks not in the corresponding candidate pool are dropped."""
        sell_pool = [_make_candidate("600015", "S15", 0.3, 18)]
        buy_pool = [_make_candidate("600000", "B00", 0.9, 1)]
        # LLM hallucinates 999999 → must be filtered
        resp = _llm_response(
            sells=[
                {"symbol": "999999", "reason": "ghost", "evidence": ""},
                {"symbol": "600015", "reason": "ok", "evidence": ""},
            ],
            buys=[
                {"symbol": "888888", "reason": "ghost", "evidence": ""},
                {"symbol": "600000", "reason": "ok", "evidence": ""},
            ],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_completion(json.dumps(resp))):
            decision = await engine.judge_rebalance(
                today="2026-02-16",
                sell_candidates=sell_pool,
                buy_candidates=buy_pool,
                telegraph=[],
                global_market=_GLOBAL_DF,
                top_n=25,
                n_drop=2,
            )

        assert [p.symbol for p in decision.sells] == ["600015"]
        assert [p.symbol for p in decision.buys] == ["600000"]

    @pytest.mark.asyncio
    async def test_caps_at_n_drop(self, engine):
        """LLM picks beyond ``n_drop`` are truncated."""
        sell_pool = [_make_candidate(f"60{i:04d}", f"S{i}", 0.3 - i * 0.01, 20 - i) for i in range(5)]
        buy_pool = [_make_candidate(f"61{i:04d}", f"B{i}", 0.9 - i * 0.01, i + 1) for i in range(5)]
        resp = _llm_response(
            sells=[{"symbol": c["symbol"], "reason": "x", "evidence": ""} for c in sell_pool],
            buys=[{"symbol": c["symbol"], "reason": "x", "evidence": ""} for c in buy_pool],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_completion(json.dumps(resp))):
            decision = await engine.judge_rebalance(
                today="2026-02-16",
                sell_candidates=sell_pool,
                buy_candidates=buy_pool,
                telegraph=[],
                global_market=_GLOBAL_DF,
                top_n=25,
                n_drop=2,
            )

        assert len(decision.sells) == 2
        assert len(decision.buys) == 2

    @pytest.mark.asyncio
    async def test_skips_picks_without_reason(self, engine):
        sell_pool = [_make_candidate("600015", "S15", 0.3, 18)]
        resp = _llm_response(
            sells=[
                {"symbol": "600015", "reason": "", "evidence": ""},  # empty reason → dropped
            ],
            buys=[],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_completion(json.dumps(resp))):
            decision = await engine.judge_rebalance(
                today="2026-02-16",
                sell_candidates=sell_pool,
                buy_candidates=[],
                telegraph=[],
                global_market=_GLOBAL_DF,
                top_n=25,
                n_drop=2,
            )

        assert decision.sells == []

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_decision(self, engine):
        """When the LLM raises an exception, returns source='llm_failed'."""
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API down")):
            decision = await engine.judge_rebalance(
                today="2026-02-16",
                sell_candidates=[_make_candidate("600015", "S15", 0.3, 18)],
                buy_candidates=[_make_candidate("600000", "B00", 0.9, 1)],
                telegraph=[],
                global_market=_GLOBAL_DF,
                top_n=25,
                n_drop=2,
            )

        assert decision.source == "llm_failed"
        assert decision.sells == []
        assert decision.buys == []

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_decision(self, engine):
        """A hanging LLM call beyond timeout is treated as failure."""

        async def hanging_call(*_args, **_kwargs):
            raise asyncio.TimeoutError()

        engine._client.call = hanging_call
        decision = await engine.judge_rebalance(
            today="2026-02-16",
            sell_candidates=[_make_candidate("600015", "S15", 0.3, 18)],
            buy_candidates=[],
            telegraph=[],
            global_market=_GLOBAL_DF,
            top_n=25,
            n_drop=2,
            timeout=0.01,
        )

        assert decision.source == "llm_failed"

    @pytest.mark.asyncio
    async def test_empty_pools_short_circuits(self, engine):
        """If both pools are empty, no LLM call is made."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ll:
            decision = await engine.judge_rebalance(
                today="2026-02-16",
                sell_candidates=[],
                buy_candidates=[],
                telegraph=[],
                global_market=_GLOBAL_DF,
                top_n=25,
                n_drop=2,
            )

        mock_ll.assert_not_called()
        assert decision.source == "llm_judge"
        assert decision.sells == []
        assert decision.buys == []

    @pytest.mark.asyncio
    async def test_dedupes_picks(self, engine):
        """Duplicate symbols in LLM response are kept only once."""
        sell_pool = [_make_candidate("600015", "S15", 0.3, 18)]
        resp = _llm_response(
            sells=[
                {"symbol": "600015", "reason": "first", "evidence": ""},
                {"symbol": "600015", "reason": "second", "evidence": ""},
            ],
            buys=[],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=_make_completion(json.dumps(resp))):
            decision = await engine.judge_rebalance(
                today="2026-02-16",
                sell_candidates=sell_pool,
                buy_candidates=[],
                telegraph=[],
                global_market=_GLOBAL_DF,
                top_n=25,
                n_drop=2,
            )

        assert len(decision.sells) == 1
        assert decision.sells[0].reason == "first"


# ---------------------------------------------------------------------------
# FakeLLMJudgeEngine
# ---------------------------------------------------------------------------


class TestFakeLLMJudgeEngine:
    @pytest.mark.asyncio
    async def test_returns_top_n_drop_from_each_pool(self):
        engine = FakeLLMJudgeEngine()
        sells = [_make_candidate(f"60{i:04d}", f"S{i}", 0.3, 20 - i) for i in range(5)]
        buys = [_make_candidate(f"61{i:04d}", f"B{i}", 0.9, i + 1) for i in range(5)]

        decision = await engine.judge_rebalance(
            today="2026-02-16",
            sell_candidates=sells,
            buy_candidates=buys,
            telegraph=[],
            global_market=_GLOBAL_DF,
            top_n=25,
            n_drop=3,
        )

        assert isinstance(decision, RebalanceDecision)
        assert decision.source == "llm_judge"
        assert [p.symbol for p in decision.sells] == ["600000", "600001", "600002"]
        assert [p.symbol for p in decision.buys] == ["610000", "610001", "610002"]
        assert decision.sells[0].reason == "fake sell"
        assert decision.buys[0].reason == "fake buy"

    @pytest.mark.asyncio
    async def test_empty_pools(self):
        engine = FakeLLMJudgeEngine()
        decision = await engine.judge_rebalance(
            today="2026-02-16",
            sell_candidates=[],
            buy_candidates=[],
            telegraph=[],
            global_market=_GLOBAL_DF,
            top_n=25,
            n_drop=3,
        )
        assert decision.sells == []
        assert decision.buys == []
