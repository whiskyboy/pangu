"""Tests for judge_rebalance — pool-level Bull/Bear/Judge LLM decision."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from pangu.models import NewsItem, Region
from pangu.strategy.llm import (
    DebateNotes,
    LLMClient,
    LLMJudgeEngineImpl,
    RebalanceDecision,
)
from pangu.strategy.llm.prompts import (
    build_rebalance_prompt,
    build_rebalance_system_prompt,
)


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch("pangu.strategy.llm.client.asyncio.sleep", new_callable=AsyncMock):
        yield


_GLOBAL_DF = pd.DataFrame([
    {"name": "标普500", "close": 5200.0, "change_pct": 0.35},
])


def _make_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_news(title: str = "测试新闻") -> NewsItem:
    return NewsItem(
        timestamp=datetime(2026, 2, 19, 8, 0),
        title=title,
        content="正文内容用于截断测试" * 5,
        source="财联社",
        region=Region.DOMESTIC,
    )


def _make_candidate(
    symbol: str, *, score: float = 0.5, rank: int = 1,
    prev_rank: int | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "name": f"name-{symbol}",
        "ml_score": score,
        "ml_rank": rank,
        "prev_ml_rank": prev_rank,
        "rank_delta": (rank - prev_rank) if prev_rank is not None else None,
        "factor_details": {"rsi_14": 55.0, "macd_hist": 0.12, "amount": 1e10},
        "stock_news": [_make_news()],
        "announcements": [],
    }


def _ok_response(sells: list[dict] | None = None, buys: list[dict] | None = None) -> str:
    payload = {
        "sell_debate": {"bull": "持有方理由", "bear": "卖出方理由"},
        "buy_debate": {"bull": "买入方理由", "bear": "回避方理由"},
        "sells": sells if sells is not None else [],
        "buys": buys if buys is not None else [],
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

class TestRebalancePromptBuilders:
    def test_system_prompt_substitutes_pool_sizes(self):
        p = build_rebalance_system_prompt(
            top_n=25, n_drop=3, sell_pool_size=5, buy_pool_size=10,
        )
        assert "25" in p
        assert "n_drop=3" in p
        assert "ML 评分最差的 5 只" in p
        assert "ML 评分最好的 10 只" in p

    def test_user_prompt_contains_both_pools(self):
        p = build_rebalance_prompt(
            today="2025-03-10",
            sell_candidates=[_make_candidate("600519", score=0.1, rank=300,
                                             prev_rank=50)],
            buy_candidates=[_make_candidate("000001", score=0.95, rank=1)],
            telegraph=[_make_news("利好快讯")],
            global_market=_GLOBAL_DF,
            top_n=25, n_drop=3, universe_size=800,
        )
        assert "2025-03-10" in p
        assert "SELL 候选池" in p
        assert "BUY 候选池" in p
        assert "600519" in p
        assert "000001" in p
        # rank delta is rendered for sell side
        assert "变化" in p
        # global market shown
        assert "标普500" in p

    def test_user_prompt_empty_pools(self):
        p = build_rebalance_prompt(
            today="2025-03-10",
            sell_candidates=[], buy_candidates=[],
            telegraph=[], global_market=_GLOBAL_DF,
            top_n=25, n_drop=3,
        )
        assert "SELL 候选池" in p
        assert "BUY 候选池" in p
        assert "（空）" in p


# ---------------------------------------------------------------------------
# judge_rebalance
# ---------------------------------------------------------------------------

class TestJudgeRebalance:
    @pytest.fixture
    def engine(self) -> LLMJudgeEngineImpl:
        return LLMJudgeEngineImpl(LLMClient(model="test/model"))

    @pytest.mark.asyncio
    async def test_returns_picks_on_well_formed_response(self, engine):
        sells = [
            _make_candidate("600519", score=0.1, rank=300, prev_rank=50),
            _make_candidate("000858", score=0.2, rank=290, prev_rank=60),
        ]
        buys = [
            _make_candidate("000001", score=0.95, rank=1),
            _make_candidate("601318", score=0.92, rank=2),
        ]
        content = _ok_response(
            sells=[
                {"symbol": "600519", "reason": "业绩雷", "evidence": "公告"}
            ],
            buys=[
                {"symbol": "000001", "reason": "央行降准受益", "evidence": "央行公告"},
                {"symbol": "601318", "reason": "估值低", "evidence": "PE 5"},
            ],
        )
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=sells, buy_candidates=buys,
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3, universe_size=800,
            )
        assert isinstance(decision, RebalanceDecision)
        assert decision.source == "llm_judge"
        assert [p.symbol for p in decision.sells] == ["600519"]
        assert decision.sells[0].reason == "业绩雷"
        assert decision.sells[0].evidence == "公告"
        assert [p.symbol for p in decision.buys] == ["000001", "601318"]
        assert decision.sell_debate.bull == "持有方理由"
        assert decision.buy_debate.bear == "回避方理由"

    @pytest.mark.asyncio
    async def test_drops_picks_outside_candidate_pool(self, engine):
        sells = [_make_candidate("600519", score=0.1, rank=300)]
        buys = [_make_candidate("000001", score=0.95, rank=1)]
        content = _ok_response(
            sells=[{"symbol": "999999", "reason": "未在 sell 池", "evidence": ""}],
            buys=[
                {"symbol": "000001", "reason": "OK", "evidence": ""},
                {"symbol": "888888", "reason": "未在 buy 池", "evidence": ""},
            ],
        )
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=sells, buy_candidates=buys,
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3,
            )
        assert decision.sells == []
        assert [p.symbol for p in decision.buys] == ["000001"]

    @pytest.mark.asyncio
    async def test_caps_picks_at_n_drop(self, engine):
        sells = [_make_candidate(f"60000{i}", score=0.1, rank=300 + i)
                 for i in range(5)]
        buys = [_make_candidate(f"60001{i}", score=0.95, rank=1 + i)
                for i in range(10)]
        sells_resp = [
            {"symbol": f"60000{i}", "reason": "卖", "evidence": ""}
            for i in range(5)
        ]
        buys_resp = [
            {"symbol": f"60001{i}", "reason": "买", "evidence": ""}
            for i in range(10)
        ]
        content = _ok_response(sells=sells_resp, buys=buys_resp)
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=sells, buy_candidates=buys,
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3,
            )
        assert len(decision.sells) == 3
        assert len(decision.buys) == 3

    @pytest.mark.asyncio
    async def test_skips_empty_reason(self, engine):
        sells = [_make_candidate("600519", score=0.1, rank=300)]
        buys = [_make_candidate("000001", score=0.95, rank=1)]
        content = _ok_response(
            sells=[{"symbol": "600519", "reason": "", "evidence": ""}],
            buys=[{"symbol": "000001", "reason": "OK", "evidence": ""}],
        )
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=sells, buy_candidates=buys,
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3,
            )
        assert decision.sells == []  # empty reason → rejected
        assert len(decision.buys) == 1

    @pytest.mark.asyncio
    async def test_dedups_same_symbol(self, engine):
        sells = [_make_candidate("600519", score=0.1, rank=300)]
        buys = [_make_candidate("000001", score=0.95, rank=1)]
        content = _ok_response(
            sells=[
                {"symbol": "600519", "reason": "first", "evidence": ""},
                {"symbol": "600519", "reason": "second (dup)", "evidence": ""},
            ],
            buys=[{"symbol": "000001", "reason": "OK", "evidence": ""}],
        )
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=sells, buy_candidates=buys,
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3,
            )
        assert len(decision.sells) == 1
        assert decision.sells[0].reason == "first"

    @pytest.mark.asyncio
    async def test_returns_empty_decision_when_llm_raises(self, engine):
        # All LLM attempts raise → client falls back to rule-based, but
        # rule-based output has no sells/buys keys, so we expect empty picks.
        sells = [_make_candidate("600519", score=0.1, rank=300)]
        buys = [_make_candidate("000001", score=0.95, rank=1)]
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   side_effect=Exception("API down")):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=sells, buy_candidates=buys,
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3,
            )
        assert decision.sells == []
        assert decision.buys == []
        # source comes back as llm_judge because the rule-based fallback is
        # still a valid (just unhelpful) response.
        assert decision.source in {"llm_judge", "llm_failed"}

    @pytest.mark.asyncio
    async def test_inner_timeout_returns_empty_failed_decision(self, engine):
        async def hanging(*a, **k):
            import asyncio
            raise asyncio.TimeoutError()

        engine._client.call = hanging
        decision = await engine.judge_rebalance(
            today="2025-03-10",
            sell_candidates=[_make_candidate("600519")],
            buy_candidates=[_make_candidate("000001")],
            telegraph=[], global_market=_GLOBAL_DF,
            top_n=25, n_drop=3,
        )
        assert decision.source == "llm_failed"
        assert decision.sells == []
        assert decision.buys == []

    @pytest.mark.asyncio
    async def test_empty_pools_short_circuits(self, engine):
        # When both pools are empty, judge_rebalance should not call the LLM
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   side_effect=AssertionError("LLM should not be called")):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=[], buy_candidates=[],
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3,
            )
        assert isinstance(decision, RebalanceDecision)
        assert decision.sells == []
        assert decision.buys == []

    @pytest.mark.asyncio
    async def test_missing_debate_yields_default_notes(self, engine):
        sells = [_make_candidate("600519", score=0.1, rank=300)]
        content = json.dumps({
            "sells": [{"symbol": "600519", "reason": "OK", "evidence": ""}],
            "buys": [],
        })
        with patch("litellm.acompletion", new_callable=AsyncMock,
                   return_value=_make_completion(content)):
            decision = await engine.judge_rebalance(
                today="2025-03-10",
                sell_candidates=sells, buy_candidates=[],
                telegraph=[], global_market=_GLOBAL_DF,
                top_n=25, n_drop=3,
            )
        assert isinstance(decision.sell_debate, DebateNotes)
        assert decision.sell_debate.bull == ""
        assert decision.buy_debate.bear == ""

    @pytest.mark.asyncio
    async def test_normalize_filters_factor_details(self, engine):
        # amount should be filtered (not in _KNOWN_FACTORS whitelist)
        # rsi_14, macd_hist should pass through
        sells = [_make_candidate("600519", score=0.1, rank=300)]
        captured_prompt: list[str] = []

        async def capture_call(system, user):
            captured_prompt.append(user)
            return {"sells": [], "buys": []}

        engine._client.call = capture_call
        await engine.judge_rebalance(
            today="2025-03-10",
            sell_candidates=sells, buy_candidates=[],
            telegraph=[], global_market=_GLOBAL_DF,
            top_n=25, n_drop=3,
        )
        assert len(captured_prompt) == 1
        prompt = captured_prompt[0]
        # whitelisted factor keys should appear in prompt; amount filtered out
        assert "RSI" in prompt or "rsi_14" in prompt
        # amount value rendered as 1e10 would show up as 10000000000 — check absence
        assert "10000000000" not in prompt
