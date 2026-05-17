"""Tests for LLM rebalance prompt templates."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from pangu.models import NewsCategory, NewsItem, Region
from pangu.strategy.llm.prompts import (
    _REBAL_ANNOUNCEMENTS,
    _REBAL_STOCK_NEWS,
    _REBAL_TELEGRAPH,
    build_rebalance_prompt,
    build_rebalance_system_prompt,
)


def _news(title: str, content: str = "正文", category: NewsCategory = NewsCategory.NEWS) -> NewsItem:
    return NewsItem(
        timestamp=datetime(2026, 2, 19, 8, 0),
        title=title,
        content=content,
        source="财联社",
        region=Region.DOMESTIC,
        symbols=[],
        category=category,
    )


def _candidate(
    symbol: str,
    *,
    name: str | None = None,
    ml_score: float = 0.7,
    ml_rank: int = 10,
    prev_ml_rank: int | None = None,
    rank_delta: int | None = None,
    factor_details: dict[str, float] | None = None,
    stock_news: list[NewsItem] | None = None,
    announcements: list[NewsItem] | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "name": name or symbol,
        "ml_score": ml_score,
        "ml_rank": ml_rank,
        "prev_ml_rank": prev_ml_rank,
        "rank_delta": rank_delta,
        "factor_details": factor_details or {},
        "stock_news": stock_news or [],
        "announcements": announcements or [],
    }


# ---------------------------------------------------------------------------
# build_rebalance_system_prompt
# ---------------------------------------------------------------------------


class TestRebalanceSystemPrompt:
    def test_pool_sizes_interpolated(self) -> None:
        prompt = build_rebalance_system_prompt(
            top_n=25,
            n_drop=3,
            sell_pool_size=5,
            buy_pool_size=10,
        )
        assert "25" in prompt
        assert "n_drop=3" in prompt
        assert "5 只" in prompt or "持仓中 ML 评分最差的 5" in prompt
        assert "10 只" in prompt or "非持仓中 ML 评分最好的 10" in prompt

    def test_contains_three_roles(self) -> None:
        prompt = build_rebalance_system_prompt(
            top_n=25,
            n_drop=3,
            sell_pool_size=5,
            buy_pool_size=10,
        )
        assert "牛方" in prompt
        assert "熊方" in prompt
        assert "裁判" in prompt

    def test_contains_json_schema_keywords(self) -> None:
        prompt = build_rebalance_system_prompt(
            top_n=25,
            n_drop=3,
            sell_pool_size=5,
            buy_pool_size=10,
        )
        assert "sell_debate" in prompt
        assert "buy_debate" in prompt
        assert '"sells"' in prompt
        assert '"buys"' in prompt


# ---------------------------------------------------------------------------
# build_rebalance_prompt
# ---------------------------------------------------------------------------


class TestRebalancePrompt:
    def test_empty_pools(self) -> None:
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[],
            buy_candidates=[],
            telegraph=[],
            global_market=pd.DataFrame(),
            top_n=25,
            n_drop=3,
        )
        assert "2026-02-19" in prompt
        assert "SELL 候选池" in prompt
        assert "BUY 候选池" in prompt
        assert "（空）" in prompt
        # Universe size 0 should not crash; rank not displayed as "N/0"
        assert "全球市场" in prompt or "市场快讯" in prompt

    def test_candidate_rendering(self) -> None:
        sell = _candidate(
            "600519",
            name="贵州茅台",
            ml_score=0.42,
            ml_rank=120,
            prev_ml_rank=80,
            rank_delta=40,
            factor_details={"rsi_14": 35.0, "pe_ttm": 25.0},
            stock_news=[_news("利空消息：被监管处罚")],
            announcements=[_news("年报", category=NewsCategory.ANNOUNCEMENT)],
        )
        buy = _candidate(
            "601899",
            name="紫金矿业",
            ml_score=0.92,
            ml_rank=3,
        )
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[sell],
            buy_candidates=[buy],
            telegraph=[_news("央行降准")],
            global_market=pd.DataFrame(
                [
                    {"name": "标普500", "symbol": "SPX", "close": 5200.0, "change_pct": 0.35},
                ]
            ),
            top_n=25,
            n_drop=3,
            universe_size=500,
        )
        # Sell candidate
        assert "600519" in prompt
        assert "贵州茅台" in prompt
        assert "0.4200" in prompt
        assert "120/500" in prompt
        # Rank history shown for sell side
        assert "上次 ML 排名" in prompt
        assert "RSI(14)" in prompt
        assert "利空消息：被监管处罚" in prompt
        # Buy candidate (no rank history)
        assert "601899" in prompt
        assert "0.9200" in prompt
        # Buy section does not include "上次 ML 排名" for first candidate
        buy_idx = prompt.index("601899")
        sell_section = prompt[:buy_idx]
        buy_section = prompt[buy_idx:]
        assert "上次 ML 排名" in sell_section
        assert "上次 ML 排名" not in buy_section
        # Shared sections
        assert "央行降准" in prompt
        assert "标普500" in prompt
        assert "+0.35%" in prompt

    def test_rank_history_first_entry(self) -> None:
        sell = _candidate("000001", ml_rank=50)
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[sell],
            buy_candidates=[],
            telegraph=[],
            global_market=pd.DataFrame(),
            top_n=25,
            n_drop=3,
        )
        assert "首次进入持仓" in prompt

    def test_news_truncation(self) -> None:
        many_news = [_news(f"新闻{i}") for i in range(_REBAL_STOCK_NEWS + 3)]
        cand = _candidate("000001", stock_news=many_news)
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[cand],
            buy_candidates=[],
            telegraph=[],
            global_market=pd.DataFrame(),
            top_n=25,
            n_drop=3,
        )
        assert f"共 {_REBAL_STOCK_NEWS + 3} 条" in prompt
        assert f"前 {_REBAL_STOCK_NEWS} 条" in prompt

    def test_announcement_truncation(self) -> None:
        many = [_news(f"公告{i}", category=NewsCategory.ANNOUNCEMENT) for i in range(_REBAL_ANNOUNCEMENTS + 2)]
        cand = _candidate("000001", announcements=many)
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[cand],
            buy_candidates=[],
            telegraph=[],
            global_market=pd.DataFrame(),
            top_n=25,
            n_drop=3,
        )
        assert f"共 {_REBAL_ANNOUNCEMENTS + 2} 条" in prompt

    def test_telegraph_truncation(self) -> None:
        many = [_news(f"快讯{i}") for i in range(_REBAL_TELEGRAPH + 2)]
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[],
            buy_candidates=[],
            telegraph=many,
            global_market=pd.DataFrame(),
            top_n=25,
            n_drop=3,
        )
        assert f"共 {_REBAL_TELEGRAPH + 2} 条" in prompt

    def test_nan_factor_shows_missing(self) -> None:
        cand = _candidate(
            "000001",
            factor_details={"rsi_14": 55.0, "macd_hist": float("nan")},
        )
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[cand],
            buy_candidates=[],
            telegraph=[],
            global_market=pd.DataFrame(),
            top_n=25,
            n_drop=3,
        )
        assert "RSI(14): 55.0000" in prompt
        assert "MACD" in prompt
        assert "数据缺失" in prompt

    def test_global_market_empty(self) -> None:
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[],
            buy_candidates=[],
            telegraph=[],
            global_market=pd.DataFrame(),
            top_n=25,
            n_drop=3,
        )
        # Empty DF should render gracefully (either "无" or omit the items)
        assert "全球市场" in prompt or "市场快讯" in prompt

    def test_negative_change_pct(self) -> None:
        df = pd.DataFrame(
            [
                {"name": "恒生指数", "symbol": "HSI", "close": 19500.0, "change_pct": -2.30},
            ]
        )
        prompt = build_rebalance_prompt(
            today="2026-02-19",
            sell_candidates=[],
            buy_candidates=[],
            telegraph=[],
            global_market=df,
            top_n=25,
            n_drop=3,
        )
        assert "-2.30%" in prompt
        assert "+-" not in prompt
