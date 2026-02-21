"""Tests for LLM prompt templates — M4.3."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from pangu.models import NewsCategory, NewsItem, Region
from pangu.strategy.llm.prompts import (
    LLM_OUTPUT_SCHEMA,
    TRADING_JUDGE_SYSTEM_PROMPT,
    _MAX_ANNOUNCEMENTS,
    _MAX_STOCK_NEWS,
    _MAX_TELEGRAPH,
    build_stock_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_news(
    title: str = "测试新闻",
    content: str = "新闻正文内容",
    source: str = "财联社",
    category: NewsCategory = NewsCategory.NEWS,
    symbols: list[str] | None = None,
) -> NewsItem:
    return NewsItem(
        timestamp=datetime(2026, 2, 19, 8, 0),
        title=title,
        content=content,
        source=source,
        region=Region.DOMESTIC,
        symbols=symbols or [],
        category=category,
    )


@pytest.fixture()
def factor_details() -> dict[str, float]:
    return {
        "rsi_14": 55.0,
        "macd_hist": 0.12,
        "pe_ttm": 15.3,
        "pb": 1.8,
        "roe_ttm": 0.12,
    }


@pytest.fixture()
def global_market_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "标普500", "symbol": "SPX", "close": 5200.0, "change_pct": 0.35},
        {"name": "恒生指数", "symbol": "HSI", "close": 20100.0, "change_pct": -0.50},
        {"name": "COMEX黄金", "symbol": "GC", "close": 2050.0, "change_pct": 0.10},
    ])


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_contains_role_definition(self):
        assert "投资分析师" in TRADING_JUDGE_SYSTEM_PROMPT

    def test_contains_three_perspectives(self):
        assert "牛方" in TRADING_JUDGE_SYSTEM_PROMPT
        assert "熊方" in TRADING_JUDGE_SYSTEM_PROMPT
        assert "裁判" in TRADING_JUDGE_SYSTEM_PROMPT

    def test_contains_json_output_requirement(self):
        assert "JSON" in TRADING_JUDGE_SYSTEM_PROMPT
        assert '"action"' in TRADING_JUDGE_SYSTEM_PROMPT
        assert '"confidence"' in TRADING_JUDGE_SYSTEM_PROMPT
        assert "BUY" in TRADING_JUDGE_SYSTEM_PROMPT
        assert "SELL" in TRADING_JUDGE_SYSTEM_PROMPT
        assert "HOLD" in TRADING_JUDGE_SYSTEM_PROMPT

    def test_contains_few_shot_example(self):
        assert "bull_reason" in TRADING_JUDGE_SYSTEM_PROMPT
        assert "judge_conclusion" in TRADING_JUDGE_SYSTEM_PROMPT

    def test_contains_input_description(self):
        assert "因子数据" in TRADING_JUDGE_SYSTEM_PROMPT
        assert "全球市场" in TRADING_JUDGE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_required_fields(self):
        required = {
            "action", "confidence", "bull_reason", "bear_reason",
            "judge_conclusion", "short_term_outlook", "mid_term_outlook",
        }
        assert required == set(LLM_OUTPUT_SCHEMA.keys())


# ---------------------------------------------------------------------------
# build_stock_prompt tests
# ---------------------------------------------------------------------------

class TestBuildStockPrompt:
    def test_full_data(self, factor_details, global_market_df):
        news = [_make_news(title="利好消息")]
        announcements = [_make_news(title="年报公告", category=NewsCategory.ANNOUNCEMENT)]
        telegraph = [_make_news(title="央行降准")]

        prompt = build_stock_prompt(
            symbol="600519",
            name="贵州茅台",
            factor_score=0.82,
            factor_rank=2,
            factor_details=factor_details,
            stock_news=news,
            announcements=announcements,
            telegraph=telegraph,
            global_market=global_market_df,
        )

        # Header
        assert "600519" in prompt
        assert "贵州茅台" in prompt
        # Factor section
        assert "0.8200" in prompt
        assert "第 2 名" in prompt
        assert "RSI(14)" in prompt
        assert "PE(TTM)" in prompt
        # News sections
        assert "利好消息" in prompt
        assert "年报公告" in prompt
        assert "央行降准" in prompt
        # Global market
        assert "标普500" in prompt
        assert "+0.35%" in prompt

    def test_empty_news(self, factor_details, global_market_df):
        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=global_market_df,
        )

        assert "个股新闻" in prompt
        assert "公司公告" in prompt
        assert "市场快讯" in prompt
        # Each empty section shows "无"
        lines = prompt.split("\n")
        news_idx = next(i for i, l in enumerate(lines) if "个股新闻" in l)
        assert lines[news_idx + 1] == "无"

    def test_empty_global_market(self, factor_details):
        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=pd.DataFrame(),
        )

        assert "全球市场" in prompt
        lines = prompt.split("\n")
        global_idx = next(i for i, l in enumerate(lines) if "全球市场" in l)
        assert lines[global_idx + 1] == "无"

    def test_none_global_market(self, factor_details):
        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=None,
        )

        assert "全球市场" in prompt
        assert "无" in prompt

    def test_news_truncation(self, factor_details, global_market_df):
        many_news = [_make_news(title=f"新闻{i}") for i in range(_MAX_STOCK_NEWS + 5)]

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=many_news,
            announcements=[],
            telegraph=[],
            global_market=global_market_df,
        )

        # Should show truncation notice
        assert f"共 {_MAX_STOCK_NEWS + 5} 条" in prompt
        assert f"前 {_MAX_STOCK_NEWS} 条" in prompt
        # First items shown, excess not shown
        assert "新闻0" in prompt
        assert f"新闻{_MAX_STOCK_NEWS - 1}" in prompt

    def test_announcement_truncation(self, factor_details, global_market_df):
        many = [_make_news(title=f"公告{i}", category=NewsCategory.ANNOUNCEMENT)
                for i in range(_MAX_ANNOUNCEMENTS + 3)]

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=[],
            announcements=many,
            telegraph=[],
            global_market=global_market_df,
        )

        assert f"共 {_MAX_ANNOUNCEMENTS + 3} 条" in prompt
        assert f"前 {_MAX_ANNOUNCEMENTS} 条" in prompt

    def test_telegraph_truncation(self, factor_details, global_market_df):
        many = [_make_news(title=f"快讯{i}") for i in range(_MAX_TELEGRAPH + 2)]

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=[],
            announcements=[],
            telegraph=many,
            global_market=global_market_df,
        )

        assert f"共 {_MAX_TELEGRAPH + 2} 条" in prompt
        assert f"前 {_MAX_TELEGRAPH} 条" in prompt

    def test_factor_details_formatting(self, global_market_df):
        details = {"rsi_14": 65.123, "macd_hist": -0.05, "volume_ratio": 1.5}

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.65,
            factor_rank=5,
            factor_details=details,
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=global_market_df,
        )

        assert "RSI(14): 65.1230" in prompt
        assert "MACD 柱: -0.0500" in prompt
        assert "量比: 1.5000" in prompt

    def test_empty_factor_details(self, global_market_df):
        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details={},
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=global_market_df,
        )

        assert "综合得分: 0.5000" in prompt
        assert "因子细项" not in prompt

    def test_content_snippet_in_news(self, factor_details, global_market_df):
        long_content = "这是一段很长的新闻内容" * 20  # >100 chars
        news = [_make_news(title="标题", content=long_content)]

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=news,
            announcements=[],
            telegraph=[],
            global_market=global_market_df,
        )

        # Snippet truncated with ellipsis
        assert "…" in prompt
        # Title still present
        assert "标题" in prompt

    def test_negative_change_pct(self, factor_details):
        df = pd.DataFrame([
            {"name": "恒生指数", "close": 19500.0, "change_pct": -2.30},
        ])

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=df,
        )

        assert "-2.30%" in prompt
        # No '+' prefix for negative
        assert "+-" not in prompt

    def test_nan_factor_details_skipped(self, global_market_df):
        details = {"rsi_14": 55.0, "macd_hist": float("nan"), "pe_ttm": 15.0}

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=details,
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=global_market_df,
        )

        assert "RSI(14): 55.0000" in prompt
        assert "PE(TTM): 15.0000" in prompt
        assert "MACD 柱: 数据缺失" in prompt

    def test_nan_global_market_shows_missing(self, factor_details):
        df = pd.DataFrame([
            {"name": "标普500", "close": 5200.0, "change_pct": float("nan")},
            {"name": "恒生指数", "close": 20100.0, "change_pct": -0.50},
        ])

        prompt = build_stock_prompt(
            symbol="000001",
            name="平安银行",
            factor_score=0.5,
            factor_rank=10,
            factor_details=factor_details,
            stock_news=[],
            announcements=[],
            telegraph=[],
            global_market=df,
        )

        assert "标普500: 数据缺失" in prompt
        assert "-0.50%" in prompt
