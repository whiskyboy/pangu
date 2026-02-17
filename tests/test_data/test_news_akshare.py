"""Tests for AkShareNewsDataProvider — mocked AkShare calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.data.news import AkShareNewsDataProvider
from trading_agent.data.storage import Database
from trading_agent.models import Region

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Database:
    d = Database(":memory:")
    d.init_tables()
    return d


def _fake_telegraph_df(n: int = 3) -> pd.DataFrame:
    """Simulate ak.stock_info_global_cls() return with mixed domestic/global news."""
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    templates = [
        ("央行宣布降准50基点", "中国人民银行决定下调存款准备金率。"),
        ("美联储维持利率不变", "Fed holds rates steady amid inflation."),
        ("沪指收涨1.2%", "A股三大指数集体收涨。"),
        ("OPEC+宣布减产", "原油价格大幅上涨。"),
        ("宁德时代发布新品", "新能源电池技术突破。"),
        ("港股恒生指数大涨", "恒指收涨3%。"),
    ]
    rows = []
    for i in range(min(n, len(templates))):
        rows.append({
            "标题": templates[i][0],
            "内容": templates[i][1],
            "发布日期": today,
            "发布时间": f"10:{i:02d}:00",
        })
    return pd.DataFrame(rows)


def _fake_stock_news_df(n: int = 2) -> pd.DataFrame:
    """Simulate ak.stock_news_em() return."""
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for i in range(n):
        rows.append({
            "关键词": "600519",
            "新闻标题": f"茅台新闻{i}",
            "新闻内容": f"贵州茅台相关内容{i}。",
            "发布时间": f"{today} 09:{i:02d}:00",
            "文章来源": "东财快讯",
            "新闻链接": f"http://example.com/news/{i}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_latest_news
# ---------------------------------------------------------------------------


class TestGetLatestNews:
    @patch("akshare.stock_info_global_cls")
    def test_returns_all_news(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = _fake_telegraph_df(6)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_latest_news(limit=50)

        assert len(items) == 6
        assert items[0].title == "央行宣布降准50基点"
        mock_cls.assert_called_once_with(symbol="全部")

    @patch("akshare.stock_info_global_cls")
    def test_limit_caps_results(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = _fake_telegraph_df(6)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_latest_news(limit=2)
        assert len(items) == 2

    @patch("akshare.stock_info_global_cls")
    def test_empty_returns_empty_list(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = pd.DataFrame()
        provider = AkShareNewsDataProvider(request_interval=0)
        assert provider.get_latest_news() == []

    @patch("akshare.stock_info_global_cls")
    def test_exception_returns_empty_list(self, mock_cls: MagicMock) -> None:
        mock_cls.side_effect = ConnectionError("network error")
        provider = AkShareNewsDataProvider(request_interval=0)
        from trading_agent.data.market import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        assert provider.get_latest_news() == []

    @patch("akshare.stock_info_global_cls")
    def test_persists_all_to_storage(self, mock_cls: MagicMock, db: Database) -> None:
        """All news (domestic + global) should be persisted, not just filtered."""
        mock_cls.return_value = _fake_telegraph_df(6)
        provider = AkShareNewsDataProvider(storage=db, request_interval=0)
        provider.get_latest_news(limit=50)

        stored = db.load_recent_news(hours=24)
        assert len(stored) == 6  # all 6 persisted, not just 3 domestic

    @patch("akshare.stock_info_global_cls")
    def test_parses_timestamp(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = _fake_telegraph_df(1)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_latest_news(limit=1)
        from datetime import datetime
        assert items[0].timestamp.year == datetime.now().year
        assert items[0].timestamp.hour == 10


# ---------------------------------------------------------------------------
# get_stock_news
# ---------------------------------------------------------------------------


class TestGetStockNews:
    @patch("akshare.stock_news_em")
    def test_returns_stock_specific_news(self, mock_news: MagicMock) -> None:
        mock_news.return_value = _fake_stock_news_df(3)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_stock_news("600519", limit=2)

        assert len(items) == 2
        assert all(item.region == Region.DOMESTIC for item in items)
        assert all("600519" in item.symbols for item in items)
        assert items[0].title == "茅台新闻0"
        mock_news.assert_called_once_with(symbol="600519")

    @patch("akshare.stock_news_em")
    def test_empty_returns_empty_list(self, mock_news: MagicMock) -> None:
        mock_news.return_value = pd.DataFrame()
        provider = AkShareNewsDataProvider(request_interval=0)
        assert provider.get_stock_news("600519") == []

    @patch("akshare.stock_news_em")
    def test_exception_returns_empty_list(self, mock_news: MagicMock) -> None:
        mock_news.side_effect = ConnectionError("down")
        provider = AkShareNewsDataProvider(request_interval=0)
        from trading_agent.data.market import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        assert provider.get_stock_news("600519") == []

    @patch("akshare.stock_news_em")
    def test_persists_to_storage(self, mock_news: MagicMock, db: Database) -> None:
        mock_news.return_value = _fake_stock_news_df(2)
        provider = AkShareNewsDataProvider(storage=db, request_interval=0)
        items = provider.get_stock_news("600519", limit=2)

        assert len(items) == 2
        stored = db.load_recent_news(hours=24)
        assert len(stored) == 2

    @patch("akshare.stock_news_em")
    def test_parses_source(self, mock_news: MagicMock) -> None:
        mock_news.return_value = _fake_stock_news_df(1)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_stock_news("600519", limit=1)
        assert items[0].source == "东财快讯"


# ---------------------------------------------------------------------------
# get_global_news
# ---------------------------------------------------------------------------


class TestGetGlobalNews:
    @patch("akshare.stock_info_global_cls")
    def test_returns_same_feed_as_latest(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = _fake_telegraph_df(6)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_global_news(limit=30)

        # Same unfiltered feed as get_latest_news
        assert len(items) == 6
        assert items[0].title == "央行宣布降准50基点"
        mock_cls.assert_called_once_with(symbol="全部")

    @patch("akshare.stock_info_global_cls")
    def test_limit_caps_results(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = _fake_telegraph_df(6)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_global_news(limit=2)
        assert len(items) == 2

    @patch("akshare.stock_info_global_cls")
    def test_empty_returns_empty_list(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = pd.DataFrame()
        provider = AkShareNewsDataProvider(request_interval=0)
        assert provider.get_global_news() == []

    @patch("akshare.stock_info_global_cls")
    def test_exception_returns_empty_list(self, mock_cls: MagicMock) -> None:
        mock_cls.side_effect = ConnectionError("down")
        provider = AkShareNewsDataProvider(request_interval=0)
        from trading_agent.data.market import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        assert provider.get_global_news() == []

    @patch("akshare.stock_info_global_cls")
    def test_persists_all_to_storage(self, mock_cls: MagicMock, db: Database) -> None:
        mock_cls.return_value = _fake_telegraph_df(6)
        provider = AkShareNewsDataProvider(storage=db, request_interval=0)
        provider.get_global_news(limit=30)

        stored = db.load_recent_news(hours=24)
        assert len(stored) == 6


# ---------------------------------------------------------------------------
# Dedup across calls
# ---------------------------------------------------------------------------


class TestDedup:
    @patch("akshare.stock_info_global_cls")
    def test_same_news_not_duplicated(self, mock_cls: MagicMock, db: Database) -> None:
        """Calling get_latest_news twice with same data should not duplicate in DB."""
        mock_cls.return_value = _fake_telegraph_df(3)
        provider = AkShareNewsDataProvider(storage=db, request_interval=0)

        provider.get_latest_news(limit=50)
        provider.get_latest_news(limit=50)

        stored = db.load_recent_news(hours=24)
        assert len(stored) == 3  # not 6


# ---------------------------------------------------------------------------
# _parse_timestamp edge cases
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    def test_full_datetime(self) -> None:
        ts = AkShareNewsDataProvider._parse_timestamp("2026-02-16", "10:30:00")
        assert ts.year == 2026
        assert ts.hour == 10
        assert ts.minute == 30

    def test_date_only(self) -> None:
        ts = AkShareNewsDataProvider._parse_timestamp("2026-02-16", "")
        assert ts.year == 2026
        assert ts.hour == 0

    def test_invalid_falls_back_to_now(self) -> None:
        ts = AkShareNewsDataProvider._parse_timestamp("bad", "data")
        # Should be close to now
        from datetime import datetime
        assert (datetime.now() - ts).total_seconds() < 5


# ---------------------------------------------------------------------------
# get_announcements
# ---------------------------------------------------------------------------


def _fake_announcement_df(n: int = 3) -> pd.DataFrame:
    """Simulate ak.stock_zh_a_disclosure_report_cninfo() return."""
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for i in range(n):
        rows.append({
            "代码": "601899",
            "简称": "紫金矿业",
            "公告标题": f"紫金矿业关于测试公告{i}",
            "公告时间": today,
            "公告链接": f"http://www.cninfo.com.cn/announcement/{i}",
        })
    return pd.DataFrame(rows)


class TestGetAnnouncements:
    @patch("akshare.stock_zh_a_disclosure_report_cninfo")
    def test_returns_announcements(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _fake_announcement_df(3)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_announcements("601899", limit=20)

        assert len(items) == 3
        assert items[0].source == "巨潮"
        assert items[0].region == Region.DOMESTIC
        assert "601899" in items[0].symbols
        assert items[0].title == "紫金矿业关于测试公告0"
        assert items[0].content.startswith("http://www.cninfo.com.cn/")
        # All A-shares use market="沪深京"
        call_kwargs = mock_api.call_args
        assert call_kwargs[1]["market"] == "沪深京" or call_kwargs.kwargs["market"] == "沪深京"

    @patch("akshare.stock_zh_a_disclosure_report_cninfo")
    def test_limit_caps_results(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _fake_announcement_df(5)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_announcements("601899", limit=2)
        assert len(items) == 2

    @patch("akshare.stock_zh_a_disclosure_report_cninfo")
    def test_empty_returns_empty_list(self, mock_api: MagicMock) -> None:
        mock_api.return_value = pd.DataFrame()
        provider = AkShareNewsDataProvider(request_interval=0)
        assert provider.get_announcements("601899") == []

    @patch("akshare.stock_zh_a_disclosure_report_cninfo")
    def test_exception_returns_empty_list(self, mock_api: MagicMock) -> None:
        mock_api.side_effect = ConnectionError("network error")
        provider = AkShareNewsDataProvider(request_interval=0)
        from trading_agent.data.market import CircuitBreaker
        provider._circuit = CircuitBreaker(threshold=100, cooldown=0)
        assert provider.get_announcements("601899") == []

    @patch("akshare.stock_zh_a_disclosure_report_cninfo")
    def test_persists_to_storage(self, mock_api: MagicMock, db: Database) -> None:
        mock_api.return_value = _fake_announcement_df(2)
        provider = AkShareNewsDataProvider(storage=db, request_interval=0)
        items = provider.get_announcements("601899", limit=20)

        assert len(items) == 2
        stored = db.load_recent_news(hours=24 * 30)
        assert len(stored) == 2
        assert stored[0].source == "巨潮"

    @patch("akshare.stock_zh_a_disclosure_report_cninfo")
    def test_parses_timestamp(self, mock_api: MagicMock) -> None:
        mock_api.return_value = _fake_announcement_df(1)
        provider = AkShareNewsDataProvider(request_interval=0)
        items = provider.get_announcements("601899", limit=1)
        from datetime import datetime
        assert items[0].timestamp.year == datetime.now().year
        assert items[0].timestamp.month == datetime.now().month

    @patch("akshare.stock_zh_a_disclosure_report_cninfo")
    def test_dedup_across_calls(self, mock_api: MagicMock, db: Database) -> None:
        """Same announcements fetched twice should not duplicate in DB."""
        mock_api.return_value = _fake_announcement_df(2)
        provider = AkShareNewsDataProvider(storage=db, request_interval=0)

        provider.get_announcements("601899")
        provider.get_announcements("601899")

        stored = db.load_recent_news(hours=24 * 30)
        assert len(stored) == 2  # not 4
