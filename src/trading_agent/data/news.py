"""NewsDataProvider Protocol — PRD §4.1.3 / §6."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from trading_agent.models import NewsItem, Region


class NewsDataProvider(Protocol):
    """Interface for domestic and international financial news."""

    def get_latest_news(self, limit: int = 50) -> list[NewsItem]:
        """Return latest domestic financial news."""
        ...

    def get_stock_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Return news related to a specific stock."""
        ...

    def get_global_news(self, limit: int = 30) -> list[NewsItem]:
        """Return international financial news."""
        ...


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 16, 10, 0, 0)


class FakeNewsDataProvider:
    """Deterministic fake news for testing."""

    def get_latest_news(self, limit: int = 50) -> list[NewsItem]:
        items = [
            NewsItem(
                timestamp=_NOW - timedelta(minutes=10),
                title="央行宣布降准50基点",
                content="中国人民银行决定下调金融机构存款准备金率0.5个百分点。",
                source="财联社",
                region=Region.DOMESTIC,
                symbols=[],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(minutes=30),
                title="贵州茅台发布业绩预增公告",
                content="贵州茅台预计2025年净利润同比增长15%-20%。",
                source="东方财富",
                region=Region.DOMESTIC,
                symbols=["600519"],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(hours=1),
                title="新能源汽车销量创新高",
                content="2025年新能源汽车销量突破1000万辆。",
                source="财联社",
                region=Region.DOMESTIC,
                symbols=["300750"],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(hours=2),
                title="银行板块集体上涨",
                content="多家银行股涨停，市场情绪回暖。",
                source="东方财富",
                region=Region.DOMESTIC,
                symbols=["000001", "601318"],
            ),
        ]
        return items[:limit]

    def get_stock_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        all_news = self.get_latest_news(limit=50)
        return [n for n in all_news if symbol in n.symbols][:limit]

    def get_global_news(self, limit: int = 30) -> list[NewsItem]:
        items = [
            NewsItem(
                timestamp=_NOW - timedelta(minutes=15),
                title="Fed holds rates steady amid inflation concerns",
                content="The Federal Reserve kept interest rates unchanged at 5.25%-5.50%.",
                source="Reuters",
                region=Region.GLOBAL,
                symbols=[],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(hours=1),
                title="Gold prices surge to record high",
                content="Spot gold hit $2,350/oz driven by safe-haven demand.",
                source="Bloomberg",
                region=Region.GLOBAL,
                symbols=[],
            ),
            NewsItem(
                timestamp=_NOW - timedelta(hours=3),
                title="Oil prices rise on OPEC+ production cuts",
                content="Brent crude climbed to $78/barrel after OPEC+ confirmed supply reductions.",
                source="Reuters",
                region=Region.GLOBAL,
                symbols=[],
            ),
        ]
        return items[:limit]

