"""NewsDataProvider Protocol — PRD §4.1.3 / §6."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Protocol

from trading_agent.models import NewsItem, Region

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# AkShare real implementation — PRD §4.1.3
# ---------------------------------------------------------------------------

from trading_agent.data.market import CircuitBreaker, _retry_call  # noqa: E402


class AkShareNewsDataProvider:
    """Real news data backed by AkShare + optional SQLite persistence.

    All CLS telegraph news are fetched via a single
    ``ak.stock_info_global_cls(symbol="全部")`` call.  No region
    classification is performed — both ``get_latest_news`` and
    ``get_global_news`` return the same unfiltered feed.  Region
    classification will be revisited after migration to Tushare Pro.

    API mapping:
    - get_latest_news  → CLS telegraph (all items)
    - get_stock_news   → ak.stock_news_em(symbol=...)
    - get_global_news  → CLS telegraph (same as get_latest_news)

    Parameters
    ----------
    storage : Database | None
        If provided, news items are persisted and deduplicated in SQLite.
    request_interval : float
        Minimum seconds between API calls (default 0.5).
    """

    def __init__(self, storage=None, request_interval: float = 0.5) -> None:
        import akshare  # lazy import for test mocking

        self._ak = akshare
        self._storage = storage
        self._interval = request_interval
        self._last_call: float = 0.0
        self._circuit = CircuitBreaker()
        self._throttle_lock = threading.Lock()

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.monotonic()

    # -- helpers --

    def _parse_telegraph(self, row) -> NewsItem:
        """Convert a row from stock_info_global_cls DataFrame to NewsItem."""
        title = str(row.get("标题", ""))
        content = str(row.get("内容", ""))
        pub_date = str(row.get("发布日期", ""))
        pub_time = str(row.get("发布时间", ""))

        ts = self._parse_timestamp(pub_date, pub_time)

        return NewsItem(
            timestamp=ts,
            title=title,
            content=content,
            source="财联社",
            region=Region.DOMESTIC,
            symbols=[],
        )

    @staticmethod
    def _parse_timestamp(date_str: str, time_str: str) -> datetime:
        """Parse date + time strings from AkShare into datetime."""
        combined = f"{date_str} {time_str}".strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue
        return datetime.now()

    def _fetch_telegraph(self) -> list[NewsItem]:
        """Fetch all CLS telegraph news (single API call), classify region."""
        try:
            self._throttle()
            df = _retry_call(
                lambda: self._ak.stock_info_global_cls(symbol="全部"),
                circuit=self._circuit,
            )
        except Exception:  # noqa: BLE001
            logger.warning("stock_info_global_cls failed", exc_info=True)
            return []

        if df is None or df.empty:
            return []

        items: list[NewsItem] = []
        for _, row in df.iterrows():
            items.append(self._parse_telegraph(row))

        if self._storage is not None and items:
            self._storage.save_news_items(items)
        return items

    # -- Protocol methods --

    def get_latest_news(self, limit: int = 50) -> list[NewsItem]:
        """Fetch latest financial news via CLS telegraph."""
        all_items = self._fetch_telegraph()
        return all_items[:limit]

    def get_stock_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Fetch news for a specific stock via stock_news_em."""
        try:
            self._throttle()
            df = _retry_call(
                lambda: self._ak.stock_news_em(symbol=symbol),
                circuit=self._circuit,
            )
        except Exception:  # noqa: BLE001
            logger.warning("stock_news_em(%s) failed", symbol, exc_info=True)
            return []

        if df is None or df.empty:
            return []

        items: list[NewsItem] = []
        for _, row in df.head(limit).iterrows():
            title = str(row.get("新闻标题", ""))
            content = str(row.get("新闻内容", ""))
            pub_time = str(row.get("发布时间", ""))
            source = str(row.get("文章来源", "东方财富"))

            ts = self._parse_timestamp(pub_time, "")

            items.append(NewsItem(
                timestamp=ts,
                title=title,
                content=content,
                source=source,
                region=Region.DOMESTIC,
                symbols=[symbol],
            ))

        if self._storage is not None and items:
            self._storage.save_news_items(items)
        return items

    def get_global_news(self, limit: int = 30) -> list[NewsItem]:
        """Fetch financial news via CLS telegraph (same feed as get_latest_news)."""
        all_items = self._fetch_telegraph()
        return all_items[:limit]
