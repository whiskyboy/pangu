"""AkShare NewsDataProvider — PRD §4.1.3."""

from __future__ import annotations

import logging
from datetime import timedelta

from pangu.tz import now as _now
from pangu.models import NewsCategory, NewsItem, Region
from pangu.utils import CircuitBreaker, ThrottleMixin, retry_call

logger = logging.getLogger(__name__)


class AkShareNewsDataProvider(ThrottleMixin):
    """Real news data backed by AkShare + optional SQLite persistence.

    All CLS telegraph news are fetched via a single
    ``ak.stock_info_global_cls(symbol="全部")`` call.

    API mapping:
    - get_latest_news  → CLS telegraph (all items)
    - get_stock_news   → ak.stock_news_em(symbol=...)

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
        self.__init_throttle__(request_interval)
        self._circuit = CircuitBreaker()

    # -- helpers --

    def _parse_telegraph(self, row) -> NewsItem:
        """Convert a row from stock_info_global_cls DataFrame to NewsItem."""
        from datetime import datetime

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
    def _parse_timestamp(date_str: str, time_str: str):
        """Parse date + time strings from AkShare into tz-aware datetime."""
        from datetime import datetime

        from pangu.tz import _get_tz

        combined = f"{date_str} {time_str}".strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(combined, fmt).replace(tzinfo=_get_tz())
            except ValueError:
                continue
        return _now()

    def _fetch_telegraph(self) -> list[NewsItem]:
        """Fetch all CLS telegraph news (single API call), classify region."""
        try:
            self._throttle()
            df = retry_call(
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
        from datetime import datetime

        try:
            self._throttle()
            df = retry_call(
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

    def get_announcements(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Fetch announcements for *symbol* from 巨潮 (cninfo).

        Uses ``ak.stock_zh_a_disclosure_report_cninfo`` with a 90-day
        lookback window.  Results are stored as :class:`NewsItem` with
        ``source="巨潮"`` and the announcement URL in *content*.
        """
        end_date = _now().strftime("%Y%m%d")
        start_date = (_now() - timedelta(days=90)).strftime("%Y%m%d")

        try:
            self._throttle()
            df = retry_call(
                lambda: self._ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=symbol,
                    market="沪深京",
                    start_date=start_date,
                    end_date=end_date,
                ),
                circuit=self._circuit,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "stock_zh_a_disclosure_report_cninfo(%s) failed",
                symbol,
                exc_info=True,
            )
            return []

        if df is None or df.empty:
            return []

        items: list[NewsItem] = []
        for _, row in df.head(limit).iterrows():
            title = str(row.get("公告标题", ""))
            url = str(row.get("公告链接", ""))
            pub_time = str(row.get("公告时间", ""))
            ts = self._parse_timestamp(pub_time, "")
            items.append(NewsItem(
                timestamp=ts,
                title=title,
                content=url,
                source="巨潮",
                region=Region.DOMESTIC,
                symbols=[symbol],
                category=NewsCategory.ANNOUNCEMENT,
            ))

        if self._storage is not None and items:
            self._storage.save_news_items(items)
        return items
