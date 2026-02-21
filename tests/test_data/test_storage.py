"""Tests for the SQLite storage layer (Database class)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from pangu.data.storage import Database
from pangu.models import Action, NewsItem, Region, SignalStatus, TradeSignal


@pytest.fixture()
def db() -> Database:
    """Return an in-memory Database with tables initialised."""
    d = Database(":memory:")
    d.init_tables()
    return d


# ------------------------------------------------------------------
# init_tables
# ------------------------------------------------------------------


class TestInitTables:
    def test_tables_created(self, db: Database) -> None:
        """All seven expected tables must exist after init_tables()."""
        rows = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in rows}
        expected = {
            "daily_bars",
            "news_items",
            "trade_signals",
            "backtest_results",
            "fundamentals",
            "data_sync_log",
            "trading_calendar",
            "index_constituents",
        }
        assert expected.issubset(names)

    def test_idempotent(self, db: Database) -> None:
        """Calling init_tables() twice must not raise."""
        db.init_tables()


# ------------------------------------------------------------------
# daily_bars
# ------------------------------------------------------------------


class TestDailyBars:
    @staticmethod
    def _sample_df() -> pd.DataFrame:
        return pd.DataFrame({
            "date": ["2026-01-02", "2026-01-03"],
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.5, 10.0],
            "close": [10.8, 11.2],
            "volume": [100000, 120000],
            "amount": [1_080_000, 1_344_000],
            "adj_factor": [1.0, 1.0],
        })

    def test_save_and_load(self, db: Database) -> None:
        df = self._sample_df()
        n = db.save_daily_bars("600519", df)
        assert n == 2

        loaded = db.load_daily_bars("600519", "2026-01-01", "2026-01-31")
        assert len(loaded) == 2
        assert loaded.iloc[0]["close"] == pytest.approx(10.8)

    def test_empty_df(self, db: Database) -> None:
        assert db.save_daily_bars("600519", pd.DataFrame()) == 0

    def test_upsert(self, db: Database) -> None:
        """INSERT OR REPLACE should overwrite existing rows."""
        df = self._sample_df()
        db.save_daily_bars("600519", df)

        updated = pd.DataFrame({
            "date": ["2026-01-02"],
            "open": [10.0],
            "high": [12.0],
            "low": [9.5],
            "close": [11.9],
            "volume": [150000],
        })
        db.save_daily_bars("600519", updated)

        loaded = db.load_daily_bars("600519", "2026-01-02", "2026-01-02")
        assert len(loaded) == 1
        assert loaded.iloc[0]["close"] == pytest.approx(11.9)

    def test_load_date_range_filter(self, db: Database) -> None:
        db.save_daily_bars("600519", self._sample_df())
        loaded = db.load_daily_bars("600519", "2026-01-03", "2026-01-03")
        assert len(loaded) == 1

    def test_load_nonexistent_symbol(self, db: Database) -> None:
        loaded = db.load_daily_bars("999999", "2026-01-01", "2026-12-31")
        assert loaded.empty

    def test_default_adj_factor(self, db: Database) -> None:
        """Missing adj_factor column should default to 1.0."""
        df = pd.DataFrame({
            "date": ["2026-01-02"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.8],
            "volume": [100000],
        })
        db.save_daily_bars("600519", df)
        loaded = db.load_daily_bars("600519", "2026-01-01", "2026-01-31")
        assert loaded.iloc[0]["adj_factor"] == pytest.approx(1.0)


# ------------------------------------------------------------------
# data_sync_log
# ------------------------------------------------------------------


class TestSyncLog:
    def test_none_initially(self, db: Database) -> None:
        assert db.get_last_sync_date("600519", "daily_bars") is None

    def test_update_and_get(self, db: Database) -> None:
        db.update_sync_log("600519", "daily_bars", "ok", "akshare", last_date="2026-01-03")
        assert db.get_last_sync_date("600519", "daily_bars") == "2026-01-03"

    def test_upsert(self, db: Database) -> None:
        db.update_sync_log("600519", "daily_bars", "ok", "akshare", last_date="2026-01-03")
        db.update_sync_log("600519", "daily_bars", "ok", "akshare", last_date="2026-01-04")
        assert db.get_last_sync_date("600519", "daily_bars") == "2026-01-04"

    def test_error_msg(self, db: Database) -> None:
        db.update_sync_log(
            "600519", "daily_bars", "failed", "akshare",
            error_msg="connection timeout",
        )
        row = db._conn.execute(
            "SELECT status, error_msg FROM data_sync_log "
            "WHERE symbol = ? AND data_type = ?",
            ("600519", "daily_bars"),
        ).fetchone()
        assert row[0] == "failed"
        assert row[1] == "connection timeout"


# ------------------------------------------------------------------
# news_items
# ------------------------------------------------------------------


def _make_news(title: str = "Test News", source: str = "cls") -> NewsItem:
    from pangu.tz import now as _now
    return NewsItem(
        timestamp=_now(),
        title=title,
        content="Content",
        source=source,
        region=Region.DOMESTIC,
        symbols=["600519"],
        sentiment=0.5,
    )


class TestNewsItems:
    def test_save_and_load(self, db: Database) -> None:
        items = [_make_news("News A"), _make_news("News B")]
        inserted = db.save_news_items(items)
        assert inserted == 2

        loaded = db.load_recent_news(hours=1)
        assert len(loaded) == 2

    def test_dedup_same_source_title(self, db: Database) -> None:
        """Duplicate (source, title) should be skipped."""
        items = [_make_news("Dup News"), _make_news("Dup News")]
        inserted = db.save_news_items(items)
        assert inserted == 1

    def test_different_source_same_title(self, db: Database) -> None:
        """Same title from different sources should both insert."""
        items = [_make_news("Same Title", "cls"), _make_news("Same Title", "eastmoney")]
        inserted = db.save_news_items(items)
        assert inserted == 2

    def test_load_respects_hours_cutoff(self, db: Database) -> None:
        """Items older than cutoff should not be returned."""
        old = NewsItem(
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            title="Old News",
            content="Old",
            source="cls",
            region=Region.DOMESTIC,
        )
        db.save_news_items([old])
        loaded = db.load_recent_news(hours=24)
        assert len(loaded) == 0

    def test_roundtrip_fields(self, db: Database) -> None:
        """All fields should survive a save→load round-trip."""
        item = _make_news("Field Check")
        db.save_news_items([item])
        loaded = db.load_recent_news(hours=1)
        assert len(loaded) == 1
        got = loaded[0]
        assert got.title == "Field Check"
        assert got.source == "cls"
        assert got.region == Region.DOMESTIC
        assert got.symbols == ["600519"]
        assert got.sentiment == pytest.approx(0.5)

    def test_cleanup_old_news_deletes_expired(self, db: Database) -> None:
        """Items older than retention window should be deleted."""
        old = NewsItem(
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            title="Old News",
            content="Old",
            source="cls",
            region=Region.DOMESTIC,
        )
        recent = _make_news("Recent News")
        db.save_news_items([old, recent])

        deleted = db.cleanup_old_news(days=30)
        assert deleted == 1

        remaining = db.load_recent_news(hours=24)
        assert len(remaining) == 1
        assert remaining[0].title == "Recent News"

    def test_cleanup_old_news_keeps_recent(self, db: Database) -> None:
        """All items within retention window should be kept."""
        items = [_make_news("News A"), _make_news("News B")]
        db.save_news_items(items)

        deleted = db.cleanup_old_news(days=30)
        assert deleted == 0

    def test_cleanup_old_news_returns_zero_when_empty(self, db: Database) -> None:
        """Cleanup on empty table should return 0."""
        assert db.cleanup_old_news(days=30) == 0


# ------------------------------------------------------------------
# trade_signals
# ------------------------------------------------------------------


def _make_signal(
    symbol: str = "600519", action: Action = Action.BUY
) -> TradeSignal:
    return TradeSignal(
        timestamp=datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc),
        symbol=symbol,
        name="贵州茅台",
        action=action,
        signal_status=SignalStatus.NEW_ENTRY,
        days_in_top_n=1,
        price=1800.0,
        confidence=0.85,
        source="factor",
        reason="Top-N entry",
        stop_loss=1700.0,
        take_profit=2000.0,
        metadata={"rank": 1},
    )


class TestTradeSignals:
    def test_save_and_load(self, db: Database) -> None:
        sig = _make_signal()
        row_id = db.save_trade_signal(sig)
        assert row_id >= 1

        loaded = db.load_signals("2026-02-16")
        assert len(loaded) == 1
        got = loaded[0]
        assert got.symbol == "600519"
        assert got.action == Action.BUY
        assert got.price == pytest.approx(1800.0)
        assert got.metadata == {"rank": 1}

    def test_load_wrong_date(self, db: Database) -> None:
        db.save_trade_signal(_make_signal())
        loaded = db.load_signals("2026-02-17")
        assert len(loaded) == 0

    def test_multiple_signals(self, db: Database) -> None:
        db.save_trade_signal(_make_signal("600519", Action.BUY))
        db.save_trade_signal(_make_signal("000858", Action.SELL))
        loaded = db.load_signals("2026-02-16")
        assert len(loaded) == 2

    def test_all_fields_roundtrip(self, db: Database) -> None:
        """signal_status, days_in_top_n, factor_score, prev_factor_score must survive."""
        sig = TradeSignal(
            timestamp=datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc),
            symbol="600519",
            name="贵州茅台",
            action=Action.BUY,
            signal_status=SignalStatus.SUSTAINED,
            days_in_top_n=5,
            price=1800.0,
            confidence=0.85,
            source="factor",
            reason="Sustained in Top-N",
            factor_score=0.9,
            prev_factor_score=0.8,
        )
        db.save_trade_signal(sig)
        loaded = db.load_signals("2026-02-16")
        got = loaded[0]
        assert got.signal_status == SignalStatus.SUSTAINED
        assert got.days_in_top_n == 5
        assert got.factor_score == pytest.approx(0.9)
        assert got.prev_factor_score == pytest.approx(0.8)

    def test_dedup_same_symbol_action_date(self, db: Database) -> None:
        """Same (symbol, action, date) → INSERT OR REPLACE, only 1 row."""
        sig1 = _make_signal("600519", Action.BUY)
        sig2 = _make_signal("600519", Action.BUY)
        db.save_trade_signal(sig1)
        db.save_trade_signal(sig2)
        loaded = db.load_signals("2026-02-16")
        assert len(loaded) == 1

    def test_dedup_allows_different_action(self, db: Database) -> None:
        """Same symbol+date but different action → 2 rows."""
        db.save_trade_signal(_make_signal("600519", Action.BUY))
        db.save_trade_signal(_make_signal("600519", Action.SELL))
        loaded = db.load_signals("2026-02-16")
        assert len(loaded) == 2

    def test_dedup_allows_different_date(self, db: Database) -> None:
        """Same symbol+action but different date → 2 rows."""
        sig1 = _make_signal("600519", Action.BUY)
        sig2 = TradeSignal(
            timestamp=datetime(2026, 2, 17, 10, 0, 0, tzinfo=timezone.utc),
            symbol="600519", name="贵州茅台", action=Action.BUY,
            signal_status=SignalStatus.NEW_ENTRY, days_in_top_n=0,
            price=1800.0, confidence=0.85, source="factor", reason="test",
        )
        db.save_trade_signal(sig1)
        db.save_trade_signal(sig2)
        all_signals = db.load_signals("2026-02-16") + db.load_signals("2026-02-17")
        assert len(all_signals) == 2


# ------------------------------------------------------------------
# trading_calendar
# ------------------------------------------------------------------


class TestTradingCalendar:
    def test_save_and_check(self, db: Database) -> None:
        dates = ["2026-01-02", "2026-01-03", "2026-01-06"]
        n = db.save_trading_calendar(dates)
        assert n == 3
        assert db.is_trading_day("2026-01-02") is True
        assert db.is_trading_day("2026-01-04") is False

    def test_idempotent_insert(self, db: Database) -> None:
        assert db.save_trading_calendar(["2026-01-02"]) == 1
        assert db.save_trading_calendar(["2026-01-02"]) == 0
        count = db._conn.execute(
            "SELECT COUNT(*) FROM trading_calendar"
        ).fetchone()[0]
        assert count == 1


# ------------------------------------------------------------------
# fundamentals
# ------------------------------------------------------------------


class TestFundamentals:
    @staticmethod
    def _sample_df() -> pd.DataFrame:
        return pd.DataFrame({
            "date": ["2026-01-02", "2026-01-03"],
            "pe_ttm": [30.0, 31.0],
            "pb": [8.0, 8.2],
            "roe_ttm": [0.25, 0.26],
            "revenue_yoy": [0.15, 0.16],
            "profit_yoy": [0.20, 0.21],
            "market_cap": [2_000_000, 2_050_000],
        })

    def test_save_and_load(self, db: Database) -> None:
        n = db.save_fundamentals("600519", self._sample_df())
        assert n == 2
        loaded = db.load_fundamentals("600519", "2026-01-01", "2026-01-31")
        assert len(loaded) == 2
        assert loaded.iloc[0]["pe_ttm"] == pytest.approx(30.0)

    def test_empty_df(self, db: Database) -> None:
        assert db.save_fundamentals("600519", pd.DataFrame()) == 0

    def test_upsert(self, db: Database) -> None:
        db.save_fundamentals("600519", self._sample_df())
        updated = pd.DataFrame({
            "date": ["2026-01-02"],
            "pe_ttm": [35.0],
            "pb": [9.0],
            "roe_ttm": [0.28],
            "revenue_yoy": [0.18],
            "profit_yoy": [0.22],
            "market_cap": [2_100_000],
        })
        db.save_fundamentals("600519", updated)
        loaded = db.load_fundamentals("600519", "2026-01-02", "2026-01-02")
        assert loaded.iloc[0]["pe_ttm"] == pytest.approx(35.0)


# ------------------------------------------------------------------
# close
# ------------------------------------------------------------------


class TestLifecycle:
    def test_close(self, db: Database) -> None:
        db.close()
        with pytest.raises(Exception):
            db._conn.execute("SELECT 1")


# ------------------------------------------------------------------
# index_constituents
# ------------------------------------------------------------------


class TestIndexConstituents:
    def test_save_and_load(self, db: Database) -> None:
        rows = [
            {"symbol": "600519", "name": "贵州茅台", "index_code": "000300",
             "sector": "白酒", "updated_date": "2026-02-20"},
            {"symbol": "000858", "name": "五粮液", "index_code": "000300",
             "sector": "白酒", "updated_date": "2026-02-20"},
        ]
        n = db.save_index_constituents(rows)
        assert n == 2

        loaded = db.load_index_constituents("000300")
        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "000858"  # ordered by symbol
        assert loaded[1]["sector"] == "白酒"

    def test_empty_list(self, db: Database) -> None:
        assert db.save_index_constituents([]) == 0

    def test_upsert(self, db: Database) -> None:
        db.save_index_constituents([
            {"symbol": "600519", "name": "茅台旧", "index_code": "000300",
             "sector": "白酒", "updated_date": "2026-01-01"},
        ])
        db.save_index_constituents([
            {"symbol": "600519", "name": "茅台新", "index_code": "000300",
             "sector": "食品饮料", "updated_date": "2026-02-20"},
        ])
        loaded = db.load_index_constituents("000300")
        assert len(loaded) == 1
        assert loaded[0]["name"] == "茅台新"
        assert loaded[0]["sector"] == "食品饮料"

    def test_different_index_codes_isolated(self, db: Database) -> None:
        db.save_index_constituents([
            {"symbol": "600519", "name": "茅台", "index_code": "000300",
             "sector": "白酒", "updated_date": "2026-02-20"},
            {"symbol": "600519", "name": "茅台", "index_code": "000905",
             "sector": "白酒", "updated_date": "2026-02-20"},
        ])
        assert len(db.load_index_constituents("000300")) == 1
        assert len(db.load_index_constituents("000905")) == 1


# ------------------------------------------------------------------
# factor_pool_previous_day
# ------------------------------------------------------------------


class TestFactorPoolPreviousDay:
    def test_returns_previous_day_pool(self, db: Database) -> None:
        from pangu.tz import today_str
        today = today_str()

        db.save_factor_pool("2026-02-18", pd.DataFrame({
            "symbol": ["600519"], "score": [0.8], "rank": [1],
        }))
        db.save_factor_pool(today, pd.DataFrame({
            "symbol": ["000858"], "score": [0.9], "rank": [1],
        }))
        prev = db.load_factor_pool_previous_day()
        assert len(prev) == 1
        assert prev.iloc[0]["symbol"] == "600519"

    def test_returns_empty_when_no_previous(self, db: Database) -> None:
        prev = db.load_factor_pool_previous_day()
        assert prev.empty
