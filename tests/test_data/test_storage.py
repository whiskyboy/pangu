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
        """All expected tables must exist after init_tables()."""
        rows = db._conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
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
            "portfolio_snapshots",
            "task_runs",
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
        return pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-03"],
                "open": [10.0, 10.5],
                "high": [11.0, 11.5],
                "low": [9.5, 10.0],
                "close": [10.8, 11.2],
                "volume": [100000, 120000],
                "amount": [1_080_000, 1_344_000],
                "adj_factor": [1.0, 1.0],
            }
        )

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

        updated = pd.DataFrame(
            {
                "date": ["2026-01-02"],
                "open": [10.0],
                "high": [12.0],
                "low": [9.5],
                "close": [11.9],
                "volume": [150000],
            }
        )
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
        df = pd.DataFrame(
            {
                "date": ["2026-01-02"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.5],
                "close": [10.8],
                "volume": [100000],
            }
        )
        db.save_daily_bars("600519", df)
        loaded = db.load_daily_bars("600519", "2026-01-01", "2026-01-31")
        assert loaded.iloc[0]["adj_factor"] == pytest.approx(1.0)

    def test_extended_columns_save_and_load(self, db: Database) -> None:
        """New columns (turn, preclose, tradestatus, is_st) round-trip."""
        df = pd.DataFrame(
            {
                "date": ["2026-01-02"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.5],
                "close": [10.8],
                "volume": [100000],
                "amount": [1_080_000],
                "adj_factor": [1.0],
                "turn": [0.5],
                "preclose": [10.5],
                "tradestatus": ["1"],
                "is_st": [0],
            }
        )
        db.save_daily_bars("600519", df)
        loaded = db.load_daily_bars("600519", "2026-01-01", "2026-01-31")
        row = loaded.iloc[0]
        assert row["turn"] == pytest.approx(0.5)
        assert row["preclose"] == pytest.approx(10.5)
        assert row["tradestatus"] == "1"
        assert row["is_st"] == 0

    def test_extended_columns_default_none(self, db: Database) -> None:
        """Extended columns default to None when not provided."""
        df = self._sample_df()
        db.save_daily_bars("600519", df)
        loaded = db.load_daily_bars("600519", "2026-01-01", "2026-01-31")
        assert loaded.iloc[0]["turn"] is None
        assert loaded.iloc[0]["is_st"] is None


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
            "600519",
            "daily_bars",
            "failed",
            "akshare",
            error_msg="connection timeout",
        )
        row = db._conn.execute(
            "SELECT status, error_msg FROM data_sync_log WHERE symbol = ? AND data_type = ?",
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


def _make_signal(symbol: str = "600519", action: Action = Action.BUY) -> TradeSignal:
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
            symbol="600519",
            name="贵州茅台",
            action=Action.BUY,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=1800.0,
            confidence=0.85,
            source="factor",
            reason="test",
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
        count = db._conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0]
        assert count == 1


# ------------------------------------------------------------------
# fundamentals
# ------------------------------------------------------------------


class TestFundamentals:
    @staticmethod
    def _sample_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-03"],
                "pe_ttm": [30.0, 31.0],
                "pb": [8.0, 8.2],
                "roe_ttm": [0.25, 0.26],
                "revenue_yoy": [0.15, 0.16],
                "profit_yoy": [0.20, 0.21],
                "market_cap": [2_000_000, 2_050_000],
            }
        )

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
        updated = pd.DataFrame(
            {
                "date": ["2026-01-02"],
                "pe_ttm": [35.0],
                "pb": [9.0],
                "roe_ttm": [0.28],
                "revenue_yoy": [0.18],
                "profit_yoy": [0.22],
                "market_cap": [2_100_000],
            }
        )
        db.save_fundamentals("600519", updated)
        loaded = db.load_fundamentals("600519", "2026-01-02", "2026-01-02")
        assert loaded.iloc[0]["pe_ttm"] == pytest.approx(35.0)

    def test_save_new_columns(self, db: Database) -> None:
        """New fundamental columns (net_profit_margin etc.) can be saved and loaded."""
        df = pd.DataFrame(
            {
                "date": ["2024-03-31"],
                "roe_ttm": [0.15],
                "net_profit_margin": [0.35],
                "debt_ratio": [0.42],
                "current_ratio": [1.8],
            }
        )
        db.save_fundamentals("600519", df)
        loaded = db.load_fundamentals("600519", "2024-03-31", "2024-03-31")
        assert loaded.iloc[0]["net_profit_margin"] == pytest.approx(0.35)
        assert loaded.iloc[0]["debt_ratio"] == pytest.approx(0.42)
        assert loaded.iloc[0]["current_ratio"] == pytest.approx(1.8)

    def test_upsert_preserves_existing_columns(self, db: Database) -> None:
        """Writing quarterly data (ROE) doesn't overwrite daily data (PE/PB)."""
        # First: daily PE/PB
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2024-03-31"],
                    "pe_ttm": [25.0],
                    "pb": [3.0],
                }
            ),
        )
        # Second: quarterly ROE (pe_ttm=None should NOT overwrite 25.0)
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2024-03-31"],
                    "roe_ttm": [0.15],
                    "net_profit_margin": [0.30],
                }
            ),
        )
        loaded = db.load_fundamentals("600519", "2024-03-31", "2024-03-31")
        assert loaded.iloc[0]["pe_ttm"] == pytest.approx(25.0)  # preserved
        assert loaded.iloc[0]["roe_ttm"] == pytest.approx(0.15)  # merged

    def test_load_fundamentals_filled_basic(self, db: Database) -> None:
        """Forward fill quarterly columns across daily rows."""
        rows = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-03-31", "2024-04-01", "2024-04-02"],
                "pe_ttm": [25.0, 25.5, 26.0, 26.5, 27.0],
                "pb": [3.0, 3.1, 3.2, 3.3, 3.4],
                "roe_ttm": [None, None, 0.15, None, None],
                "profit_yoy": [None, None, 0.10, None, None],
            }
        )
        db.save_fundamentals("600519", rows)
        filled = db.load_fundamentals_filled("600519", "2024-01-02", "2024-04-02")
        # Before Q1 report: ROE should still be None (no seed)
        assert pd.isna(filled.iloc[0]["roe_ttm"])
        # After Q1 report: ROE should be forward-filled
        assert filled.iloc[3]["roe_ttm"] == pytest.approx(0.15)
        assert filled.iloc[4]["roe_ttm"] == pytest.approx(0.15)
        # PE/PB should remain unchanged (daily, not ffilled)
        assert filled.iloc[0]["pe_ttm"] == pytest.approx(25.0)

    def test_load_fundamentals_filled_seed(self, db: Database) -> None:
        """Seed row from before start date fills first rows."""
        # Q4 2023 report (before our query range)
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2023-12-31"],
                    "roe_ttm": [0.20],
                }
            ),
        )
        # Daily PE rows in Jan 2024 (no quarterly data)
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "pe_ttm": [25.0, 25.5],
                }
            ),
        )
        filled = db.load_fundamentals_filled("600519", "2024-01-02", "2024-01-03")
        # Should pick up 2023-12-31 ROE as seed
        assert len(filled) == 2
        assert filled.iloc[0]["roe_ttm"] == pytest.approx(0.20)
        assert filled.iloc[1]["roe_ttm"] == pytest.approx(0.20)
        # Seed row (2023-12-31) should NOT appear in output
        assert filled.iloc[0]["date"] == "2024-01-02"

    def test_load_fundamentals_filled_empty(self, db: Database) -> None:
        """Empty table returns empty DataFrame."""
        filled = db.load_fundamentals_filled("600519", "2024-01-01", "2024-12-31")
        assert filled.empty

    def test_update_gross_margin_batch(self, db: Database) -> None:
        """Batch gross_margin update only touches existing rows."""
        # Pre-populate rows so UPDATE can find them
        db.save_fundamentals("600519", pd.DataFrame({"date": ["2024-03-31"], "pe_ttm": [30.0]}))
        db.save_fundamentals("000001", pd.DataFrame({"date": ["2024-03-31"], "pe_ttm": [8.0]}))
        data = {"600519": 0.9187, "000001": 0.2977, "999999": 0.5}  # 999999 not in DB
        n = db.update_gross_margin_batch("2024-03-31", data)
        assert n == 2  # only 600519 and 000001 exist, 999999 skipped
        loaded = db.load_fundamentals("600519", "2024-03-31", "2024-03-31")
        assert len(loaded) == 1
        assert loaded.iloc[0]["gross_margin"] == pytest.approx(0.9187)
        # 999999 should NOT have been inserted
        orphan = db.load_fundamentals("999999", "2024-03-31", "2024-03-31")
        assert len(orphan) == 0

    def test_update_gross_margin_batch_preserves_existing(self, db: Database) -> None:
        """Batch gross_margin update does not overwrite existing columns."""
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2024-03-31"],
                    "pe_ttm": [25.0],
                    "roe_ttm": [0.15],
                }
            ),
        )
        db.update_gross_margin_batch("2024-03-31", {"600519": 0.92})
        loaded = db.load_fundamentals("600519", "2024-03-31", "2024-03-31")
        assert loaded.iloc[0]["pe_ttm"] == pytest.approx(25.0)  # preserved
        assert loaded.iloc[0]["roe_ttm"] == pytest.approx(0.15)  # preserved
        assert loaded.iloc[0]["gross_margin"] == pytest.approx(0.92)  # updated

    def test_update_gross_margin_batch_empty(self, db: Database) -> None:
        """Empty data dict is a no-op."""
        assert db.update_gross_margin_batch("2024-03-31", {}) == 0

    # ------------------------------------------------------------------
    # pub_date / PIT tests
    # ------------------------------------------------------------------

    def test_update_pub_date_batch(self, db: Database) -> None:
        """Batch pub_date update writes to existing quarterly rows."""
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2024-03-31", "2024-06-30"],
                    "roe_ttm": [0.15, 0.18],
                }
            ),
        )
        data = {"2024-03-31": "2024-04-27", "2024-06-30": "2024-08-25"}
        n = db.update_pub_date_batch("600519", data)
        assert n == 2
        loaded = db.load_fundamentals("600519", "2024-03-31", "2024-06-30")
        assert loaded.iloc[0]["pub_date"] == "2024-04-27"
        assert loaded.iloc[1]["pub_date"] == "2024-08-25"

    def test_update_pub_date_batch_empty(self, db: Database) -> None:
        """Empty data dict is a no-op."""
        assert db.update_pub_date_batch("600519", {}) == 0

    def test_load_fundamentals_filled_pit_delays_quarterly(self, db: Database) -> None:
        """PIT: quarterly values delayed to pub_date, not available from report date."""
        # Q1 2024 report (period-end 2024-03-31, announced 2024-04-27)
        rows = pd.DataFrame(
            {
                "date": [
                    "2024-03-29",
                    "2024-03-31",
                    "2024-04-01",
                    "2024-04-26",
                    "2024-04-27",
                    "2024-04-28",
                ],
                "pe_ttm": [25.0, 25.5, 26.0, 27.0, 27.5, 28.0],
                "roe_ttm": [None, 0.15, None, None, None, None],
                "pub_date": [None, "2024-04-27", None, None, None, None],
            }
        )
        db.save_fundamentals("600519", rows)
        filled = db.load_fundamentals_filled("600519", "2024-03-29", "2024-04-28")

        # Before pub_date (2024-04-27): ROE should NOT be available
        assert pd.isna(filled.loc[filled["date"] == "2024-03-29", "roe_ttm"].iloc[0])
        assert pd.isna(filled.loc[filled["date"] == "2024-03-31", "roe_ttm"].iloc[0])
        assert pd.isna(filled.loc[filled["date"] == "2024-04-01", "roe_ttm"].iloc[0])
        assert pd.isna(filled.loc[filled["date"] == "2024-04-26", "roe_ttm"].iloc[0])

        # From pub_date onward: ROE should be 0.15
        assert filled.loc[filled["date"] == "2024-04-27", "roe_ttm"].iloc[0] == pytest.approx(0.15)
        assert filled.loc[filled["date"] == "2024-04-28", "roe_ttm"].iloc[0] == pytest.approx(0.15)

    def test_load_fundamentals_filled_pit_pub_date_not_in_rows(self, db: Database) -> None:
        """PIT: when pub_date falls on a non-trading day, values go to next available row."""
        # pub_date 2024-04-27 is Saturday — next trading day is 2024-04-29
        rows = pd.DataFrame(
            {
                "date": ["2024-03-31", "2024-04-26", "2024-04-29"],
                "pe_ttm": [25.0, 26.0, 27.0],
                "roe_ttm": [0.15, None, None],
                "pub_date": ["2024-04-27", None, None],
            }
        )
        db.save_fundamentals("600519", rows)
        filled = db.load_fundamentals_filled("600519", "2024-03-31", "2024-04-29")

        # 03-31 and 04-26 should NOT have ROE (pub_date is after)
        assert pd.isna(filled.loc[filled["date"] == "2024-03-31", "roe_ttm"].iloc[0])
        assert pd.isna(filled.loc[filled["date"] == "2024-04-26", "roe_ttm"].iloc[0])

        # 04-29 (first row >= pub_date 04-27) should have ROE
        assert filled.loc[filled["date"] == "2024-04-29", "roe_ttm"].iloc[0] == pytest.approx(0.15)

    def test_load_fundamentals_filled_pit_no_pub_date_falls_back(self, db: Database) -> None:
        """Without pub_date, quarterly ffill works as before (backward compatible)."""
        rows = pd.DataFrame(
            {
                "date": ["2024-03-31", "2024-04-01", "2024-04-02"],
                "pe_ttm": [25.0, 25.5, 26.0],
                "roe_ttm": [0.15, None, None],
                # no pub_date column or all NULL
            }
        )
        db.save_fundamentals("600519", rows)
        filled = db.load_fundamentals_filled("600519", "2024-03-31", "2024-04-02")

        # Without PIT, ffill from report date as before
        assert filled.iloc[0]["roe_ttm"] == pytest.approx(0.15)
        assert filled.iloc[1]["roe_ttm"] == pytest.approx(0.15)
        assert filled.iloc[2]["roe_ttm"] == pytest.approx(0.15)

    def test_load_fundamentals_filled_pit_same_pub_date(self, db: Database) -> None:
        """PIT: when Q3 and Q4 share the same pub_date, newer quarter wins."""
        rows = pd.DataFrame(
            {
                "date": ["2024-09-30", "2024-12-31", "2025-04-27", "2025-04-28"],
                "pe_ttm": [20.0, 21.0, 22.0, 23.0],
                "roe_ttm": [0.18, 0.20, None, None],
                "pub_date": ["2025-04-27", "2025-04-27", None, None],
            }
        )
        db.save_fundamentals("600519", rows)
        filled = db.load_fundamentals_filled("600519", "2024-09-30", "2025-04-28")

        # Before pub_date: both Q3 and Q4 not yet announced
        assert pd.isna(filled.loc[filled["date"] == "2024-09-30", "roe_ttm"].iloc[0])
        assert pd.isna(filled.loc[filled["date"] == "2024-12-31", "roe_ttm"].iloc[0])

        # On pub_date: Q4 (newer, roe=0.20) should overwrite Q3 (roe=0.18)
        assert filled.loc[filled["date"] == "2025-04-27", "roe_ttm"].iloc[0] == pytest.approx(0.20)
        assert filled.loc[filled["date"] == "2025-04-28", "roe_ttm"].iloc[0] == pytest.approx(0.20)

    def test_load_fundamentals_filled_pit_seed_with_pub_date(self, db: Database) -> None:
        """PIT: seed row from before range is still used correctly."""
        # Old Q4 data with pub_date already passed
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2023-12-31"],
                    "roe_ttm": [0.20],
                    "pub_date": ["2024-04-03"],  # announced before our query range
                }
            ),
        )
        # Daily rows in May 2024
        db.save_fundamentals(
            "600519",
            pd.DataFrame(
                {
                    "date": ["2024-05-01", "2024-05-02"],
                    "pe_ttm": [25.0, 25.5],
                }
            ),
        )
        filled = db.load_fundamentals_filled("600519", "2024-05-01", "2024-05-02")

        # Seed pub_date 2024-04-03 < query start 2024-05-01 → data already available
        assert filled.iloc[0]["roe_ttm"] == pytest.approx(0.20)
        assert filled.iloc[1]["roe_ttm"] == pytest.approx(0.20)


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
            {"symbol": "600519", "name": "贵州茅台", "index_code": "000300", "sector": "白酒", "date": "2026-02-20"},
            {"symbol": "000858", "name": "五粮液", "index_code": "000300", "sector": "白酒", "date": "2026-02-20"},
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
        db.save_index_constituents(
            [
                {"symbol": "600519", "name": "茅台旧", "index_code": "000300", "sector": "白酒", "date": "2026-01-01"},
            ]
        )
        db.save_index_constituents(
            [
                {
                    "symbol": "600519",
                    "name": "茅台新",
                    "index_code": "000300",
                    "sector": "食品饮料",
                    "date": "2026-02-20",
                },
            ]
        )
        loaded = db.load_index_constituents("000300")
        assert len(loaded) == 1
        assert loaded[0]["name"] == "茅台新"
        assert loaded[0]["sector"] == "食品饮料"

    def test_different_index_codes_isolated(self, db: Database) -> None:
        db.save_index_constituents(
            [
                {"symbol": "600519", "name": "茅台", "index_code": "000300", "sector": "白酒", "date": "2026-02-20"},
                {"symbol": "600519", "name": "茅台", "index_code": "000905", "sector": "白酒", "date": "2026-02-20"},
            ]
        )
        assert len(db.load_index_constituents("000300")) == 1
        assert len(db.load_index_constituents("000905")) == 1


# ------------------------------------------------------------------
# factor_pool_previous_day
# ------------------------------------------------------------------


class TestFactorPoolPreviousDay:
    def test_returns_previous_day_pool(self, db: Database) -> None:
        from pangu.tz import today_str

        today = today_str()

        db.save_factor_pool(
            "2026-02-18",
            pd.DataFrame(
                {
                    "symbol": ["600519"],
                    "score": [0.8],
                    "rank": [1],
                }
            ),
        )
        db.save_factor_pool(
            today,
            pd.DataFrame(
                {
                    "symbol": ["000858"],
                    "score": [0.9],
                    "rank": [1],
                }
            ),
        )
        prev = db.load_factor_pool_previous_day()
        assert len(prev) == 1
        assert prev.iloc[0]["symbol"] == "600519"

    def test_returns_empty_when_no_previous(self, db: Database) -> None:
        prev = db.load_factor_pool_previous_day()
        assert prev.empty


# ------------------------------------------------------------------
# portfolio_snapshots
# ------------------------------------------------------------------


class TestPortfolioSnapshots:
    def test_save_and_get_single(self, db: Database) -> None:
        db.save_portfolio_snapshot(
            "2026-02-16",
            ["600519", "000858"],
            is_rebalance=True,
        )
        snap = db.get_portfolio_snapshot("2026-02-16")
        assert snap is not None
        assert snap["symbols"] == ["000858", "600519"]  # stored sorted
        assert snap["is_rebalance"] is True

    def test_save_replaces_existing(self, db: Database) -> None:
        db.save_portfolio_snapshot("2026-02-16", ["600519"], is_rebalance=False)
        db.save_portfolio_snapshot(
            "2026-02-16",
            ["000858"],
            is_rebalance=True,
        )
        snap = db.get_portfolio_snapshot("2026-02-16")
        assert snap["symbols"] == ["000858"]
        assert snap["is_rebalance"] is True

    def test_get_returns_none_when_missing(self, db: Database) -> None:
        assert db.get_portfolio_snapshot("2099-01-01") is None

    def test_get_snapshots_ordered_by_date(self, db: Database) -> None:
        for d in ["2026-02-20", "2026-02-09", "2026-02-13"]:
            db.save_portfolio_snapshot(d, [], is_rebalance=False)
        rows = db.get_portfolio_snapshots()
        dates = [r["date"] for r in rows]
        assert dates == ["2026-02-09", "2026-02-13", "2026-02-20"]

    def test_get_snapshots_filters_date_range(self, db: Database) -> None:
        for d in ["2026-02-09", "2026-02-13", "2026-02-20"]:
            db.save_portfolio_snapshot(d, [], is_rebalance=False)
        rows = db.get_portfolio_snapshots(start="2026-02-10", end="2026-02-15")
        dates = [r["date"] for r in rows]
        assert dates == ["2026-02-13"]

    def test_get_latest_returns_max_date(self, db: Database) -> None:
        for d in ["2026-02-09", "2026-02-20", "2026-02-13"]:
            db.save_portfolio_snapshot(d, ["X"], is_rebalance=False)
        latest = db.get_latest_portfolio_snapshot()
        assert latest is not None
        assert latest["date"] == "2026-02-20"

    def test_get_latest_returns_none_when_empty(self, db: Database) -> None:
        assert db.get_latest_portfolio_snapshot() is None


# ------------------------------------------------------------------
# task_runs
# ------------------------------------------------------------------


class TestTaskRuns:
    def test_record_and_get_recent(self, db: Database) -> None:
        db.record_task_run(
            task_id="T1",
            name="全球行情同步",
            started_at="2026-02-16 08:00:00",
            completed_at="2026-02-16 08:00:30",
            status="success",
            duration_ms=30000,
        )
        rows = db.get_recent_task_runs("T1")
        assert len(rows) == 1
        assert rows[0]["task_id"] == "T1"
        assert rows[0]["status"] == "success"
        assert rows[0]["duration_ms"] == 30000

    def test_get_recent_filters_by_task_id(self, db: Database) -> None:
        for tid in ["T1", "T2", "T1"]:
            db.record_task_run(
                task_id=tid,
                name=tid,
                started_at="2026-02-16 08:00:00",
                completed_at="2026-02-16 08:00:01",
                status="success",
            )
        t1 = db.get_recent_task_runs("T1")
        assert len(t1) == 2
        all_runs = db.get_recent_task_runs()
        assert len(all_runs) == 3

    def test_get_recent_ordered_desc(self, db: Database) -> None:
        for ts in ["2026-02-16 08:00:00", "2026-02-16 09:00:00", "2026-02-16 07:00:00"]:
            db.record_task_run(
                task_id="T1",
                name="T1",
                started_at=ts,
                completed_at=ts,
                status="success",
            )
        rows = db.get_recent_task_runs("T1")
        starts = [r["started_at"] for r in rows]
        assert starts == sorted(starts, reverse=True)

    def test_get_recent_respects_limit(self, db: Database) -> None:
        for i in range(5):
            db.record_task_run(
                task_id="T1",
                name="T1",
                started_at=f"2026-02-{16 + i:02d} 08:00:00",
                completed_at=f"2026-02-{16 + i:02d} 08:00:01",
                status="success",
            )
        assert len(db.get_recent_task_runs("T1", limit=3)) == 3

    def test_get_last_successful_run(self, db: Database) -> None:
        db.record_task_run(
            task_id="T1",
            name="T1",
            started_at="2026-02-15 08:00:00",
            completed_at="2026-02-15 08:00:01",
            status="success",
        )
        db.record_task_run(
            task_id="T1",
            name="T1",
            started_at="2026-02-16 08:00:00",
            completed_at="2026-02-16 08:00:01",
            status="failed",
            error_msg="boom",
        )
        last = db.get_last_successful_run("T1")
        assert last == "2026-02-15 08:00:00"

    def test_get_last_successful_run_none_when_no_success(self, db: Database) -> None:
        db.record_task_run(
            task_id="T1",
            name="T1",
            started_at="2026-02-16 08:00:00",
            completed_at="2026-02-16 08:00:01",
            status="failed",
        )
        assert db.get_last_successful_run("T1") is None

    def test_cleanup_old_task_runs(self, db: Database) -> None:
        from datetime import timedelta

        from pangu.tz import now as _now

        recent = _now().strftime("%Y-%m-%d %H:%M:%S")
        old = (_now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
        for ts in [recent, old]:
            db.record_task_run(
                task_id="T1",
                name="T1",
                started_at=ts,
                completed_at=ts,
                status="success",
            )
        deleted = db.cleanup_old_task_runs(days=30)
        assert deleted == 1
        remaining = db.get_recent_task_runs("T1")
        assert len(remaining) == 1
        assert remaining[0]["started_at"] == recent
