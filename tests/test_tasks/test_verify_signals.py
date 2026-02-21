"""Tests for T6 signal post-verification."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from trading_agent.data.storage import Database
from trading_agent.tasks.verify_signals import verify_signals

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    """Create a database with schema and calendar data."""
    d = Database(str(tmp_path / "test.db"))
    d.init_tables()
    d.save_trading_calendar([
        "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    ])
    return d


@pytest.fixture()
def components(db):
    """Minimal Components mock with a real Database."""
    c = MagicMock()
    c.db = db
    c.notif_manager = MagicMock()
    c.notif_manager.notify_markdown = AsyncMock(return_value={"FeishuNotifier": True})
    return c


def _insert_signal(db: Database, symbol: str, action: str, price: float, signal_date: str):
    """Insert a signal directly for testing."""
    db._conn.execute(
        "INSERT INTO trade_signals "
        "(timestamp, symbol, name, action, price, confidence, source, reason, signal_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (f"{signal_date} 08:15:00", symbol, f"Stock {symbol}", action,
         price, 0.75, "test", "reason", signal_date),
    )
    db._conn.commit()


def _insert_bars(db: Database, symbol: str, date: str, close: float):
    """Insert a daily bar for testing."""
    df = pd.DataFrame([{
        "date": date, "open": close, "high": close, "low": close,
        "close": close, "volume": 1000,
    }])
    db.save_daily_bars(symbol, df)


# ---------------------------------------------------------------------------
# Tests: storage methods
# ---------------------------------------------------------------------------

class TestStorageVerification:
    """Test new storage methods for signal verification."""

    def test_get_trading_day_offset(self, db):
        assert db.get_trading_day_offset("2026-02-20", 1) == "2026-02-19"
        assert db.get_trading_day_offset("2026-02-20", 3) == "2026-02-17"
        assert db.get_trading_day_offset("2026-02-20", 10) is None

    def test_load_unverified_signals(self, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-17")
        _insert_signal(db, "600002", "SELL", 20.0, "2026-02-17")
        _insert_signal(db, "600003", "HOLD", 30.0, "2026-02-17")

        result = db.load_unverified_signals("2026-02-17", 1)
        assert len(result) == 2  # HOLD excluded
        symbols = {r["symbol"] for r in result}
        assert symbols == {"600001", "600002"}

    def test_load_unverified_skips_already_verified(self, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-17")
        db._conn.execute(
            "UPDATE trade_signals SET return_1d = 5.0 WHERE symbol = '600001'"
        )
        db._conn.commit()

        result = db.load_unverified_signals("2026-02-17", 1)
        assert len(result) == 0

    def test_update_signal_return(self, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-17")
        row = db._conn.execute("SELECT id FROM trade_signals WHERE symbol = '600001'").fetchone()
        db.update_signal_return(row[0], 1, 5.0)
        db.update_signal_return(row[0], 3, -2.5)

        updated = db._conn.execute(
            "SELECT return_1d, return_3d, return_5d FROM trade_signals WHERE id = ?",
            (row[0],),
        ).fetchone()
        assert updated[0] == 5.0
        assert updated[1] == -2.5
        assert updated[2] is None

    def test_get_signal_returns(self, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-17")
        _insert_signal(db, "600002", "SELL", 20.0, "2026-02-17")

        row1 = db._conn.execute("SELECT id FROM trade_signals WHERE symbol = '600001'").fetchone()
        row2 = db._conn.execute("SELECT id FROM trade_signals WHERE symbol = '600002'").fetchone()

        db.update_signal_return(row1[0], 1, 5.0)   # BUY +5% → strategy +5%
        db.update_signal_return(row2[0], 1, -3.0)   # SELL -3% → strategy +3%

        ret = db.get_signal_returns(30)
        assert ret["1d"]["count"] == 2
        assert ret["1d"]["avg_return"] == 4.0  # (5+3)/2
        assert ret["3d"]["count"] == 0

    def test_migration_adds_columns(self, tmp_path):
        """Existing DB without return columns gets migrated."""
        db = Database(str(tmp_path / "old.db"))
        db.init_tables()
        cols = {
            row[1]
            for row in db._conn.execute("PRAGMA table_info(trade_signals)").fetchall()
        }
        assert "return_1d" in cols
        assert "return_3d" in cols
        assert "return_5d" in cols


# ---------------------------------------------------------------------------
# Tests: verify_signals task
# ---------------------------------------------------------------------------

class TestVerifySignalsTask:
    """Test the T6 verify_signals task."""

    @pytest.mark.asyncio
    @patch("trading_agent.tasks.verify_signals.today_str", return_value="2026-02-20")
    async def test_buy_positive_return_is_correct(self, _mock, components, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-19")
        _insert_bars(db, "600001", "2026-02-20", 11.0)

        await verify_signals(components)

        row = db._conn.execute(
            "SELECT return_1d FROM trade_signals WHERE symbol = '600001'"
        ).fetchone()
        assert row[0] == pytest.approx(10.0, abs=0.01)
        components.notif_manager.notify_markdown.assert_called_once()
        content = components.notif_manager.notify_markdown.call_args[0][1]
        assert "✅" in content

    @pytest.mark.asyncio
    @patch("trading_agent.tasks.verify_signals.today_str", return_value="2026-02-20")
    async def test_buy_negative_return_is_incorrect(self, _mock, components, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-19")
        _insert_bars(db, "600001", "2026-02-20", 9.0)

        await verify_signals(components)

        row = db._conn.execute(
            "SELECT return_1d FROM trade_signals WHERE symbol = '600001'"
        ).fetchone()
        assert row[0] == pytest.approx(-10.0, abs=0.01)
        content = components.notif_manager.notify_markdown.call_args[0][1]
        assert "❌" in content

    @pytest.mark.asyncio
    @patch("trading_agent.tasks.verify_signals.today_str", return_value="2026-02-20")
    async def test_sell_negative_return_is_correct(self, _mock, components, db):
        _insert_signal(db, "600001", "SELL", 10.0, "2026-02-19")
        _insert_bars(db, "600001", "2026-02-20", 8.0)

        await verify_signals(components)

        row = db._conn.execute(
            "SELECT return_1d FROM trade_signals WHERE symbol = '600001'"
        ).fetchone()
        assert row[0] == pytest.approx(-20.0, abs=0.01)
        content = components.notif_manager.notify_markdown.call_args[0][1]
        assert "✅" in content

    @pytest.mark.asyncio
    @patch("trading_agent.tasks.verify_signals.today_str", return_value="2026-02-20")
    async def test_multiple_lookbacks(self, _mock, components, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-19")  # 1d ago
        _insert_signal(db, "600002", "BUY", 20.0, "2026-02-17")  # 3d ago
        for sym, close in [("600001", 11.0), ("600002", 22.0)]:
            _insert_bars(db, sym, "2026-02-20", close)

        await verify_signals(components)

        r1 = db._conn.execute(
            "SELECT return_1d FROM trade_signals WHERE symbol = '600001'"
        ).fetchone()
        assert r1[0] == pytest.approx(10.0, abs=0.01)

        r3 = db._conn.execute(
            "SELECT return_3d FROM trade_signals WHERE symbol = '600002'"
        ).fetchone()
        assert r3[0] == pytest.approx(10.0, abs=0.01)

    @pytest.mark.asyncio
    @patch("trading_agent.tasks.verify_signals.today_str", return_value="2026-02-20")
    async def test_no_signals_no_push(self, _mock, components, db):
        await verify_signals(components)
        components.notif_manager.notify_markdown.assert_not_called()

    @pytest.mark.asyncio
    @patch("trading_agent.tasks.verify_signals.today_str", return_value="2026-02-20")
    async def test_no_notifier_no_error(self, _mock, components, db):
        components.notif_manager = None
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-19")
        _insert_bars(db, "600001", "2026-02-20", 11.0)

        await verify_signals(components)

        row = db._conn.execute(
            "SELECT return_1d FROM trade_signals WHERE symbol = '600001'"
        ).fetchone()
        assert row[0] == pytest.approx(10.0, abs=0.01)

    @pytest.mark.asyncio
    @patch("trading_agent.tasks.verify_signals.today_str", return_value="2026-02-20")
    async def test_missing_price_data_skipped(self, _mock, components, db):
        _insert_signal(db, "600001", "BUY", 10.0, "2026-02-19")

        await verify_signals(components)

        row = db._conn.execute(
            "SELECT return_1d FROM trade_signals WHERE symbol = '600001'"
        ).fetchone()
        assert row[0] is None
