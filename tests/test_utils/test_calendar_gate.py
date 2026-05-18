"""Tests for is_rebalance_day — production gate using configurable schedule.

Default config (``mode="weekly", weekly_day=1``) preserves the original
ISO-week-first-trading-day behaviour, so these existing cases continue
to pass after the cadence refactor.
"""

from __future__ import annotations

import pytest

from pangu.data.storage import Database
from pangu.utils import is_rebalance_day


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(str(tmp_path / "calendar.db"))
    d.init_tables()
    return d


def _populate(db: Database, dates: list[str]) -> None:
    """Insert trading calendar dates."""
    db.save_trading_calendar(dates)


class TestIsRebalanceDay:
    def test_non_trading_day_returns_false(self, db: Database):
        _populate(db, ["2024-01-02", "2024-01-03"])
        # Sunday 2024-01-07 not in calendar
        assert is_rebalance_day("2024-01-07", db) is False

    def test_first_trading_day_no_prior_returns_true(self, db: Database):
        # Calendar contains only today → cold start
        _populate(db, ["2024-01-02"])
        assert is_rebalance_day("2024-01-02", db) is True

    def test_first_trading_day_of_new_iso_week_is_true(self, db: Database):
        # Mon 2024-01-08 begins ISO week 2; previous trading day 2024-01-05 (Fri) is week 1
        _populate(db, ["2024-01-04", "2024-01-05", "2024-01-08"])
        assert is_rebalance_day("2024-01-08", db) is True

    def test_mid_week_trading_day_is_false(self, db: Database):
        # 2024-01-09 (Tue) is the same ISO week as 2024-01-08 (Mon)
        _populate(db, ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
        assert is_rebalance_day("2024-01-09", db) is False

    def test_friday_in_same_week_is_false(self, db: Database):
        _populate(db, ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"])
        assert is_rebalance_day("2024-01-12", db) is False

    def test_year_boundary_rebalance(self, db: Database):
        # 2023-12-29 (Fri, ISO week 52) → 2024-01-02 (Tue, ISO week 1) crosses years
        _populate(db, ["2023-12-29", "2024-01-02"])
        assert is_rebalance_day("2024-01-02", db) is True

    def test_iso_week_53_handling(self, db: Database):
        # 2025-12-29 is Mon, ISO week 1 of 2026; 2025-12-26 Fri is ISO week 52 of 2025
        _populate(db, ["2025-12-26", "2025-12-29"])
        assert is_rebalance_day("2025-12-29", db) is True

    def test_works_when_first_trading_day_is_tuesday(self, db: Database):
        # New Year's Day was holiday; Tue 2024-01-02 is the first trading day
        _populate(db, ["2024-01-02", "2024-01-03"])
        # No prior trading day → cold start branch returns True
        assert is_rebalance_day("2024-01-02", db) is True


class TestConfigurableSchedule:
    """Verify is_rebalance_day honors ``[rebalance]`` settings via the
    ``RebalanceSchedule`` integration."""

    @pytest.fixture(autouse=True)
    def _reset_settings(self):
        from pangu.config import reset_settings

        reset_settings()
        yield
        reset_settings()

    def test_weekly_friday_schedule(self, db, monkeypatch):
        from pangu.config import get_settings

        get_settings()  # load defaults
        monkeypatch.setitem(get_settings().rebalance, "weekly_day", 5)  # weekly:5 (Friday)
        get_settings().rebalance["mode"] = "weekly"
        # Calendar: full week — Mon..Fri all trading
        _populate(db, ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"])
        # Cold-start would short-circuit; need prev trading day in DB. Use
        # 2024-01-12 (Fri) with prev = 01-11 (Thu) → schedule says Fri rebalance.
        assert is_rebalance_day("2024-01-12", db) is True
        # Monday is NOT a rebalance day under weekly:5 (prev Fri 01-05 not in
        # calendar, so 01-08 absorbs the deferred prev-Fri rebalance).
        # We test with prev Fri populated to disambiguate:
        _populate(db, ["2024-01-05"])
        assert is_rebalance_day("2024-01-08", db) is False
        assert is_rebalance_day("2024-01-12", db) is True

    def test_monthly_schedule(self, db, monkeypatch):
        from pangu.config import get_settings

        get_settings()
        get_settings().rebalance["mode"] = "monthly"
        monkeypatch.setitem(get_settings().rebalance, "monthly_day", 15)
        # 2024-02-15 = Thursday, trading day
        _populate(db, ["2024-02-14", "2024-02-15", "2024-02-16"])
        assert is_rebalance_day("2024-02-15", db) is True
        # 2024-02-16 is the day after — not a rebalance day
        assert is_rebalance_day("2024-02-16", db) is False

    def test_invalid_mode_in_config_raises(self, db, monkeypatch):
        from pangu.config import get_settings

        get_settings()
        get_settings().rebalance["mode"] = "daily"
        _populate(db, ["2024-01-04", "2024-01-05"])
        with pytest.raises(ValueError, match="Invalid rebalance mode"):
            is_rebalance_day("2024-01-05", db)
