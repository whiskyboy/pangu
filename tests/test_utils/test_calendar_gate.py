"""Tests for is_rebalance_day — ISO week-start gate."""

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
