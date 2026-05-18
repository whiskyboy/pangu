"""Tests for ``RebalanceSchedule``."""

from __future__ import annotations

from datetime import date

import pytest

from pangu.rebalance import RebalanceSchedule


def make_predicate(trading_days: list[str]):
    """Build an ``is_trading_day`` predicate from ISO-format dates."""
    s = {date.fromisoformat(d) for d in trading_days}
    return lambda d: d in s


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_weekly_valid(self):
        s = RebalanceSchedule(mode="weekly", day=1)
        assert s.mode == "weekly"
        assert s.day == 1

    def test_monthly_valid(self):
        s = RebalanceSchedule(mode="monthly", day=15)
        assert s.mode == "monthly"
        assert s.day == 15

    @pytest.mark.parametrize("day", [0, 6, 7, -1, 10])
    def test_weekly_day_out_of_range(self, day):
        with pytest.raises(ValueError, match="weekly_day"):
            RebalanceSchedule(mode="weekly", day=day)

    @pytest.mark.parametrize("day", [0, 29, 30, 31, -1, 100])
    def test_monthly_day_out_of_range(self, day):
        with pytest.raises(ValueError, match="monthly_day"):
            RebalanceSchedule(mode="monthly", day=day)

    @pytest.mark.parametrize("mode", ["daily", "yearly", "", "WEEKLY"])
    def test_invalid_mode(self, mode):
        with pytest.raises(ValueError, match="Invalid rebalance mode"):
            RebalanceSchedule(mode=mode, day=1)  # type: ignore[arg-type]


class TestFromString:
    def test_weekly(self):
        s = RebalanceSchedule.from_string("weekly:1")
        assert s == RebalanceSchedule("weekly", 1)

    def test_monthly(self):
        s = RebalanceSchedule.from_string("monthly:15")
        assert s == RebalanceSchedule("monthly", 15)

    @pytest.mark.parametrize("bad", ["weekly", "monthly", "weekly:", "weekly:abc", "", ":1"])
    def test_invalid_spec(self, bad):
        with pytest.raises(ValueError):
            RebalanceSchedule.from_string(bad)

    def test_invalid_mode_propagates(self):
        with pytest.raises(ValueError, match="Invalid rebalance mode"):
            RebalanceSchedule.from_string("daily:1")

    def test_invalid_day_propagates(self):
        with pytest.raises(ValueError, match="monthly_day"):
            RebalanceSchedule.from_string("monthly:29")


class TestFromConfig:
    def test_empty_uses_defaults(self):
        s = RebalanceSchedule.from_config(None)
        assert s == RebalanceSchedule("weekly", 1)

    def test_empty_dict_uses_defaults(self):
        s = RebalanceSchedule.from_config({})
        assert s == RebalanceSchedule("weekly", 1)

    def test_weekly_with_day(self):
        s = RebalanceSchedule.from_config({"mode": "weekly", "weekly_day": 5})
        assert s == RebalanceSchedule("weekly", 5)

    def test_monthly_with_day(self):
        s = RebalanceSchedule.from_config({"mode": "monthly", "monthly_day": 15})
        assert s == RebalanceSchedule("monthly", 15)

    def test_monthly_uses_monthly_day_not_weekly_day(self):
        s = RebalanceSchedule.from_config({"mode": "monthly", "monthly_day": 10, "weekly_day": 3})
        assert s.mode == "monthly"
        assert s.day == 10


# ---------------------------------------------------------------------------
# Core predicate — weekly mode
# ---------------------------------------------------------------------------


class TestWeeklyMatches:
    def test_monday_on_monday_schedule(self):
        # 2024-01-08 = Monday, trading day
        s = RebalanceSchedule("weekly", 1)
        pred = make_predicate(["2024-01-05", "2024-01-08", "2024-01-09"])
        assert s.matches(date(2024, 1, 8), pred) is True

    def test_tuesday_after_monday_on_monday_schedule(self):
        # Monday already taken; Tuesday should NOT rebalance
        s = RebalanceSchedule("weekly", 1)
        pred = make_predicate(["2024-01-08", "2024-01-09"])
        assert s.matches(date(2024, 1, 9), pred) is False

    def test_friday_in_same_week_on_monday_schedule(self):
        s = RebalanceSchedule("weekly", 1)
        pred = make_predicate(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"])
        assert s.matches(date(2024, 1, 12), pred) is False

    def test_deferred_when_monday_is_holiday(self):
        # 2024-01-01 (Mon, New Year) non-trading; 2024-01-02 (Tue) first trading day
        s = RebalanceSchedule("weekly", 1)
        pred = make_predicate(["2024-01-02", "2024-01-03"])
        assert s.matches(date(2024, 1, 2), pred) is True
        assert s.matches(date(2024, 1, 3), pred) is False

    def test_friday_schedule_normal_friday(self):
        # 2024-01-12 = Friday, trading
        # Include the previous Friday (2024-01-05) so weekly:5's defer mechanism
        # treats prev Fri (not Mon 01-08) as the prev-week rebalance.
        s = RebalanceSchedule("weekly", 5)
        pred = make_predicate(
            [
                "2024-01-05",  # prev Friday, trading
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
            ]
        )
        assert s.matches(date(2024, 1, 12), pred) is True
        # Monday of same week is NOT a Friday rebalance day (prev Fri consumed it)
        assert s.matches(date(2024, 1, 8), pred) is False

    def test_friday_schedule_deferred_to_next_monday(self):
        # Suppose Fri 2024-02-09 is holiday; next trading day = Mon 2024-02-12
        s = RebalanceSchedule("weekly", 5)
        pred = make_predicate(["2024-02-08", "2024-02-12", "2024-02-13"])
        assert s.matches(date(2024, 2, 12), pred) is True

    def test_cross_boundary_double_trigger_weekly(self):
        # Week 1: Fri 02-09 holiday → defers to Mon 02-12 (trading)
        # Week 2: Fri 02-16 trading
        # Both Mon 02-12 AND Fri 02-16 should fire under weekly:5 (no dedup).
        s = RebalanceSchedule("weekly", 5)
        pred = make_predicate(["2024-02-08", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16"])
        assert s.matches(date(2024, 2, 12), pred) is True
        assert s.matches(date(2024, 2, 16), pred) is True

    def test_tuesday_when_monday_holiday(self):
        # weekly:2 (Tue), Mon non-trading → Tue is exact target (not deferred)
        s = RebalanceSchedule("weekly", 2)
        pred = make_predicate(["2024-01-02", "2024-01-03"])
        assert s.matches(date(2024, 1, 2), pred) is True

    def test_non_trading_day_returns_false(self):
        s = RebalanceSchedule("weekly", 1)
        pred = make_predicate(["2024-01-08"])
        # 2024-01-09 not in calendar
        assert s.matches(date(2024, 1, 9), pred) is False


# ---------------------------------------------------------------------------
# Core predicate — monthly mode
# ---------------------------------------------------------------------------


class TestMonthlyMatches:
    def test_first_of_month_trading(self):
        s = RebalanceSchedule("monthly", 1)
        # 2024-02-01 is Thursday
        pred = make_predicate(["2024-02-01", "2024-02-02"])
        assert s.matches(date(2024, 2, 1), pred) is True

    def test_first_of_month_holiday_deferred(self):
        # 2024-01-01 (Mon, New Year) non-trading; 2024-01-02 (Tue) first trading day
        s = RebalanceSchedule("monthly", 1)
        pred = make_predicate(["2024-01-02", "2024-01-03"])
        assert s.matches(date(2024, 1, 2), pred) is True
        assert s.matches(date(2024, 1, 3), pred) is False

    def test_mid_month_not_target(self):
        s = RebalanceSchedule("monthly", 1)
        pred = make_predicate(["2024-02-01", "2024-02-05", "2024-02-15"])
        assert s.matches(date(2024, 2, 5), pred) is False
        assert s.matches(date(2024, 2, 15), pred) is False

    def test_day_15_monthly(self):
        s = RebalanceSchedule("monthly", 15)
        pred = make_predicate(["2024-02-15", "2024-02-16"])
        assert s.matches(date(2024, 2, 15), pred) is True

    def test_day_28_normal(self):
        s = RebalanceSchedule("monthly", 28)
        # 2024-03-28 = Thursday
        pred = make_predicate(["2024-03-28", "2024-03-29"])
        assert s.matches(date(2024, 3, 28), pred) is True

    def test_day_28_february_weekend_defers_to_march(self):
        # 2026-02-28 is Saturday; 2026-03-01 Sunday → both non-trading.
        # First trading day on/after 2026-02-28 is 2026-03-02 (Monday).
        s = RebalanceSchedule("monthly", 28)
        pred = make_predicate(["2026-02-27", "2026-03-02", "2026-03-03"])
        assert s.matches(date(2026, 3, 2), pred) is True
        assert s.matches(date(2026, 3, 3), pred) is False

    def test_cross_boundary_double_trigger_monthly(self):
        # Feb-28 weekend → defers to Mar-2 (Monday). Mar-28 (Sat) → defers to Mar-30.
        # Both Mar-2 AND Mar-30 are rebalance days (no dedup).
        s = RebalanceSchedule("monthly", 28)
        pred = make_predicate(
            [
                "2026-02-27",
                "2026-03-02",
                "2026-03-27",
                "2026-03-30",
            ]
        )
        assert s.matches(date(2026, 3, 2), pred) is True
        assert s.matches(date(2026, 3, 30), pred) is True
        # Mar-27 (Fri) is NOT a rebalance day — Mar-28 target hasn't passed yet
        assert s.matches(date(2026, 3, 27), pred) is False

    def test_year_boundary_wrap(self):
        # monthly:28, today 2024-01-15 → latest target ≤ today is 2023-12-28.
        s = RebalanceSchedule("monthly", 28)
        pred = make_predicate(["2023-12-28", "2024-01-15"])
        assert s.matches(date(2024, 1, 15), pred) is False
        assert s.matches(date(2023, 12, 28), pred) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_today_is_target_and_trading(self):
        s = RebalanceSchedule("weekly", 3)  # Wednesday
        # 2024-01-10 = Wednesday
        pred = make_predicate(["2024-01-10"])
        assert s.matches(date(2024, 1, 10), pred) is True

    def test_sparse_calendar_no_match(self):
        s = RebalanceSchedule("weekly", 1)
        pred = lambda d: False  # noqa: E731
        assert s.matches(date(2024, 1, 8), pred) is False
