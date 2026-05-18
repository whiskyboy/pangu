"""Tests for stock_profiles storage CRUD.

The ``stock_profiles`` table stores the latest cninfo company profile per
symbol (single snapshot, no PIT history). It feeds the LLM rebalance prompt
as grounding context.
"""

from __future__ import annotations

import pytest

from pangu.data.storage import Database


@pytest.fixture()
def db() -> Database:
    d = Database(":memory:")
    d.init_tables()
    return d


# ---------------------------------------------------------------------------
# count_stock_profiles
# ---------------------------------------------------------------------------


class TestCountStockProfiles:
    def test_empty_db_returns_zero(self, db: Database) -> None:
        assert db.count_stock_profiles() == 0

    def test_counts_inserted_rows(self, db: Database) -> None:
        db.save_stock_profile("600519", {"name": "贵州茅台"})
        db.save_stock_profile("000858", {"name": "五粮液"})
        assert db.count_stock_profiles() == 2

    def test_upsert_does_not_increase_count(self, db: Database) -> None:
        db.save_stock_profile("600519", {"name": "贵州茅台"})
        db.save_stock_profile("600519", {"name": "茅台 v2"})
        assert db.count_stock_profiles() == 1


# ---------------------------------------------------------------------------
# save_stock_profile + load_stock_profile
# ---------------------------------------------------------------------------


class TestSingleProfileCrud:
    def test_save_and_load_full_profile(self, db: Database) -> None:
        db.save_stock_profile(
            "600519",
            {
                "name": "贵州茅台",
                "full_name": "贵州茅台酒股份有限公司",
                "sector": "酒、饮料和精制茶制造业",
                "list_date": "2001-08-27",
                "main_business": "茅台酒及系列酒的生产与销售",
                "registered_area": "贵州省仁怀市",
            },
        )
        p = db.load_stock_profile("600519")
        assert p is not None
        assert p["symbol"] == "600519"
        assert p["name"] == "贵州茅台"
        assert p["full_name"] == "贵州茅台酒股份有限公司"
        assert p["sector"] == "酒、饮料和精制茶制造业"
        assert p["list_date"] == "2001-08-27"
        assert p["main_business"] == "茅台酒及系列酒的生产与销售"
        assert p["registered_area"] == "贵州省仁怀市"
        assert p["updated_at"]  # timestamp stamped

    def test_load_missing_returns_none(self, db: Database) -> None:
        assert db.load_stock_profile("999999") is None

    def test_minimal_profile_uses_empty_defaults(self, db: Database) -> None:
        """Partial profile fields default to empty string, not NULL."""
        db.save_stock_profile("300750", {})
        p = db.load_stock_profile("300750")
        assert p is not None
        assert p["name"] == ""
        assert p["full_name"] == ""
        assert p["sector"] == ""
        assert p["list_date"] == ""
        assert p["main_business"] == ""
        assert p["registered_area"] == ""

    def test_upsert_overwrites_all_fields(self, db: Database) -> None:
        """save_stock_profile is full upsert: missing keys reset to empty."""
        db.save_stock_profile(
            "600519",
            {"name": "茅台", "sector": "食品饮料", "main_business": "旧描述"},
        )
        # Re-save with only name → sector / main_business reset to ""
        db.save_stock_profile("600519", {"name": "茅台 v2"})
        p = db.load_stock_profile("600519")
        assert p is not None
        assert p["name"] == "茅台 v2"
        assert p["sector"] == ""
        assert p["main_business"] == ""

    def test_none_values_become_empty_strings(self, db: Database) -> None:
        """Defensive: None field values from upstream become empty strings."""
        db.save_stock_profile("600519", {"name": None, "sector": None})
        p = db.load_stock_profile("600519")
        assert p is not None
        assert p["name"] == ""
        assert p["sector"] == ""


# ---------------------------------------------------------------------------
# save_stock_profiles_batch + load_all_stock_profiles
# ---------------------------------------------------------------------------


class TestBatchProfileCrud:
    def test_save_batch_and_load_all(self, db: Database) -> None:
        n = db.save_stock_profiles_batch(
            [
                {"symbol": "600519", "name": "贵州茅台", "sector": "酒类"},
                {"symbol": "000858", "name": "五粮液", "sector": "酒类"},
                {"symbol": "300750", "name": "宁德时代"},  # minimal
            ]
        )
        assert n == 3

        all_p = db.load_all_stock_profiles()
        assert set(all_p.keys()) == {"600519", "000858", "300750"}
        assert all_p["600519"]["name"] == "贵州茅台"
        assert all_p["300750"]["sector"] == ""

    def test_empty_batch_returns_zero(self, db: Database) -> None:
        assert db.save_stock_profiles_batch([]) == 0
        assert db.load_all_stock_profiles() == {}

    def test_batch_upsert_replaces_existing(self, db: Database) -> None:
        db.save_stock_profile("600519", {"name": "old"})
        db.save_stock_profiles_batch(
            [
                {"symbol": "600519", "name": "new"},
                {"symbol": "000858", "name": "another"},
            ]
        )
        assert db.count_stock_profiles() == 2
        p = db.load_stock_profile("600519")
        assert p is not None and p["name"] == "new"

    def test_load_all_empty_db(self, db: Database) -> None:
        assert db.load_all_stock_profiles() == {}
