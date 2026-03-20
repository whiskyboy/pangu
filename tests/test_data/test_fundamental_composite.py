"""Tests for CompositeFundamentalProvider — gross_margin backfill."""

from __future__ import annotations

import pytest

from pangu.data.fundamental.composite import CompositeFundamentalProvider
from pangu.data.storage import Database
from pangu.utils import quarter_dates

# ---------------------------------------------------------------------------
# quarter_dates helper
# ---------------------------------------------------------------------------


class TestQuarterDates:
    def test_single_year(self) -> None:
        result = quarter_dates("2024-01-01", "2024-12-31")
        assert result == ["20240331", "20240630", "20240930", "20241231"]

    def test_partial_year(self) -> None:
        result = quarter_dates("2024-04-01", "2024-09-30")
        assert result == ["20240630", "20240930"]

    def test_cross_year(self) -> None:
        result = quarter_dates("2024-10-01", "2025-03-31")
        assert result == ["20241231", "20250331"]

    def test_empty_range(self) -> None:
        result = quarter_dates("2024-04-01", "2024-06-29")
        assert result == []

    def test_exact_boundary(self) -> None:
        result = quarter_dates("2024-03-31", "2024-03-31")
        assert result == ["20240331"]


# ---------------------------------------------------------------------------
# refresh_gross_margin
# ---------------------------------------------------------------------------


class _FakeGrossMarginProvider:
    """Minimal provider that supports fetch_gross_margin_batch."""

    def __init__(self, data: dict[str, dict[str, float]]) -> None:
        self._data = data  # quarter_date → {symbol: margin}

    def fetch_gross_margin_batch(self, quarter_date: str) -> dict[str, float]:
        return self._data.get(quarter_date, {})


class TestBackfillGrossMargin:
    @pytest.fixture()
    def db(self) -> Database:
        d = Database(":memory:")
        d.init_tables()
        return d

    def test_full_backfill(self, db: Database) -> None:
        import pandas as pd

        # Pre-populate fundamentals rows (simulating prior per-stock sync)
        db.save_fundamentals("600519", pd.DataFrame({
            "date": ["2024-03-31", "2024-06-30"], "pe_ttm": [30.0, 31.0],
        }))
        db.save_fundamentals("000001", pd.DataFrame({
            "date": ["2024-03-31"], "pe_ttm": [8.0],
        }))
        provider = _FakeGrossMarginProvider({
            "20240331": {"600519": 0.92, "000001": 0.30, "999999": 0.5},
            "20240630": {"600519": 0.91},
        })
        composite = CompositeFundamentalProvider(storage=db, providers=[provider])
        ok, fail = composite.refresh_gross_margin("2024-01-01", "2024-06-30")
        assert ok == 2
        assert fail == 0

        loaded = db.load_fundamentals("600519", "2024-03-31", "2024-06-30")
        assert len(loaded) == 2
        assert loaded.iloc[0]["gross_margin"] == pytest.approx(0.92)
        assert loaded.iloc[1]["gross_margin"] == pytest.approx(0.91)

        # 999999 not in DB — should NOT have been inserted
        orphan = db.load_fundamentals("999999", "2024-03-31", "2024-03-31")
        assert orphan.empty

    def test_narrow_range_only_fetches_covered_quarters(self, db: Database) -> None:
        import pandas as pd

        # Pre-populate rows for Q3/Q4 only
        db.save_fundamentals("600519", pd.DataFrame({
            "date": ["2024-09-30", "2024-12-31"], "pe_ttm": [30.0, 31.0],
        }))
        provider = _FakeGrossMarginProvider({
            "20240331": {"600519": 0.92},
            "20240630": {"600519": 0.91},
            "20240930": {"600519": 0.90},
            "20241231": {"600519": 0.89},
        })
        composite = CompositeFundamentalProvider(storage=db, providers=[provider])
        # Narrow date range covers only Q3 and Q4
        ok, fail = composite.refresh_gross_margin("2024-07-01", "2024-12-31")
        assert ok == 2
        assert fail == 0

        # Q1 and Q2 should NOT have been fetched
        q1 = db.load_fundamentals("600519", "2024-03-31", "2024-03-31")
        assert q1.empty

    def test_no_provider(self, db: Database) -> None:
        """No provider supports fetch_gross_margin_batch → returns (0, 0)."""
        composite = CompositeFundamentalProvider(storage=db, providers=[])
        ok, fail = composite.refresh_gross_margin("2024-01-01", "2024-12-31")
        assert ok == 0
        assert fail == 0

    def test_empty_response_counted_as_fail(self, db: Database) -> None:
        provider = _FakeGrossMarginProvider({"20240331": {}})
        composite = CompositeFundamentalProvider(storage=db, providers=[provider])
        ok, fail = composite.refresh_gross_margin("2024-01-01", "2024-03-31")
        assert ok == 0
        assert fail == 1

    def test_preserves_existing_data(self, db: Database) -> None:
        """Gross margin backfill should not overwrite PE/PB etc."""
        db.save_fundamentals("600519", __import__("pandas").DataFrame({
            "date": ["2024-03-31"],
            "pe_ttm": [25.0],
            "pb": [3.0],
        }))
        provider = _FakeGrossMarginProvider({"20240331": {"600519": 0.92}})
        composite = CompositeFundamentalProvider(storage=db, providers=[provider])
        composite.refresh_gross_margin("2024-01-01", "2024-03-31")

        loaded = db.load_fundamentals("600519", "2024-03-31", "2024-03-31")
        assert loaded.iloc[0]["pe_ttm"] == pytest.approx(25.0)
        assert loaded.iloc[0]["gross_margin"] == pytest.approx(0.92)
