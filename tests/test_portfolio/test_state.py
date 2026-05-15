"""Tests for PortfolioState — JSON read/write of the virtual portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pangu.portfolio import Portfolio, PortfolioState


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "target_portfolio.json"


class TestPortfolioStateLoadSave:
    def test_load_missing_file_returns_none(self, state_path):
        ps = PortfolioState(state_path)
        assert ps.load() is None

    def test_save_then_load_roundtrip(self, state_path):
        ps = PortfolioState(state_path)
        p = Portfolio(
            date="2024-12-09",
            symbols=["600519", "300750"],
            scores={"600519": 0.92, "300750": 0.81},
            ranks={"600519": 1, "300750": 5},
        )
        ps.save(p)

        loaded = ps.load()
        assert loaded is not None
        assert loaded.date == "2024-12-09"
        assert loaded.symbols == ["600519", "300750"]
        assert loaded.scores["600519"] == pytest.approx(0.92)
        assert loaded.ranks["300750"] == 5

    def test_save_creates_parent_dir(self, tmp_path: Path):
        deep = tmp_path / "deep" / "nested" / "p.json"
        ps = PortfolioState(deep)
        ps.save(Portfolio(date="2025-01-01", symbols=["600519"]))
        assert deep.exists()

    def test_save_is_atomic_no_partial_tmp(self, state_path):
        ps = PortfolioState(state_path)
        ps.save(Portfolio(date="2025-01-01", symbols=["600519"]))
        # No leftover tmp file in the directory
        for f in state_path.parent.iterdir():
            assert not f.name.startswith(f".{state_path.name}.")

    def test_save_overwrites_previous(self, state_path):
        ps = PortfolioState(state_path)
        ps.save(Portfolio(date="2025-01-01", symbols=["600519"]))
        ps.save(Portfolio(date="2025-01-08", symbols=["300750", "601318"]))

        loaded = ps.load()
        assert loaded is not None
        assert loaded.date == "2025-01-08"
        assert loaded.symbols == ["300750", "601318"]

    def test_load_corrupt_json_returns_none(self, state_path):
        state_path.write_text("{ this is not valid json", encoding="utf-8")
        ps = PortfolioState(state_path)
        assert ps.load() is None

    def test_load_schema_mismatch_returns_none(self, state_path):
        state_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        ps = PortfolioState(state_path)
        assert ps.load() is None

    def test_load_with_missing_optional_fields(self, state_path):
        state_path.write_text(
            json.dumps({"date": "2025-02-01", "symbols": ["600519"]}),
            encoding="utf-8",
        )
        ps = PortfolioState(state_path)
        loaded = ps.load()
        assert loaded is not None
        assert loaded.symbols == ["600519"]
        assert loaded.scores == {}
        assert loaded.ranks == {}

    def test_clear_removes_file(self, state_path):
        ps = PortfolioState(state_path)
        ps.save(Portfolio(date="2025-01-01", symbols=["600519"]))
        assert state_path.exists()
        ps.clear()
        assert not state_path.exists()
        assert ps.load() is None

    def test_clear_when_missing_is_noop(self, state_path):
        ps = PortfolioState(state_path)
        ps.clear()  # should not raise
        assert not state_path.exists()
