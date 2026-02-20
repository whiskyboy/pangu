"""Shared pytest fixtures for TradingAgent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import (
    FakeFundamentalDataProvider,
    FakeMarketDataProvider,
    FakeNewsDataProvider,
    FakeStockPool,
)


@pytest.fixture
def fake_market() -> FakeMarketDataProvider:
    return FakeMarketDataProvider()


@pytest.fixture
def fake_fundamental() -> FakeFundamentalDataProvider:
    return FakeFundamentalDataProvider()


@pytest.fixture
def fake_news() -> FakeNewsDataProvider:
    return FakeNewsDataProvider()


@pytest.fixture
def fake_stock_pool(tmp_path: Path) -> FakeStockPool:
    """FakeStockPool backed by a temporary watchlist YAML."""
    yaml_content = """watchlist:
  - symbol: "600519"
    name: "贵州茅台"
    sector: "白酒"
  - symbol: "000858"
    name: "五粮液"
    sector: "白酒"
  - symbol: "300750"
    name: "宁德时代"
    sector: "新能源"
"""
    p = tmp_path / "watchlist.yaml"  # type: ignore[operator]
    p.write_text(yaml_content)
    return FakeStockPool(watchlist_path=p)
