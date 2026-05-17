"""Shared pytest fixtures for PanGu tests."""

from __future__ import annotations

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
def fake_stock_pool() -> FakeStockPool:
    """FakeStockPool with a fixed 3-stock universe."""
    return FakeStockPool(["600519", "000858", "300750"])
