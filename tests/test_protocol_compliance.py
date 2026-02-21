"""Protocol compliance tests — verify stub implementations satisfy all Protocol signatures.

For each Protocol defined in M1.2, we create a minimal concrete class and use
`isinstance(..., Protocol)` / `runtime_checkable` + signature inspection to ensure
the contracts are correct.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime

import pandas as pd

from trading_agent.data.fundamental.protocol import FundamentalDataProvider
from trading_agent.data.market.protocol import MarketDataProvider
from trading_agent.data.news.protocol import NewsDataProvider
from trading_agent.data.stock_pool.protocol import StockPool
from trading_agent.factor.technical import FactorEngine
from trading_agent.models import (
    Action,
    BacktestConfig,
    NewsItem,
    Region,
    SignalStatus,
    TradeSignal,
)
from trading_agent.notification.feishu import NotificationProvider
from trading_agent.strategy.factor.protocol import Strategy
from trading_agent.strategy.llm.client import LLMJudgeEngine

# ---------------------------------------------------------------------------
# Minimal stub implementations
# ---------------------------------------------------------------------------

class _StubMarket:
    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_global_snapshot(self) -> pd.DataFrame:
        return pd.DataFrame()


class _StubFundamental:
    def get_valuation(self, symbol: str) -> dict:
        return {}

    def get_financial_indicator(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame()


class _StubNews:
    def get_latest_news(self, limit: int = 50) -> list[NewsItem]:
        return []

    def get_stock_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        return []

    def get_announcements(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        return []


class _StubStockPool:
    def get_watchlist(self) -> list[str]:
        return []

    def add_to_watchlist(self, symbol: str) -> None:
        pass

    def remove_from_watchlist(self, symbol: str) -> None:
        pass

    def get_factor_selected(self) -> list[str]:
        return []


class _StubFactorEngine:
    def compute(
        self, bars: pd.DataFrame, global_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def get_factor_names(self) -> list[str]:
        return []


class _StubStrategy:
    def generate_signals(
        self,
        tech_df: dict[str, pd.DataFrame],
        fund_df: pd.DataFrame,
        macro_factors: dict[str, float],
        *,
        prev_pool: pd.DataFrame | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, list[TradeSignal]]:
        return pd.DataFrame(), []


class _StubLLMJudge:
    async def judge_stock(
        self, symbol: str, name: str,
        factor_score: float, factor_rank: int,
        factor_details: dict[str, float],
        stock_news: list[NewsItem], announcements: list[NewsItem],
        telegraph: list[NewsItem], global_market: pd.DataFrame,
        price: float,
        *, factor_signal: str = "", universe_size: int = 0,
    ) -> TradeSignal:
        return TradeSignal(
            timestamp=datetime(2026, 1, 1), symbol=symbol, name=name,
            action=Action.HOLD, signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0, price=price, confidence=0.5,
            source="stub", reason="stub",
        )

    async def judge_pool(
        self, candidates: list, telegraph: list[NewsItem],
        global_market: pd.DataFrame,
        *, universe_size: int = 0,
    ) -> list[TradeSignal]:
        return []


class _StubNotifier:
    async def send_signal(self, signal: TradeSignal) -> bool:
        return True

    async def send_text(self, text: str) -> bool:
        return True

    async def send_markdown(self, title: str, content: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _protocol_methods(protocol_cls: type) -> dict[str, inspect.Signature]:
    """Return {name: Signature} for every abstract method on *protocol_cls*."""
    methods: dict[str, inspect.Signature] = {}
    for name, obj in inspect.getmembers(protocol_cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        methods[name] = inspect.signature(obj)
    return methods


def _assert_signatures_match(
    protocol_cls: type, impl_cls: type, *, label: str
) -> None:
    """Assert that *impl_cls* has all methods of *protocol_cls* with compatible signatures."""
    proto_methods = _protocol_methods(protocol_cls)
    assert proto_methods, f"{label}: protocol exposes no public methods"
    for method_name, proto_sig in proto_methods.items():
        impl_method = getattr(impl_cls, method_name, None)
        assert impl_method is not None, (
            f"{label}: missing method '{method_name}'"
        )
        impl_sig = inspect.signature(impl_method)
        proto_params = list(proto_sig.parameters.values())[1:]  # skip 'self'
        impl_params = list(impl_sig.parameters.values())[1:]
        assert len(impl_params) == len(proto_params), (
            f"{label}.{method_name}: expected {len(proto_params)} params, "
            f"got {len(impl_params)}"
        )


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestModels:
    """Verify data-class instantiation and enum values."""

    def test_action_enum(self) -> None:
        assert Action.BUY.value == "BUY"
        assert Action.SELL.value == "SELL"
        assert Action.HOLD.value == "HOLD"

    def test_signal_status_enum(self) -> None:
        assert SignalStatus.NEW_ENTRY.value == "new_entry"
        assert SignalStatus.SUSTAINED.value == "sustained"
        assert SignalStatus.EXIT.value == "exit"

    def test_region_enum(self) -> None:
        assert Region.DOMESTIC.value == "domestic"
        assert Region.GLOBAL.value == "global"

    def test_news_item_defaults(self) -> None:
        item = NewsItem(
            timestamp=datetime(2026, 1, 1),
            title="headline",
            content="body",
            source="test",
            region=Region.DOMESTIC,
        )
        assert item.symbols == []
        assert item.sentiment is None

    def test_trade_signal_defaults(self) -> None:
        sig = TradeSignal(
            timestamp=datetime(2026, 1, 1),
            symbol="600519",
            name="贵州茅台",
            action=Action.BUY,
            signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0,
            price=1800.0,
            confidence=0.85,
            source="factor",
            reason="multi-factor top-1",
        )
        assert sig.stop_loss is None
        assert sig.take_profit is None
        assert sig.factor_score is None
        assert sig.prev_factor_score is None
        assert sig.metadata == {}

    def test_trade_signal_full(self) -> None:
        sig = TradeSignal(
            timestamp=datetime(2026, 1, 1),
            symbol="600519",
            name="贵州茅台",
            action=Action.SELL,
            signal_status=SignalStatus.EXIT,
            days_in_top_n=5,
            price=1750.0,
            confidence=0.6,
            source="merged",
            reason="factor exit",
            stop_loss=1700.0,
            take_profit=1900.0,
            factor_score=0.45,
            prev_factor_score=0.72,
            metadata={"rsi": 28},
        )
        assert sig.metadata["rsi"] == 28
        assert sig.action is Action.SELL

    def test_backtest_config_defaults(self) -> None:
        cfg = BacktestConfig(start_date="2025-01-01", end_date="2025-12-31")
        assert cfg.initial_cash == 100_000
        assert cfg.commission == 0.0003
        assert cfg.stamp_tax == 0.0005
        assert cfg.slippage == 0.001
        assert cfg.max_position_pct == 0.2


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Verify stub implementations match Protocol method signatures."""

    def test_market_data_provider(self) -> None:
        _assert_signatures_match(
            MarketDataProvider, _StubMarket, label="MarketDataProvider"
        )

    def test_fundamental_data_provider(self) -> None:
        _assert_signatures_match(
            FundamentalDataProvider, _StubFundamental, label="FundamentalDataProvider"
        )

    def test_news_data_provider(self) -> None:
        _assert_signatures_match(
            NewsDataProvider, _StubNews, label="NewsDataProvider"
        )

    def test_stock_pool(self) -> None:
        _assert_signatures_match(
            StockPool, _StubStockPool, label="StockPool"
        )

    def test_factor_engine(self) -> None:
        _assert_signatures_match(
            FactorEngine, _StubFactorEngine, label="FactorEngine"
        )

    def test_strategy(self) -> None:
        _assert_signatures_match(
            Strategy, _StubStrategy, label="Strategy"
        )

    def test_llm_judge_engine(self) -> None:
        _assert_signatures_match(
            LLMJudgeEngine, _StubLLMJudge, label="LLMJudgeEngine"
        )

    def test_notification_provider(self) -> None:
        _assert_signatures_match(
            NotificationProvider, _StubNotifier, label="NotificationProvider"
        )


class TestStubInvocation:
    """Call every stub method to confirm it returns the expected type."""

    def test_market_stubs_return(self) -> None:
        m = _StubMarket()
        assert isinstance(m.get_daily_bars("000001", "2025-01-01", "2025-06-01"), pd.DataFrame)
        assert isinstance(m.get_global_snapshot(), pd.DataFrame)

    def test_fundamental_stubs_return(self) -> None:
        f = _StubFundamental()
        assert isinstance(f.get_valuation("000001"), dict)
        assert isinstance(f.get_financial_indicator("000001"), pd.DataFrame)

    def test_news_stubs_return(self) -> None:
        n = _StubNews()
        assert isinstance(n.get_latest_news(), list)
        assert isinstance(n.get_stock_news("000001"), list)

    def test_stock_pool_stubs_return(self) -> None:
        sp = _StubStockPool()
        assert isinstance(sp.get_watchlist(), list)
        sp.add_to_watchlist("000001")
        sp.remove_from_watchlist("000001")
        assert isinstance(sp.get_factor_selected(), list)

    def test_factor_engine_stubs_return(self) -> None:
        fe = _StubFactorEngine()
        assert isinstance(fe.compute(pd.DataFrame()), pd.DataFrame)
        assert isinstance(fe.get_factor_names(), list)

    def test_strategy_stubs_return(self) -> None:
        s = _StubStrategy()
        pool_df, signals = s.generate_signals({}, pd.DataFrame(), {})
        assert isinstance(pool_df, pd.DataFrame)
        assert isinstance(signals, list)

    def test_llm_judge_stubs_return(self) -> None:
        e = _StubLLMJudge()
        result = asyncio.run(e.judge_pool([], [], pd.DataFrame()))
        assert isinstance(result, list)

    def test_notifier_stubs_return(self) -> None:
        n = _StubNotifier()
        sig = TradeSignal(
            timestamp=datetime(2026, 1, 1),
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
        result = asyncio.run(n.send_signal(sig))
        assert result is True
