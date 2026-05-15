"""Integration tests for T4 generate_signals (LLM-TopkDropout path)."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from pangu.data.storage import Database
from pangu.portfolio import Portfolio, PortfolioState
from pangu.strategy.llm import RebalanceDecision

gs_module = importlib.import_module("pangu.tasks.generate_signals")
generate_signals = gs_module.generate_signals


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(str(tmp_path / "t4.db"))
    d.init_tables()
    # Trading calendar — set up so 2026-02-16 (Mon) is start of ISO week
    d.save_trading_calendar([
        "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",  # week 7
        "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",  # week 8
    ])
    # Insert a recent news row so freshness check passes silently
    d._conn.execute(
        "INSERT INTO news_items (timestamp, title, content, source, region, category) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("2026-02-16T07:00:00", "news", "content", "test", "domestic", "news"),
    )
    # Insert a recent bar so K-line freshness check passes
    d._conn.execute(
        "INSERT INTO daily_bars (symbol, date, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("600000", "2026-02-13", 10.0, 10.5, 9.8, 10.2, 1000000),
    )
    d._conn.commit()
    return d


@pytest.fixture
def portfolio_state(tmp_path) -> PortfolioState:
    return PortfolioState(tmp_path / "target_portfolio.json")


@pytest.fixture
def stock_pool() -> MagicMock:
    sp = MagicMock()
    universe = [f"60{i:04d}" for i in range(30)]
    sp.get_all_symbols.return_value = universe
    sp.get_watchlist.return_value = []
    # Stock metadata = symbol → StockMeta(name=symbol)
    from pangu.models import StockMeta
    sp.get_stock_metadata.return_value = {
        s: StockMeta(name=f"Stock-{s}") for s in universe
    }
    return sp


@pytest.fixture
def news_provider() -> MagicMock:
    np_ = MagicMock()
    np_.get_stock_news.return_value = []
    np_.get_announcements.return_value = []
    return np_


@pytest.fixture
def notif_manager() -> MagicMock:
    nm = MagicMock()
    nm.notify_signal = AsyncMock(return_value={"FakeChannel": True})
    nm.notify_text = AsyncMock(return_value={"FakeChannel": True})
    return nm


@pytest.fixture
def ml_strategy() -> MagicMock:
    """Mock MLScoringStrategy that returns deterministic pool_df."""
    ml = MagicMock()
    ml.top_n = 5
    ml.n_drop = 2
    ml.buy_candidate_size = 3
    ml.sell_candidate_size = 3
    # Build a synthetic pool: 20 stocks ranked 1..20
    symbols = [f"60{i:04d}" for i in range(20)]
    pool_df = pd.DataFrame({
        "symbol": symbols,
        "score": [1.0 - i * 0.05 for i in range(20)],
        "rank": list(range(1, 21)),
    })
    ml.score_pool.return_value = pool_df
    ml._scorer = MagicMock(window_id=1, n_models=1)

    def cold(_df):
        return pool_df["symbol"].head(5).tolist()
    ml.cold_start_portfolio.side_effect = cold

    def get_buy(_df, holdings):
        held = set(holdings)
        non_held = pool_df[~pool_df["symbol"].isin(held)]
        return non_held.sort_values("rank")["symbol"].head(3).tolist()
    ml.get_buy_candidate_pool.side_effect = get_buy

    def get_sell(_df, holdings):
        held_df = pool_df[pool_df["symbol"].isin(holdings)]
        # worst first
        return held_df.sort_values("rank", ascending=False)["symbol"].head(3).tolist()
    ml.get_sell_candidate_pool.side_effect = get_sell

    def fb_sells(sell_pool, excluded, _df, n):
        return [s for s in sell_pool if s not in excluded][:n]
    ml.fallback_sells.side_effect = fb_sells

    def fb_buys(buy_pool, excluded, _df, n):
        return [s for s in buy_pool if s not in excluded][:n]
    ml.fallback_buys.side_effect = fb_buys

    return ml


@pytest.fixture
def judge_engine() -> MagicMock:
    je = MagicMock()
    je.judge_rebalance = AsyncMock(return_value=RebalanceDecision())
    return je


@pytest.fixture
def components(db, portfolio_state, stock_pool, news_provider,
               notif_manager, ml_strategy, judge_engine):
    """Components with mocked dependencies."""
    c = MagicMock()
    c.db = db
    c.stock_pool = stock_pool
    c.news = news_provider
    c.notif_manager = notif_manager
    c.judge_engine = judge_engine
    c.ml_strategy = ml_strategy
    c.portfolio_state = portfolio_state
    # alert is awaited inside the task
    c.alert = AsyncMock()
    return c


def _patch_today(monkeypatch, today: str):
    """Pin both today_str references used by generate_signals."""
    import datetime as _dt
    monkeypatch.setattr(gs_module, "today_str", lambda: today)
    monkeypatch.setattr(gs_module, "_tz_now",
                        lambda: _dt.datetime.fromisoformat(f"{today}T08:15:00+08:00"))


def _stub_candidate_factors(monkeypatch):
    """Replace _build_candidate_factors so tests don't need real bars/funds."""
    def stub(c, today, candidates):
        return {}, pd.DataFrame()
    monkeypatch.setattr(gs_module, "_build_candidate_factors", stub)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNonRebalanceDay:
    @pytest.mark.asyncio
    async def test_skips_when_not_iso_week_start(self, components, monkeypatch):
        # 2026-02-17 (Tue) is mid-week
        _patch_today(monkeypatch, "2026-02-17")
        await generate_signals(components)

        # Should not call score_pool / judge_rebalance / save signals
        components.ml_strategy.score_pool.assert_not_called()
        components.judge_engine.judge_rebalance.assert_not_called()
        components.notif_manager.notify_signal.assert_not_called()
        # PortfolioState file untouched
        assert not components.portfolio_state.path.exists()


class TestColdStart:
    @pytest.mark.asyncio
    async def test_first_run_creates_portfolio_without_llm(
        self, components, monkeypatch,
    ):
        _patch_today(monkeypatch, "2026-02-16")  # Mon, ISO week 8 start
        _stub_candidate_factors(monkeypatch)

        await generate_signals(components)

        # LLM should NOT be invoked in cold start
        components.judge_engine.judge_rebalance.assert_not_called()
        # Portfolio file was written with top-5
        loaded = components.portfolio_state.load()
        assert loaded is not None
        assert loaded.date == "2026-02-16"
        assert len(loaded.symbols) == 5
        # Notification: 5 BUY signals pushed
        assert components.notif_manager.notify_signal.await_count == 5


class TestRebalance:
    @pytest.mark.asyncio
    async def test_llm_picks_used_and_fallback_fills_remainder(
        self, components, monkeypatch,
    ):
        # Pre-existing portfolio: holding 600015..600019 (ranks 16..20 — worst-ranked)
        held = ["600015", "600016", "600017", "600018", "600019"]
        components.portfolio_state.save(Portfolio(
            date="2026-02-09", symbols=held,
            scores={s: 0.5 for s in held},
            ranks={s: i + 16 for i, s in enumerate(held)},
        ))

        # LLM picks only 1 sell + 1 buy; remaining n_drop-1 from fallback
        from pangu.strategy.llm import DebateNotes, Pick
        components.judge_engine.judge_rebalance = AsyncMock(
            return_value=RebalanceDecision(
                sells=[Pick(symbol="600019", reason="LLM 卖", evidence="ev1")],
                buys=[Pick(symbol="600000", reason="LLM 买", evidence="ev2")],
                sell_debate=DebateNotes(bull="b", bear="x"),
                buy_debate=DebateNotes(bull="b2", bear="x2"),
            )
        )
        components.judge_engine.judge_rebalance.__qualname__ = "Mock"

        _patch_today(monkeypatch, "2026-02-16")
        _stub_candidate_factors(monkeypatch)

        await generate_signals(components)

        loaded = components.portfolio_state.load()
        assert loaded is not None
        # n_drop=2: 1 LLM-sell + 1 fallback-sell; 1 LLM-buy + 1 fallback-buy
        # New portfolio = (held - sells) ∪ buys, size still 5
        assert len(loaded.symbols) == 5
        # 600019 (LLM sell) is gone
        assert "600019" not in loaded.symbols
        # 600000 (LLM buy) is in
        assert "600000" in loaded.symbols
        # 4 signals pushed (2 sells + 2 buys)
        assert components.notif_manager.notify_signal.await_count == 4

    @pytest.mark.asyncio
    async def test_llm_failure_degenerates_to_classic_topkdropout(
        self, components, monkeypatch,
    ):
        held = ["600015", "600016", "600017", "600018", "600019"]
        components.portfolio_state.save(Portfolio(
            date="2026-02-09", symbols=held,
            scores={s: 0.5 for s in held},
            ranks={s: i + 16 for i, s in enumerate(held)},
        ))
        # LLM returns empty (failed) decision
        components.judge_engine.judge_rebalance = AsyncMock(
            return_value=RebalanceDecision(source="llm_failed")
        )

        _patch_today(monkeypatch, "2026-02-16")
        _stub_candidate_factors(monkeypatch)

        await generate_signals(components)

        loaded = components.portfolio_state.load()
        assert loaded is not None
        # n_drop=2 turnover should still occur via fallback
        assert len(loaded.symbols) == 5
        # Worst-ranked held (600019, 600018) should be sold by fallback
        # Best-ranked non-held (600000, 600001) should be bought by fallback
        assert "600019" not in loaded.symbols
        assert "600000" in loaded.symbols
        # 4 signals pushed (2 sell + 2 buy, all fallback-origin)
        assert components.notif_manager.notify_signal.await_count == 4
        # User was alerted about LLM failure
        components.alert.assert_awaited()

    @pytest.mark.asyncio
    async def test_no_op_when_holdings_become_universe(
        self, components, monkeypatch,
    ):
        # Hold every symbol in universe → no buy candidates
        all_syms = components.stock_pool.get_all_symbols()
        components.portfolio_state.save(Portfolio(
            date="2026-02-09",
            symbols=all_syms,
            scores={s: 0.5 for s in all_syms},
            ranks={s: 1 for s in all_syms},
        ))

        # Override get_sell/get_buy to return empty (matches behaviour when all held)
        components.ml_strategy.get_buy_candidate_pool.side_effect = (
            lambda _df, _hold: []
        )
        components.ml_strategy.get_sell_candidate_pool.side_effect = (
            lambda _df, _hold: []
        )

        _patch_today(monkeypatch, "2026-02-16")
        _stub_candidate_factors(monkeypatch)

        await generate_signals(components)

        components.judge_engine.judge_rebalance.assert_not_called()
        components.notif_manager.notify_signal.assert_not_called()


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_empty_score_pool_alerts_and_returns(
        self, components, monkeypatch,
    ):
        components.ml_strategy.score_pool.return_value = pd.DataFrame(
            columns=["symbol", "score", "rank"]
        )
        _patch_today(monkeypatch, "2026-02-16")

        await generate_signals(components)

        components.alert.assert_awaited()
        components.judge_engine.judge_rebalance.assert_not_called()
        components.notif_manager.notify_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_alerts_user(self, components, monkeypatch):
        components.ml_strategy.score_pool.side_effect = RuntimeError("boom")
        _patch_today(monkeypatch, "2026-02-16")

        # Should not raise — generate_signals catches and alerts
        await generate_signals(components)

        components.alert.assert_awaited()
