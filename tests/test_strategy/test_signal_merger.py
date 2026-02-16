"""Tests for SimpleSignalMerger (M1.5)."""

from __future__ import annotations

from datetime import datetime

from trading_agent.models import Action, SignalStatus, TradeSignal
from trading_agent.strategy.signal_merger import SimpleSignalMerger


def _sig(
    symbol: str = "600519",
    action: Action = Action.BUY,
    confidence: float = 0.8,
    source: str = "factor",
    reason: str = "test",
) -> TradeSignal:
    """Helper to build a minimal TradeSignal."""
    return TradeSignal(
        timestamp=datetime(2026, 2, 16, 10, 0),
        symbol=symbol,
        name=symbol,
        action=action,
        signal_status=SignalStatus.NEW_ENTRY,
        days_in_top_n=0,
        price=100.0,
        confidence=confidence,
        source=source,
        reason=reason,
    )


class TestWeightModes:
    """Verify correct weight mode selection based on news_availability."""

    def test_news_mode_metadata(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(source="factor")],
            [_sig(source="llm_event")],
            [],
            {"600519": True},
        )
        assert len(result) == 1
        assert result[0].metadata["weight_mode"] == "news"

    def test_announcement_mode_metadata(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(source="factor")],
            [_sig(source="llm_event")],
            [],
            {"600519": False},  # no news
        )
        assert len(result) == 1
        assert result[0].metadata["weight_mode"] == "announcement"

    def test_no_news_mode_metadata(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(source="factor")],
            [],
            [],
            {"600519": False},
        )
        assert len(result) == 1
        assert result[0].metadata["weight_mode"] == "no_news"


class TestResonance:
    """Factor + Event agree → confidence boost."""

    def test_buy_buy_resonance(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(action=Action.BUY, confidence=0.8, source="factor")],
            [_sig(action=Action.BUY, confidence=0.7, source="llm_event")],
            [],
            {"600519": True},
        )
        assert len(result) == 1
        sig = result[0]
        assert sig.action is Action.BUY
        assert sig.source == "merged"
        # Confidence should be boosted above the max input (0.8)
        assert sig.confidence > 0.8

    def test_sell_sell_resonance(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(action=Action.SELL, confidence=0.7, source="factor")],
            [_sig(action=Action.SELL, confidence=0.6, source="llm_event")],
            [],
            {"600519": True},
        )
        assert len(result) == 1
        sig = result[0]
        assert sig.action is Action.SELL
        assert sig.confidence > 0.7


class TestConflict:
    """Factor + Event disagree → forced HOLD."""

    def test_buy_sell_conflict(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(action=Action.BUY, confidence=0.9, source="factor")],
            [_sig(action=Action.SELL, confidence=0.9, source="llm_event")],
            [],
            {"600519": True},
        )
        assert len(result) == 1
        sig = result[0]
        assert sig.action is Action.HOLD
        assert sig.metadata["merged_score"] == 0.5

    def test_sell_buy_conflict(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(action=Action.SELL, confidence=0.8, source="factor")],
            [_sig(action=Action.BUY, confidence=0.8, source="llm_event")],
            [],
            {"600519": True},
        )
        assert len(result) == 1
        assert result[0].action is Action.HOLD


class TestThresholds:
    """BUY > 0.7, SELL < 0.3, else HOLD."""

    def test_score_above_buy_threshold(self) -> None:
        m = SimpleSignalMerger()
        # Strong BUY from factor only (no event, no_news mode)
        result = m.merge(
            [_sig(action=Action.BUY, confidence=1.0, source="factor")],
            [],
            [],
            {},
        )
        assert len(result) == 1
        assert result[0].action is Action.BUY

    def test_score_below_sell_threshold(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(action=Action.SELL, confidence=1.0, source="factor")],
            [],
            [],
            {},
        )
        assert len(result) == 1
        assert result[0].action is Action.SELL

    def test_moderate_confidence_hold(self) -> None:
        m = SimpleSignalMerger()
        # Low-confidence BUY → merged score near 0.5 → HOLD
        result = m.merge(
            [_sig(action=Action.BUY, confidence=0.2, source="factor")],
            [],
            [],
            {},
        )
        assert len(result) == 1
        assert result[0].action is Action.HOLD

    def test_custom_thresholds(self) -> None:
        m = SimpleSignalMerger(buy_threshold=0.55, sell_threshold=0.45)
        result = m.merge(
            [_sig(action=Action.BUY, confidence=0.3, source="factor")],
            [],
            [],
            {},
        )
        assert len(result) == 1
        assert result[0].action is Action.BUY


class TestSingleSource:
    """Only one signal source present."""

    def test_factor_only(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(action=Action.BUY, confidence=0.9, source="factor")],
            [],
            [],
            {},
        )
        assert len(result) == 1
        assert result[0].source == "merged"

    def test_event_only(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [],
            [_sig(action=Action.BUY, confidence=0.9, source="llm_event")],
            [],
            {"600519": True},
        )
        assert len(result) == 1

    def test_anomaly_only(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [],
            [],
            [_sig(action=Action.BUY, confidence=0.9, source="anomaly")],
            {},
        )
        assert len(result) == 1


class TestNoSignals:
    """Edge case: empty inputs."""

    def test_all_empty(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge([], [], [], {})
        assert result == []


class TestMultipleSymbols:
    """Signals for different symbols are merged independently."""

    def test_two_symbols(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [
                _sig(symbol="600519", action=Action.BUY, confidence=0.9, source="factor"),
                _sig(symbol="000858", action=Action.SELL, confidence=0.8, source="factor"),
            ],
            [],
            [],
            {},
        )
        assert len(result) == 2
        by_sym = {s.symbol: s for s in result}
        assert by_sym["600519"].action is Action.BUY
        assert by_sym["000858"].action is Action.SELL


class TestMultiDay:
    """Signals on different dates for the same symbol should not collapse."""

    def test_same_symbol_different_days(self) -> None:
        m = SimpleSignalMerger()
        sig_day1 = TradeSignal(
            timestamp=datetime(2026, 2, 16, 10, 0),
            symbol="600519", name="600519",
            action=Action.BUY, signal_status=SignalStatus.NEW_ENTRY,
            days_in_top_n=0, price=100.0, confidence=0.9,
            source="factor", reason="day1",
        )
        sig_day2 = TradeSignal(
            timestamp=datetime(2026, 2, 17, 10, 0),
            symbol="600519", name="600519",
            action=Action.SELL, signal_status=SignalStatus.EXIT,
            days_in_top_n=1, price=95.0, confidence=0.8,
            source="factor", reason="day2",
        )
        result = m.merge([sig_day1, sig_day2], [], [], {})
        assert len(result) == 2


class TestDeduplication:
    """Same symbol + action + source + date should be deduplicated."""

    def test_duplicate_factor_signals(self) -> None:
        m = SimpleSignalMerger()
        sig = _sig(symbol="600519", action=Action.BUY, source="factor")
        result = m.merge([sig, sig], [], [], {})
        assert len(result) == 1


class TestMergedSignalFields:
    """Verify the merged signal has correct metadata and reason."""

    def test_reason_format(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge(
            [_sig(source="factor")],
            [_sig(source="llm_event")],
            [],
            {"600519": True},
        )
        assert "factor(BUY)" in result[0].reason
        assert "event(BUY)" in result[0].reason

    def test_source_is_merged(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge([_sig(source="factor")], [], [], {})
        assert result[0].source == "merged"

    def test_factor_score_is_merged_score(self) -> None:
        m = SimpleSignalMerger()
        result = m.merge([_sig(source="factor")], [], [], {})
        assert result[0].factor_score is not None
        assert 0.0 <= result[0].factor_score <= 1.0

    def test_confidence_clamped(self) -> None:
        m = SimpleSignalMerger()
        # Very high confidence + boost should not exceed 1.0
        result = m.merge(
            [_sig(confidence=0.99, source="factor")],
            [_sig(confidence=0.99, source="llm_event")],
            [],
            {"600519": True},
        )
        assert result[0].confidence <= 1.0

        # Very low confidence in conflict should not go below 0.0
        result2 = m.merge(
            [_sig(action=Action.BUY, confidence=0.1, source="factor")],
            [_sig(action=Action.SELL, confidence=0.1, source="llm_event")],
            [],
            {"600519": True},
        )
        assert result2[0].confidence >= 0.0
