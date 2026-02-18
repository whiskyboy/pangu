"""Tests for PriceVolumeAnomalyDetector — M3.5."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_agent.models import Action
from trading_agent.strategy.anomaly_detector import PriceVolumeAnomalyDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SECTOR_MAPPING = [
    {"source": "S&P500", "a_share_sectors": ["大盘蓝筹", "外贸"], "sentiment_direction": "same"},
    {"source": "NASDAQ", "a_share_sectors": ["科技", "半导体"], "sentiment_direction": "same"},
    {"source": "恒生指数", "a_share_sectors": ["AH股"], "sentiment_direction": "same"},
    {"source": "COMEX黄金", "a_share_sectors": ["黄金", "有色金属"], "sentiment_direction": "same"},
    {"source": "WTI原油", "a_share_sectors": ["石油", "化工", "航空"], "sentiment_direction": "same"},
]


@pytest.fixture
def detector() -> PriceVolumeAnomalyDetector:
    return PriceVolumeAnomalyDetector(
        volume_ratio_threshold=3.0,
        price_change_threshold=5.0,
        global_change_threshold=2.0,
        sector_mapping=_SECTOR_MAPPING,
    )


def _make_a_share_quotes(**overrides) -> pd.DataFrame:
    """Build a single-row A-share quote DataFrame."""
    row = {
        "symbol": "601899",
        "name": "紫金矿业",
        "price": 38.0,
        "change_pct": 1.0,
        "volume_ratio": 1.5,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _make_global_quotes(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A-share volume spike
# ---------------------------------------------------------------------------


class TestVolumeSpike:
    def test_volume_ratio_above_threshold(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(volume_ratio=4.0, change_pct=2.0)
        signals = detector.detect(quotes)
        vol_signals = [s for s in signals if "volume_ratio" in s.reason]
        assert len(vol_signals) == 1
        assert vol_signals[0].action == Action.BUY
        assert vol_signals[0].source == "anomaly"

    def test_volume_ratio_below_threshold(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(volume_ratio=2.5)
        signals = detector.detect(quotes)
        vol_signals = [s for s in signals if "volume_ratio" in s.reason]
        assert len(vol_signals) == 0

    def test_volume_spike_sell_on_negative(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(volume_ratio=5.0, change_pct=-3.0)
        signals = detector.detect(quotes)
        vol_signals = [s for s in signals if "volume_ratio" in s.reason]
        assert len(vol_signals) == 1
        assert vol_signals[0].action == Action.SELL

    def test_confidence_scales_with_ratio(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(volume_ratio=8.0)
        signals = detector.detect(quotes)
        vol_signals = [s for s in signals if "volume_ratio" in s.reason]
        assert vol_signals[0].confidence == pytest.approx(0.8)

    def test_missing_volume_ratio_column(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = pd.DataFrame([{"symbol": "601899", "change_pct": 1.0}])
        signals = detector.detect(quotes)
        vol_signals = [s for s in signals if "volume_ratio" in s.reason]
        assert len(vol_signals) == 0


# ---------------------------------------------------------------------------
# A-share price swing
# ---------------------------------------------------------------------------


class TestPriceSwing:
    def test_positive_swing(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(change_pct=6.5)
        signals = detector.detect(quotes)
        price_signals = [s for s in signals if "price_change" in s.reason]
        assert len(price_signals) == 1
        assert price_signals[0].action == Action.BUY

    def test_negative_swing(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(change_pct=-7.0)
        signals = detector.detect(quotes)
        price_signals = [s for s in signals if "price_change" in s.reason]
        assert len(price_signals) == 1
        assert price_signals[0].action == Action.SELL

    def test_within_threshold(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(change_pct=3.0)
        signals = detector.detect(quotes)
        price_signals = [s for s in signals if "price_change" in s.reason]
        assert len(price_signals) == 0

    def test_exact_threshold_no_signal(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(change_pct=5.0)
        signals = detector.detect(quotes)
        price_signals = [s for s in signals if "price_change" in s.reason]
        assert len(price_signals) == 0  # > not >=


# ---------------------------------------------------------------------------
# Global anomaly → A-share sector
# ---------------------------------------------------------------------------


class TestGlobalAnomaly:
    def test_us_index_crash(self, detector: PriceVolumeAnomalyDetector) -> None:
        global_quotes = _make_global_quotes([
            {"symbol": "SPX", "change_pct": -3.5, "close": 5000},
        ])
        signals = detector.detect(pd.DataFrame(), global_quotes)
        assert len(signals) > 0
        assert all(s.symbol.startswith("SECTOR:") for s in signals)
        assert any("大盘蓝筹" in s.symbol for s in signals)
        assert signals[0].action == Action.SELL

    def test_commodity_surge(self, detector: PriceVolumeAnomalyDetector) -> None:
        global_quotes = _make_global_quotes([
            {"symbol": "GC", "change_pct": 4.0, "close": 2800},
        ])
        signals = detector.detect(pd.DataFrame(), global_quotes)
        sectors = [s.symbol for s in signals]
        assert "SECTOR:黄金" in sectors
        assert "SECTOR:有色金属" in sectors
        assert all(s.action == Action.BUY for s in signals)

    def test_below_global_threshold(self, detector: PriceVolumeAnomalyDetector) -> None:
        global_quotes = _make_global_quotes([
            {"symbol": "SPX", "change_pct": -1.5, "close": 5000},
        ])
        signals = detector.detect(pd.DataFrame(), global_quotes)
        assert len(signals) == 0

    def test_unmapped_symbol_ignored(self, detector: PriceVolumeAnomalyDetector) -> None:
        global_quotes = _make_global_quotes([
            {"symbol": "UNKNOWN", "change_pct": -5.0, "close": 100},
        ])
        signals = detector.detect(pd.DataFrame(), global_quotes)
        assert len(signals) == 0

    def test_no_sector_mapping(self) -> None:
        detector = PriceVolumeAnomalyDetector(sector_mapping=[])
        global_quotes = _make_global_quotes([
            {"symbol": "SPX", "change_pct": -5.0, "close": 5000},
        ])
        signals = detector.detect(pd.DataFrame(), global_quotes)
        assert len(signals) == 0

    def test_hk_index_anomaly(self, detector: PriceVolumeAnomalyDetector) -> None:
        global_quotes = _make_global_quotes([
            {"symbol": "HSI", "change_pct": -2.5, "close": 20000},
        ])
        signals = detector.detect(pd.DataFrame(), global_quotes)
        assert any("AH股" in s.symbol for s in signals)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_a_share_quotes(self, detector: PriceVolumeAnomalyDetector) -> None:
        signals = detector.detect(pd.DataFrame())
        assert signals == []

    def test_none_a_share_quotes(self, detector: PriceVolumeAnomalyDetector) -> None:
        signals = detector.detect(None)
        assert signals == []

    def test_both_anomalies_same_stock(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(volume_ratio=5.0, change_pct=8.0)
        signals = detector.detect(quotes)
        assert len(signals) == 2  # one volume, one price

    def test_multiple_stocks(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = pd.DataFrame([
            {"symbol": "A", "name": "StockA", "price": 10, "change_pct": 7.0, "volume_ratio": 1.0},
            {"symbol": "B", "name": "StockB", "price": 20, "change_pct": 1.0, "volume_ratio": 4.0},
            {"symbol": "C", "name": "StockC", "price": 30, "change_pct": 0.5, "volume_ratio": 1.0},
        ])
        signals = detector.detect(quotes)
        assert len(signals) == 2  # A: price swing, B: volume spike

    def test_nan_values_skipped(self, detector: PriceVolumeAnomalyDetector) -> None:
        quotes = _make_a_share_quotes(volume_ratio=float("nan"), change_pct=float("nan"))
        signals = detector.detect(quotes)
        assert len(signals) == 0

    def test_configurable_thresholds(self) -> None:
        detector = PriceVolumeAnomalyDetector(
            volume_ratio_threshold=2.0,
            price_change_threshold=3.0,
        )
        quotes = _make_a_share_quotes(volume_ratio=2.5, change_pct=3.5)
        signals = detector.detect(quotes)
        assert len(signals) == 2  # both trigger with lower thresholds

    def test_duplicate_sector_dedup(self, detector: PriceVolumeAnomalyDetector) -> None:
        """SPX and DJI both map to S&P500 → same sectors; only highest confidence kept."""
        global_quotes = _make_global_quotes([
            {"symbol": "SPX", "change_pct": -3.0, "close": 5000},
            {"symbol": "DJI", "change_pct": -2.5, "close": 35000},
        ])
        signals = detector.detect(pd.DataFrame(), global_quotes)
        sector_symbols = [s.symbol for s in signals]
        # Should have exactly 1 大盘蓝筹 and 1 外贸 (not 2 each)
        assert sector_symbols.count("SECTOR:大盘蓝筹") == 1
        assert sector_symbols.count("SECTOR:外贸") == 1
        # Higher confidence (SPX -3.0% > DJI -2.5%)
        for s in signals:
            assert s.confidence == pytest.approx(3.0 / 5.0)

    def test_invalid_string_values_skipped(self, detector: PriceVolumeAnomalyDetector) -> None:
        """Non-numeric strings should be skipped without crashing."""
        quotes = pd.DataFrame([
            {"symbol": "A", "name": "StockA", "price": 10, "change_pct": "N/A", "volume_ratio": "N/A"},
            {"symbol": "B", "name": "StockB", "price": 20, "change_pct": 8.0, "volume_ratio": 1.0},
        ])
        signals = detector.detect(quotes)
        # Only B should trigger (price swing), A skipped
        assert len(signals) == 1
        assert signals[0].symbol == "B"
