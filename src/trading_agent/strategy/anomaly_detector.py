"""AnomalyDetector Protocol + implementations — PRD §4.3.3 / §6."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Protocol

import pandas as pd

from trading_agent.models import Action, SignalStatus, TradeSignal

logger = logging.getLogger(__name__)


class AnomalyDetector(Protocol):
    """Interface for detecting unusual market activity."""

    def detect(
        self,
        a_share_quotes: pd.DataFrame,
        global_quotes: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        """Detect anomalies: volume ratio >3, price change >5%, etc."""
        ...


# ---------------------------------------------------------------------------
# Symbol → source mapping for global anomaly detection
# ---------------------------------------------------------------------------

_GLOBAL_SYMBOL_MAP: dict[str, str] = {
    "SPX": "S&P500",
    "DJI": "S&P500",
    "IXIC": "NASDAQ",
    "HSI": "恒生指数",
    "HSTECH": "恒生科技指数",
    "GC": "COMEX黄金",
    "SI": "COMEX白银",
    "CL": "WTI原油",
    "HG": "LME铜",
    "FEF": "铁矿石",
    "NG": "NYMEX天然气",
    "CT": "NYBOT棉花",
}


# ---------------------------------------------------------------------------
# Real implementation
# ---------------------------------------------------------------------------


class PriceVolumeAnomalyDetector:
    """Detects A-share and global market anomalies.

    Pure computation — no DB, no API.  Receives DataFrames, returns signals.
    """

    def __init__(
        self,
        *,
        volume_ratio_threshold: float = 3.0,
        price_change_threshold: float = 5.0,
        global_change_threshold: float = 2.0,
        sector_mapping: list[dict] | None = None,
    ) -> None:
        self._vol_thresh = volume_ratio_threshold
        self._price_thresh = price_change_threshold
        self._global_thresh = global_change_threshold
        self._sector_mapping = sector_mapping or []

    def detect(
        self,
        a_share_quotes: pd.DataFrame,
        global_quotes: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        """Detect anomalies from realtime quote snapshots."""
        from trading_agent.tz import now as _now

        now = _now()
        signals: list[TradeSignal] = []

        # A-share anomalies
        if a_share_quotes is not None and not a_share_quotes.empty:
            signals.extend(self._detect_volume_spike(a_share_quotes, now))
            signals.extend(self._detect_price_swing(a_share_quotes, now))

        # Global anomalies → mapped to A-share sectors
        if global_quotes is not None and not global_quotes.empty:
            signals.extend(self._detect_global_anomaly(global_quotes, now))

        return signals

    # ------------------------------------------------------------------
    # A-share: volume spike
    # ------------------------------------------------------------------

    def _detect_volume_spike(
        self, df: pd.DataFrame, now: datetime
    ) -> list[TradeSignal]:
        if "volume_ratio" not in df.columns:
            return []

        signals: list[TradeSignal] = []
        for _, row in df.iterrows():
            vr = row.get("volume_ratio")
            if vr is None or (isinstance(vr, float) and math.isnan(vr)):
                continue
            try:
                vr_f = float(vr)
            except (ValueError, TypeError):
                continue
            if vr_f <= self._vol_thresh:
                continue

            try:
                chg = float(row.get("change_pct", 0) or 0)
            except (ValueError, TypeError):
                chg = 0.0
            action = Action.BUY if chg >= 0 else Action.SELL
            symbol = str(row.get("symbol", ""))
            try:
                price = float(row.get("price", row.get("close", 0)) or 0)
            except (ValueError, TypeError):
                price = 0.0

            signals.append(TradeSignal(
                timestamp=now,
                symbol=symbol,
                name=str(row.get("name", symbol)),
                action=action,
                signal_status=SignalStatus.NEW_ENTRY,
                days_in_top_n=0,
                price=price,
                confidence=min(vr_f / 10.0, 1.0),
                source="anomaly",
                reason=f"volume_ratio={vr_f:.1f} (>{self._vol_thresh})",
            ))
        return signals

    # ------------------------------------------------------------------
    # A-share: price swing
    # ------------------------------------------------------------------

    def _detect_price_swing(
        self, df: pd.DataFrame, now: datetime
    ) -> list[TradeSignal]:
        if "change_pct" not in df.columns:
            return []

        signals: list[TradeSignal] = []
        for _, row in df.iterrows():
            chg = row.get("change_pct")
            if chg is None or (isinstance(chg, float) and math.isnan(chg)):
                continue
            try:
                chg = float(chg)
            except (ValueError, TypeError):
                continue
            if abs(chg) <= self._price_thresh:
                continue

            action = Action.BUY if chg > 0 else Action.SELL
            symbol = str(row.get("symbol", ""))
            try:
                price = float(row.get("price", row.get("close", 0)) or 0)
            except (ValueError, TypeError):
                price = 0.0

            signals.append(TradeSignal(
                timestamp=now,
                symbol=symbol,
                name=str(row.get("name", symbol)),
                action=action,
                signal_status=SignalStatus.NEW_ENTRY,
                days_in_top_n=0,
                price=price,
                confidence=min(abs(chg) / 10.0, 1.0),
                source="anomaly",
                reason=f"price_change={chg:+.2f}% (>{self._price_thresh}%)",
            ))
        return signals

    # ------------------------------------------------------------------
    # Global anomaly → A-share sector mapping
    # ------------------------------------------------------------------

    def _detect_global_anomaly(
        self, df: pd.DataFrame, now: datetime
    ) -> list[TradeSignal]:
        if not self._sector_mapping:
            return []

        # Deduplicate: keep highest confidence per sector
        best: dict[str, TradeSignal] = {}
        for _, row in df.iterrows():
            symbol = str(row.get("symbol", ""))
            chg = row.get("change_pct")
            if chg is None or (isinstance(chg, float) and math.isnan(chg)):
                continue
            try:
                chg = float(chg)
            except (ValueError, TypeError):
                continue
            if abs(chg) <= self._global_thresh:
                continue

            # Find matching sector mapping
            source_name = _GLOBAL_SYMBOL_MAP.get(symbol)
            if source_name is None:
                continue

            sectors = self._get_mapped_sectors(source_name)
            if not sectors:
                continue

            action = Action.BUY if chg > 0 else Action.SELL
            # Check sentiment_direction
            direction = self._get_sentiment_direction(source_name)
            if direction == "inverse":
                action = Action.SELL if chg > 0 else Action.BUY

            confidence = min(abs(chg) / 5.0, 1.0)
            for sector in sectors:
                prev = best.get(sector)
                if prev is not None and prev.confidence >= confidence:
                    continue
                best[sector] = TradeSignal(
                    timestamp=now,
                    symbol=f"SECTOR:{sector}",
                    name=f"{source_name}→{sector}",
                    action=action,
                    signal_status=SignalStatus.NEW_ENTRY,
                    days_in_top_n=0,
                    price=0.0,
                    confidence=confidence,
                    source="anomaly",
                    reason=f"global:{source_name} {chg:+.2f}% → {sector}",
                )
        return list(best.values())

    def _get_mapped_sectors(self, source_name: str) -> list[str]:
        for entry in self._sector_mapping:
            if entry.get("source") == source_name:
                return entry.get("a_share_sectors", [])
        return []

    def _get_sentiment_direction(self, source_name: str) -> str:
        for entry in self._sector_mapping:
            if entry.get("source") == source_name:
                return entry.get("sentiment_direction", "same")
        return "same"


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------


class FakeAnomalyDetector:
    """Returns empty list — anomaly detection is a stub in M1."""

    def detect(
        self,
        a_share_quotes: pd.DataFrame,
        global_quotes: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        return []
