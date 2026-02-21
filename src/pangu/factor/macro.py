"""Macro factor engine — PRD §4.2.3.

Derives macro/global factors from the international market snapshot
produced by ``MarketDataProvider.get_global_snapshot()``.
"""

from __future__ import annotations

import math

import pandas as pd

# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------

_FACTOR_NAMES: list[str] = [
    "us_overnight",
    "hk_intraday",
    "hk_tech",
    "gold_chg",
    "silver_chg",
    "oil_chg",
    "copper_chg",
    "iron_chg",
    "ng_chg",
    "cotton_chg",
    "vhsi",
    "global_risk",
]

# US index weights for composite overnight factor
_US_WEIGHTS: dict[str, float] = {
    "SPX": 0.4,
    "DJI": 0.3,
    "IXIC": 0.3,
}

# Global risk composite weights (negative = risk-off asset)
_RISK_WEIGHTS: dict[str, float] = {
    "us_overnight": 0.30,
    "hk_intraday": 0.20,
    "gold_chg": -0.15,
    "oil_chg": 0.10,
    "copper_chg": 0.10,
    "vhsi": -0.15,
}


class MacroFactorEngine:
    """Compute macro factors from a global market snapshot DataFrame.

    The snapshot must have columns: ``symbol``, ``change_pct``, ``close``.
    """

    def compute(self, global_snapshot: pd.DataFrame) -> dict[str, float]:
        """Return a dict of macro factor values."""
        if global_snapshot is None or global_snapshot.empty:
            return {k: float("nan") for k in _FACTOR_NAMES}

        result: dict[str, float] = {}

        # Single-symbol factors
        result["hk_intraday"] = self._get_chg(global_snapshot, "HSI")
        result["hk_tech"] = self._get_chg(global_snapshot, "HSTECH")
        result["gold_chg"] = self._get_chg(global_snapshot, "GC")
        result["silver_chg"] = self._get_chg(global_snapshot, "SI")
        result["oil_chg"] = self._get_chg(global_snapshot, "CL")
        result["copper_chg"] = self._get_chg(global_snapshot, "HG")
        result["iron_chg"] = self._get_chg(global_snapshot, "FEF")
        result["ng_chg"] = self._get_chg(global_snapshot, "NG")
        result["cotton_chg"] = self._get_chg(global_snapshot, "CT")

        # VHSI: absolute value (not change_pct)
        result["vhsi"] = self._get_close(global_snapshot, "VHSI")

        # US overnight: weighted composite
        result["us_overnight"] = self._compute_us_overnight(global_snapshot)

        # Global risk: weighted composite of other factors
        result["global_risk"] = self._compute_global_risk(result, global_snapshot)

        return result

    def get_factor_names(self) -> list[str]:
        return list(_FACTOR_NAMES)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_chg(snapshot: pd.DataFrame, symbol: str) -> float:
        """Extract change_pct for *symbol* from snapshot."""
        match = snapshot[snapshot["symbol"] == symbol]
        if match.empty:
            return float("nan")
        val = match.iloc[0].get("change_pct")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        return float(val)

    @staticmethod
    def _get_close(snapshot: pd.DataFrame, symbol: str) -> float:
        """Extract close price for *symbol* (used for VHSI level)."""
        match = snapshot[snapshot["symbol"] == symbol]
        if match.empty:
            return float("nan")
        val = match.iloc[0].get("close")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan")
        return float(val)

    @staticmethod
    def _compute_us_overnight(snapshot: pd.DataFrame) -> float:
        """Weighted average of US index change_pct."""
        total_w = 0.0
        weighted_sum = 0.0
        for sym, w in _US_WEIGHTS.items():
            match = snapshot[snapshot["symbol"] == sym]
            if match.empty:
                continue
            val = match.iloc[0].get("change_pct")
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            weighted_sum += float(val) * w
            total_w += w
        if total_w == 0:
            return float("nan")
        return weighted_sum / total_w

    @staticmethod
    def _compute_global_risk(factors: dict[str, float], snapshot: pd.DataFrame) -> float:
        """Weighted composite risk score. NaN items are skipped.

        Uses VHSI change_pct (not absolute level) for scale consistency.
        """
        # Override vhsi with change_pct for balanced weighting
        vhsi_chg = MacroFactorEngine._get_chg(snapshot, "VHSI")
        adjusted = dict(factors)
        adjusted["vhsi"] = vhsi_chg

        total_w = 0.0
        weighted_sum = 0.0
        for name, w in _RISK_WEIGHTS.items():
            val = adjusted.get(name, float("nan"))
            if isinstance(val, float) and math.isnan(val):
                continue
            weighted_sum += val * w
            total_w += abs(w)
        if total_w == 0:
            return float("nan")
        return weighted_sum / total_w
