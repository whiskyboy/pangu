"""FactorEngine Protocol + PandasTAFactorEngine — PRD §4.2 / §6."""

from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class FactorEngine(Protocol):
    """Interface for computing technical / fundamental factor scores."""

    def compute(
        self, bars: pd.DataFrame, global_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Compute factor values; *global_data* is the international snapshot for macro factors."""
        ...

    def get_factor_names(self) -> list[str]:
        """Return the list of factor names this engine computes."""
        ...


# ---------------------------------------------------------------------------
# Real implementation
# ---------------------------------------------------------------------------

_FACTOR_NAMES: list[str] = [
    # Trend (6)
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "ema12",
    "ema26",
    # Momentum (5)
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "roc_10",
    # Volatility (5)
    "atr_14",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "hv_20",
    # Volume (3)
    "obv",
    "vwap",
    "volume_ratio",
    # Custom (2)
    "bias_20",
    "ma_alignment",
]


class PandasTAFactorEngine:
    """Technical factor engine using pandas-ta (~21 factors).

    Input *bars* must have columns: date, open, high, low, close, volume.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        bars: pd.DataFrame,
        global_data: pd.DataFrame | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Append all technical factor columns to *bars* (copy)."""
        if bars.empty:
            return bars.copy()

        df = bars.copy()

        self._compute_trend(df)
        self._compute_momentum(df)
        self._compute_volatility(df)
        self._compute_volume(df)
        self._compute_custom(df)

        return df

    def get_factor_names(self) -> list[str]:
        return list(_FACTOR_NAMES)

    # ------------------------------------------------------------------
    # Pseudo-bar for intraday
    # ------------------------------------------------------------------

    @staticmethod
    def build_pseudo_bar(
        historical: pd.DataFrame, realtime_quote: dict
    ) -> pd.DataFrame:
        """Append an intraday pseudo bar to *historical* for factor calc.

        *realtime_quote* must contain keys: ``open``, ``high``, ``low``,
        ``price`` (latest), ``volume``.
        """
        pseudo = {
            "date": realtime_quote.get("date", "intraday"),
            "open": float(realtime_quote["open"]),
            "high": float(realtime_quote["high"]),
            "low": float(realtime_quote["low"]),
            "close": float(realtime_quote["price"]),
            "volume": int(realtime_quote["volume"]),
        }
        pseudo_df = pd.DataFrame([pseudo])
        return pd.concat([historical, pseudo_df], ignore_index=True)

    # ------------------------------------------------------------------
    # Trend factors
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trend(df: pd.DataFrame) -> None:
        close = df["close"]
        df["ma5"] = _safe_ta(ta.sma, close, length=5)
        df["ma10"] = _safe_ta(ta.sma, close, length=10)
        df["ma20"] = _safe_ta(ta.sma, close, length=20)
        df["ma60"] = _safe_ta(ta.sma, close, length=60)
        df["ema12"] = _safe_ta(ta.ema, close, length=12)
        df["ema26"] = _safe_ta(ta.ema, close, length=26)

    # ------------------------------------------------------------------
    # Momentum factors
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_momentum(df: pd.DataFrame) -> None:
        close = df["close"]
        nan_s = pd.Series(float("nan"), index=df.index)
        df["rsi_14"] = _safe_ta(ta.rsi, close, length=14)

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            df["macd"] = macd_df.iloc[:, 0]
            df["macd_signal"] = macd_df.iloc[:, 1]
            df["macd_hist"] = macd_df.iloc[:, 2]
        else:
            df["macd"] = nan_s
            df["macd_signal"] = nan_s
            df["macd_hist"] = nan_s

        df["roc_10"] = _safe_ta(ta.roc, close, length=10)

    # ------------------------------------------------------------------
    # Volatility factors
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_volatility(df: pd.DataFrame) -> None:
        high, low, close = df["high"], df["low"], df["close"]
        nan_s = pd.Series(float("nan"), index=df.index)
        df["atr_14"] = _safe_ta(ta.atr, high, low, close, length=14)

        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_mid"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
        else:
            df["bb_upper"] = nan_s
            df["bb_mid"] = nan_s
            df["bb_lower"] = nan_s

        df["hv_20"] = _safe_ta(ta.stdev, close, length=20)

    # ------------------------------------------------------------------
    # Volume factors
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_volume(df: pd.DataFrame) -> None:
        close, volume = df["close"], df["volume"]
        nan_s = pd.Series(float("nan"), index=df.index)
        df["obv"] = _safe_ta(ta.obv, close, volume)

        # VWAP requires a DatetimeIndex
        if {"high", "low"}.issubset(df.columns) and "date" in df.columns:
            try:
                tmp = df.copy()
                tmp.index = pd.DatetimeIndex(tmp["date"])
                vwap = ta.vwap(tmp["high"], tmp["low"], tmp["close"], tmp["volume"])
                df["vwap"] = vwap.values if vwap is not None else nan_s
            except Exception:  # noqa: BLE001
                df["vwap"] = nan_s
        else:
            df["vwap"] = nan_s

        df["volume_ratio"] = _compute_volume_ratio(volume, period=5)

    # ------------------------------------------------------------------
    # Custom factors
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_custom(df: pd.DataFrame) -> None:
        close = df["close"]
        df["bias_20"] = _compute_bias(close, period=20)
        df["ma_alignment"] = _compute_ma_alignment_score(df)


# ---------------------------------------------------------------------------
# Standalone helpers (testable independently)
# ---------------------------------------------------------------------------


def _safe_ta(func, *args, **kwargs) -> pd.Series:  # noqa: ANN002, ANN003
    """Call a pandas-ta function; returns NaN Series on failure."""
    try:
        result = func(*args, **kwargs)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        pass
    # Derive index from first positional arg (a Series)
    idx = args[0].index if len(args) > 0 and hasattr(args[0], "index") else range(0)
    return pd.Series(float("nan"), index=idx)


def _compute_bias(close: pd.Series, period: int = 20) -> pd.Series:
    """Deviation rate: (close - MA) / MA."""
    ma = ta.sma(close, length=period)
    if ma is None:
        return pd.Series(float("nan"), index=close.index)
    return (close - ma) / ma.replace(0, float("nan"))


def _compute_ma_alignment_score(df: pd.DataFrame) -> pd.Series:
    """Bull alignment score (0-4).

    +1 for each: MA5>MA10, MA10>MA20, MA20>MA60, MA5>MA60.
    Requires ma5/ma10/ma20/ma60 columns already computed.
    """
    score = pd.Series(0.0, index=df.index, dtype=float)
    for col in ("ma5", "ma10", "ma20", "ma60"):
        if col not in df.columns:
            return score

    # Convert None to NaN for safe comparison
    for col in ("ma5", "ma10", "ma20", "ma60"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    score += (df["ma5"] > df["ma10"]).astype(float)
    score += (df["ma10"] > df["ma20"]).astype(float)
    score += (df["ma20"] > df["ma60"]).astype(float)
    score += (df["ma5"] > df["ma60"]).astype(float)
    return score


def _compute_volume_ratio(volume: pd.Series, period: int = 5) -> pd.Series:
    """Volume ratio = current volume / average of past *period* days."""
    ma = volume.rolling(window=period).mean().shift(1)
    return volume / ma.replace(0, float("nan"))
