"""Strategy Protocol + MultiFactorStrategy — PRD §4.3.1 / §6."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Protocol

import pandas as pd

from trading_agent.models import Action, SignalStatus, TradeSignal

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    """Interface for factor-based signal generation."""

    def generate_signals(
        self,
        tech_df: dict[str, pd.DataFrame],
        fund_df: pd.DataFrame,
        macro_factors: dict[str, float],
        *,
        prev_pool: pd.DataFrame | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, list[TradeSignal]]:
        """Generate trade signals from pre-computed factor data.

        Returns:
            (factor_pool DataFrame, list of TradeSignal)
        """
        ...


# ---------------------------------------------------------------------------
# Default factor weights (from PRD §4.3.1 / settings.toml)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "rsi_14": 0.15,
    "macd_hist": 0.20,
    "bias_20": 0.10,
    "obv": 0.10,
    "atr_14": 0.10,
    "volume_ratio": 0.05,
    "pe_ttm": 0.10,
    "pb": 0.05,
    "roe_ttm": 0.10,
    "macro_adj": 0.05,
}

# Macro source → factor column mapping for sector adjustment
_MACRO_SOURCE_MAP: dict[str, str] = {
    "COMEX黄金": "gold_chg",
    "COMEX白银": "silver_chg",
    "WTI原油": "oil_chg",
    "LME铜": "copper_chg",
    "铁矿石": "iron_chg",
    "NYMEX天然气": "ng_chg",
    "NYBOT棉花": "cotton_chg",
}


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: (x - mean) / std per column."""
    result = df.copy()
    for col in result.columns:
        s = result[col]
        valid = s.dropna()
        if len(valid) < 2:
            continue
        mean, std = valid.mean(), valid.std()
        if std > 0:
            result[col] = (s - mean) / std
        else:
            result[col] = 0.0
    return result


def _weighted_score(normalized: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted sum of factor columns. Missing columns are skipped."""
    score = pd.Series(0.0, index=normalized.index)
    total_w = 0.0
    for col, w in weights.items():
        if col in normalized.columns:
            score += normalized[col].fillna(0.0) * w
            total_w += w
    if total_w > 0:
        score /= total_w
    return score


def _minmax_normalize(scores: pd.Series) -> pd.Series:
    """Normalize to 0-1 range."""
    mn, mx = scores.min(), scores.max()
    if mx - mn > 0:
        return (scores - mn) / (mx - mn)
    return pd.Series(0.5, index=scores.index)


class MultiFactorStrategy:
    """Multi-factor scoring strategy with sector-adjusted macro factors.

    Pure computation — no DB, no API, no side effects.
    Receives pre-computed factor data, returns scores and signals.
    """

    def __init__(
        self,
        *,
        top_n: int = 3,
        buy_threshold: float = 0.7,
        sell_threshold: float = 0.3,
        risk_dampen_threshold: float = -1.0,
        weights: dict[str, float] | None = None,
        sector_mapping: list[dict] | None = None,
    ) -> None:
        self._top_n = top_n
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._risk_dampen = risk_dampen_threshold
        self._weights = weights or dict(_DEFAULT_WEIGHTS)
        self._sector_mapping = sector_mapping or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        tech_df: dict[str, pd.DataFrame],
        fund_df: pd.DataFrame,
        macro_factors: dict[str, float],
        *,
        prev_pool: pd.DataFrame | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, list[TradeSignal]]:
        """Generate trade signals from pre-computed factor data.

        Args:
            tech_df: symbol → DataFrame with technical factor columns appended.
            fund_df: DataFrame (index=symbol, columns=fundamental factors).
            macro_factors: dict of global macro factor values.
            prev_pool: previous factor_pool DataFrame for status tracking.
            sector_map: symbol → sector string for macro adjustment.

        Returns:
            (factor_pool_df, signals)
        """
        if not tech_df:
            return pd.DataFrame(columns=["symbol", "score", "rank"]), []

        from trading_agent.tz import now as _now

        pool = list(tech_df.keys())
        sector_map = sector_map or {}

        # 1. Build factor matrix
        matrix = self._build_factor_matrix(
            pool, tech_df, fund_df, macro_factors, sector_map
        )
        if matrix.empty:
            return pd.DataFrame(columns=["symbol", "score", "rank"]), []

        # 2. Z-score normalize
        normalized = _zscore_normalize(matrix)

        # 3. Weighted score
        scores = _weighted_score(normalized, self._weights)

        # 4. Normalize to 0-1
        scores = _minmax_normalize(scores)

        # 5. Rank
        ranks = scores.rank(ascending=False, method="min").astype(int)

        # 6. Build factor pool DataFrame
        pool_df = pd.DataFrame({
            "symbol": scores.index,
            "score": scores.values,
            "rank": ranks.values,
        })

        # 7. Generate signals
        global_risk = macro_factors.get("global_risk", 0.0)
        if isinstance(global_risk, float) and math.isnan(global_risk):
            global_risk = 0.0

        signals = self._generate_signals(
            scores, ranks, pool, tech_df, prev_pool, global_risk, _now()
        )

        return pool_df, signals

    # ------------------------------------------------------------------
    # Internal: factor matrix construction
    # ------------------------------------------------------------------

    def _build_factor_matrix(
        self,
        pool: list[str],
        tech_df: dict[str, pd.DataFrame],
        fund_df: pd.DataFrame,
        macro_factors: dict[str, float],
        sector_map: dict[str, str],
    ) -> pd.DataFrame:
        """Build cross-sectional factor matrix (rows=symbols, cols=factors)."""
        rows: list[dict[str, float]] = []
        valid_symbols: list[str] = []

        for sym in pool:
            bars = tech_df.get(sym)
            if bars is None or bars.empty:
                continue

            # Technical factors: take last row
            tech_last = {}
            for col in bars.columns:
                if col in ("date", "open", "high", "low", "close", "volume"):
                    continue
                val = bars[col].iloc[-1]
                tech_last[col] = float(val) if val is not None else float("nan")

            # Fundamental factors
            fund_row = {}
            if fund_df is not None and not fund_df.empty and sym in fund_df.index:
                for col in fund_df.columns:
                    val = fund_df.loc[sym, col]
                    fund_row[col] = float(val) if val is not None else float("nan")

            # Sector-adjusted macro score
            sector = sector_map.get(sym, "")
            macro_adj = self._compute_sector_macro(macro_factors, sector)

            combined = {**tech_last, **fund_row, "macro_adj": macro_adj}
            rows.append(combined)
            valid_symbols.append(sym)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows, index=valid_symbols)

    def _compute_sector_macro(
        self, macro_factors: dict[str, float], sector: str
    ) -> float:
        """Compute sector-adjusted macro score using global_market_mapping."""
        if not sector or not self._sector_mapping:
            return 0.0

        total = 0.0
        count = 0
        for entry in self._sector_mapping:
            sectors = entry.get("a_share_sectors", [])
            if sector not in sectors and "全行业" not in sectors:
                continue
            source_name = entry.get("source", "")
            factor_key = _MACRO_SOURCE_MAP.get(source_name)
            if factor_key is None:
                continue
            val = macro_factors.get(factor_key, float("nan"))
            if isinstance(val, float) and math.isnan(val):
                continue
            w = entry.get("weight", 0.0)
            total += val * w
            count += 1

        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal: signal generation
    # ------------------------------------------------------------------

    def _generate_signals(
        self,
        scores: pd.Series,
        ranks: pd.Series,
        pool: list[str],
        tech_df: dict[str, pd.DataFrame],
        prev_pool: pd.DataFrame | None,
        global_risk: float,
        now: datetime,
    ) -> list[TradeSignal]:
        """Convert scores and ranks into TradeSignal list."""
        buy_threshold = self._buy_threshold
        if global_risk < self._risk_dampen:
            buy_threshold += 0.1

        prev_top: set[str] = set()
        if prev_pool is not None and not prev_pool.empty:
            prev_top = set(
                prev_pool[prev_pool["rank"] <= self._top_n]["symbol"].tolist()
            )

        signals: list[TradeSignal] = []
        for sym in pool:
            if sym not in scores.index:
                continue

            score = float(scores[sym])
            rank = int(ranks[sym])

            # Get price and ATR from tech bars
            bars = tech_df.get(sym)
            price = float(bars["close"].iloc[-1]) if bars is not None and not bars.empty else 0.0
            atr = self._get_atr(bars)

            in_top = rank <= self._top_n
            was_in_top = sym in prev_top

            # Determine action
            if in_top and score >= buy_threshold:
                action = Action.BUY
                status = SignalStatus.SUSTAINED if was_in_top else SignalStatus.NEW_ENTRY
                signals.append(self._make_signal(
                    now, sym, action, status, price, score, atr,
                    f"rank={rank} score={score:.3f}",
                ))
            elif was_in_top and not in_top:
                signals.append(self._make_signal(
                    now, sym, Action.SELL, SignalStatus.EXIT, price, score, atr,
                    f"exit top-{self._top_n}: rank={rank} score={score:.3f}",
                ))

        return signals

    def _make_signal(
        self,
        now: datetime,
        symbol: str,
        action: Action,
        status: SignalStatus,
        price: float,
        score: float,
        atr: float,
        reason: str,
    ) -> TradeSignal:
        stop_loss = price - 2 * atr if atr > 0 and action == Action.BUY else None
        take_profit = price + 3 * atr if atr > 0 and action == Action.BUY else None
        return TradeSignal(
            timestamp=now,
            symbol=symbol,
            name=symbol,
            action=action,
            signal_status=status,
            days_in_top_n=0,
            price=price,
            confidence=score,
            source="factor",
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            factor_score=score,
        )

    @staticmethod
    def _get_atr(bars: pd.DataFrame | None) -> float:
        """Get ATR from bars (last row of atr_14 if computed, else 0)."""
        if bars is None or bars.empty:
            return 0.0
        if "atr_14" in bars.columns:
            val = bars["atr_14"].iloc[-1]
            return float(val) if pd.notna(val) else 0.0
        return 0.0


# ---------------------------------------------------------------------------
# Fake implementation for testing / development
# ---------------------------------------------------------------------------

_FAKE_SCORES: dict[str, float] = {
    "600967": 0.78,
    "601899": 0.85,
    "603993": 0.72,
    "000750": 0.30,
    "002466": 0.55,
}


class FakeFactorStrategy:
    """Generates deterministic signals based on hardcoded factor scores."""

    def generate_signals(
        self,
        tech_df: dict[str, pd.DataFrame],
        fund_df: pd.DataFrame,
        macro_factors: dict[str, float],
        *,
        prev_pool: pd.DataFrame | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, list[TradeSignal]]:
        from trading_agent.tz import now as _now
        now = _now()
        signals: list[TradeSignal] = []
        pool = list(tech_df.keys()) if tech_df else list(_FAKE_SCORES.keys())
        for symbol in pool:
            score = _FAKE_SCORES.get(symbol, 0.5)
            if score >= 0.7:
                action = Action.BUY
                confidence = score
            elif score <= 0.3:
                action = Action.SELL
                confidence = 1.0 - score
            else:
                continue  # no signal for HOLD-range scores
            signals.append(TradeSignal(
                timestamp=now,
                symbol=symbol,
                name=symbol,
                action=action,
                signal_status=SignalStatus.NEW_ENTRY,
                days_in_top_n=0,
                price=100.0,
                confidence=confidence,
                source="factor",
                reason=f"fake factor score={score:.2f}",
                factor_score=score,
            ))
        pool_df = pd.DataFrame({
            "symbol": pool,
            "score": [_FAKE_SCORES.get(s, 0.5) for s in pool],
            "rank": list(range(1, len(pool) + 1)),
        })
        return pool_df, signals
