"""Baseline (Strategy 0) scoring for backtest validation.

Replicates PanGu multi_factor.py scoring pipeline:
8 factors + manual weights → z-score → weighted sum → winsorize → minmax.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pangu.strategy.factor.multi_factor import (
    _minmax_normalize,
    _weighted_score,
    _zscore_normalize,
)

BASELINE_WEIGHTS = {
    "rsi_14": 0.15,
    "macd_hist": 0.20,
    "bias_20": 0.10,
    "obv": 0.10,
    "volume_ratio": 0.05,
    "pe_ttm": -0.10,
    "pb": -0.05,
    "roe_ttm": 0.10,
}


def compute_technical_factors(
    all_bars: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute technical factors on (date × stock) wide tables.

    Prices in all_bars are unadjusted (real market prices). This function
    converts to forward-adjusted prices using adj_factor before computing
    technical indicators, ensuring price continuity across ex-dividend dates.

    Returns dict mapping factor name → DataFrame(date × stock).
    """
    raw_close = all_bars.pivot(index="date", columns="symbol", values="close")
    adj = all_bars.pivot(index="date", columns="symbol", values="adj_factor").ffill()
    volume = all_bars.pivot(index="date", columns="symbol", values="volume")

    # Forward-adjusted close for continuous technical indicators
    close = raw_close * adj

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1.0 / 14, min_periods=14).mean()
    al = loss.ewm(alpha=1.0 / 14, min_periods=14).mean()
    rsi_14 = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    # MACD signal line (replicating PanGu bug: macd_hist = signal line)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line.ewm(span=9, adjust=False).mean()

    # BIAS(20)
    sma20 = close.rolling(20).mean()
    bias_20 = (close - sma20) / sma20.replace(0, np.nan)

    # OBV (60-day rolling, matching JoinQuant's 60-day window)
    direction = np.sign(close.diff())
    obv = (volume * direction).rolling(60).sum()

    # Volume Ratio
    vol_ma5 = volume.rolling(5).mean().shift(1)
    volume_ratio = volume / vol_ma5.replace(0, np.nan)

    return {
        "rsi_14": rsi_14,
        "macd_hist": macd_hist,
        "bias_20": bias_20,
        "obv": obv,
        "volume_ratio": volume_ratio,
    }


def load_fundamental_factors(
    storage: object, pool: list[str], start: str, end: str, dates_index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Load PE/PB/ROE via load_fundamentals_filled, aligned to dates_index."""
    fund_frames = []
    for sym in pool:
        df = storage.load_fundamentals_filled(sym, start, end)
        if df is not None and not df.empty:
            fund_frames.append(df[["symbol", "date", "pe_ttm", "pb", "roe_ttm"]])

    if fund_frames:
        fund_df = pd.concat(fund_frames, ignore_index=True)
        fund_df["date"] = pd.to_datetime(fund_df["date"])
    else:
        fund_df = pd.DataFrame(columns=["symbol", "date", "pe_ttm", "pb", "roe_ttm"])

    result = {}
    for col in ("pe_ttm", "pb", "roe_ttm"):
        wide = fund_df.pivot(index="date", columns="symbol", values=col)
        result[col] = wide.reindex(dates_index, method="ffill")
    return result


def compute_baseline_scores(
    all_bars: pd.DataFrame,
    storage: object,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Compute Strategy 0 daily scores for all stocks.

    Parameters
    ----------
    all_bars : DataFrame
        Long-format bars with columns: date, symbol, open, close, high, low, volume.
        Should include warmup data before *start* for factor stability.
    storage : DataStorage
        Database handle for fundamentals and constituents.
    start, end : str
        Backtest date range. Scores before *start* (warmup) are dropped.

    Returns
    -------
    DataFrame (date × stock) with scores in [0, 1]. Higher = better.
    """
    # Ensure datetime index
    bars = all_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])

    pool = bars["symbol"].unique().tolist()

    # Phase 1: Technical factors
    tech = compute_technical_factors(bars)
    dates_index = tech["rsi_14"].index

    # Phase 2: Fundamental factors
    fund = load_fundamental_factors(storage, pool, start, end, dates_index)

    factors = {**tech, **fund}

    # Phase 3: Day-by-day scoring
    all_scores = {}
    for date in dates_index:
        row = {name: wide.loc[date] for name, wide in factors.items() if date in wide.index}
        if not row:
            continue

        matrix = pd.DataFrame(row)
        if matrix.empty or len(matrix.dropna(how="all")) < 5:
            continue

        # Filter to point-in-time constituents before z-scoring
        universe = set(storage.load_constituents_for_date(date.strftime("%Y-%m-%d")))
        matrix = matrix[matrix.index.isin(universe)]
        if len(matrix.dropna(how="all")) < 5:
            continue

        normalized = _zscore_normalize(matrix)
        score = _weighted_score(normalized, BASELINE_WEIGHTS)
        score = _minmax_normalize(score)
        all_scores[date] = score

    result = pd.DataFrame(all_scores).T
    return result[result.index >= start]
