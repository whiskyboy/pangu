"""QLib Alpha158 full factor set + fundamental factors.

Computes 159 technical factors + 18 fundamental factors = 177 total,
all vectorized on (date × symbol) wide-format DataFrames.

Technical factors follow the exact definitions from Microsoft QLib's Alpha158:
  - KBar (9): candlestick shape features
  - Price (5): OHLC/VWAP ratios to close + overnight return
  - Rolling (145): 29 operations × 5 windows {5, 10, 20, 30, 60}

Fundamental factors (18): PE, PB, PS, PCF, ROE, revenue_yoy, profit_yoy,
  ln_mktcap, turnover, gross_margin, net_profit_margin, debt_ratio,
  asset_turnover, current_ratio, equity_yoy, asset_yoy,
  cashflow_per_share, cashflow_to_profit.

Prices are forward-adjusted (unadj × adj_factor) for continuity;
volume and amount are NOT adjusted.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pangu.data.storage import DataStorage

_EPS = 1e-12
_WINDOWS = [5, 10, 20, 30, 60]
_WARMUP = 60  # max rolling window; first 59 rows produce NaN

_FUNDAMENTAL_COLS = [
    "PE",
    "PB",
    "PS",
    "PCF",
    "ROE",
    "REVENUE_YOY",
    "PROFIT_YOY",
    "LN_MKTCAP",
    "TURNOVER",
    "GROSS_MARGIN",
    "NET_PROFIT_MARGIN",
    "DEBT_RATIO",
    "ASSET_TURNOVER",
    "CURRENT_RATIO",
    "EQUITY_YOY",
    "ASSET_YOY",
    "CASHFLOW_PER_SHARE",
    "CASHFLOW_TO_PROFIT",
]


# ---------------------------------------------------------------------------
# Factor name registry (deterministic order, 177 total)
# ---------------------------------------------------------------------------

def _build_factor_names() -> list[str]:
    """Return the ordered list of all 177 factor names."""
    names: list[str] = []

    # KBar (9)
    names += [
        "KMID", "KLEN", "KMID2", "KUP", "KUP2",
        "KLOW", "KLOW2", "KSFT", "KSFT2",
    ]

    # Price (5)
    names += ["OPEN0", "HIGH0", "LOW0", "VWAP0", "OVERNIGHT_RET"]

    # Rolling (29 ops × 5 windows = 145)
    _rolling_ops = [
        "ROC", "MA", "STD", "BETA", "RSQR", "RESI",
        "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",
        "IMAX", "IMIN", "IMXD",
        "CORR", "CORD",
        "CNTP", "CNTN", "CNTD",
        "SUMP", "SUMN", "SUMD",
        "VMA", "VSTD", "WVMA",
        "VSUMP", "VSUMN", "VSUMD",
    ]
    for d in _WINDOWS:
        for op in _rolling_ops:
            names.append(f"{op}{d}")

    # Fundamental (18)
    names += _FUNDAMENTAL_COLS

    assert len(names) == 177, f"Expected 177, got {len(names)}"
    return names


FACTOR_NAMES: list[str] = _build_factor_names()


# ---------------------------------------------------------------------------
# Wide-format pivot helpers
# ---------------------------------------------------------------------------

def _pivot(bars: pd.DataFrame, col: str) -> pd.DataFrame:
    return bars.pivot(index="date", columns="symbol", values=col)


def _prepare_wide_tables(
    all_bars: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Pivot long-format bars to wide (date × symbol) and forward-adjust prices."""
    bars = all_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])

    # Deduplicate in case of data errors (keep latest row per date+symbol)
    bars = bars.drop_duplicates(subset=["date", "symbol"], keep="last")

    raw_open = _pivot(bars, "open")
    raw_high = _pivot(bars, "high")
    raw_low = _pivot(bars, "low")
    raw_close = _pivot(bars, "close")
    volume = _pivot(bars, "volume").astype("float64")
    amount = _pivot(bars, "amount").astype("float64")
    adj = _pivot(bars, "adj_factor").ffill()

    O = raw_open * adj  # noqa: E741
    H = raw_high * adj
    L = raw_low * adj
    C = raw_close * adj
    VWAP = (amount / (volume + _EPS)) * adj

    return {
        "O": O, "H": H, "L": L, "C": C,
        "V": volume, "amount": amount, "VWAP": VWAP,
        "raw_open": raw_open, "preclose": _pivot(bars, "preclose") if "preclose" in bars.columns else None,
    }


# ---------------------------------------------------------------------------
# KBar factors (9)
# ---------------------------------------------------------------------------

def _compute_kbar(
    O: pd.DataFrame, H: pd.DataFrame, L: pd.DataFrame, C: pd.DataFrame,  # noqa: E741
) -> dict[str, pd.DataFrame]:
    oc_max = pd.DataFrame(
        np.maximum(O.values, C.values), index=O.index, columns=O.columns,
    )
    oc_min = pd.DataFrame(
        np.minimum(O.values, C.values), index=O.index, columns=O.columns,
    )
    hl = H - L + _EPS
    o_safe = O + _EPS  # protect against O=0 (data errors)

    return {
        "KMID": (C - O) / o_safe,
        "KLEN": (H - L) / o_safe,
        "KMID2": (C - O) / hl,
        "KUP": (H - oc_max) / o_safe,
        "KUP2": (H - oc_max) / hl,
        "KLOW": (oc_min - L) / o_safe,
        "KLOW2": (oc_min - L) / hl,
        "KSFT": (2 * C - H - L) / o_safe,
        "KSFT2": (2 * C - H - L) / hl,
    }


# ---------------------------------------------------------------------------
# Price factors (4)
# ---------------------------------------------------------------------------

def _compute_price(
    O: pd.DataFrame, H: pd.DataFrame, L: pd.DataFrame,  # noqa: E741
    C: pd.DataFrame, VWAP: pd.DataFrame,
    raw_open: pd.DataFrame | None = None,
    preclose: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    c_safe = C + _EPS  # protect against C=0 (data errors)
    factors: dict[str, pd.DataFrame] = {
        "OPEN0": O / c_safe,
        "HIGH0": H / c_safe,
        "LOW0": L / c_safe,
        "VWAP0": VWAP / c_safe,
    }
    # OVERNIGHT_RET uses UNADJUSTED prices (raw_open / preclose).
    # This is the only factor NOT using forward-adjusted prices, because:
    #   - preclose is the exchange reference price (adjusted for ex-dividend by exchange)
    #   - Using forward-adjusted prices would mute the overnight gap on ex-dividend days
    if raw_open is not None and preclose is not None:
        pc_safe = preclose.astype("float64") + _EPS
        factors["OVERNIGHT_RET"] = raw_open.astype("float64") / pc_safe - 1
    else:
        factors["OVERNIGHT_RET"] = pd.DataFrame(np.nan, index=O.index, columns=O.columns)
    return factors


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------

def _rolling_rank(s: pd.DataFrame, d: int) -> pd.DataFrame:
    """Percentile rank of current value in rolling window."""
    return s.rolling(d, min_periods=d).apply(
        lambda x: (x[-1] >= x).mean(), raw=True,
    )


def _rolling_idxmax(s: pd.DataFrame, d: int) -> pd.DataFrame:
    """Index of max in window (0=oldest), normalized by d."""
    return s.rolling(d, min_periods=d).apply(
        lambda x: np.argmax(x) / d, raw=True,
    )


def _rolling_idxmin(s: pd.DataFrame, d: int) -> pd.DataFrame:
    """Index of min in window (0=oldest), normalized by d."""
    return s.rolling(d, min_periods=d).apply(
        lambda x: np.argmin(x) / d, raw=True,
    )


def _rolling_slope_rsqr_resi(
    s: pd.DataFrame, d: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Efficient rolling linear regression using analytical formulas.

    Regresses s on [0, 1, ..., d-1] time index.
    Returns (slope, rsquare, residual) DataFrames.
    """
    # Pre-computed constants for x = [0, 1, ..., d-1]
    sum_x = d * (d - 1) / 2.0
    sum_x2 = d * (d - 1) * (2 * d - 1) / 6.0
    denom = d * sum_x2 - sum_x ** 2

    # Rolling sum of x*y:
    # sum_xy = sum_{i=0}^{d-1} i * y_{t-d+1+i}
    # We use the identity: sum_xy = (d-1)*sum_y - sum_{k=1}^{d-1} cumsum_shift_k
    # More efficient approach: construct weighted series and use rolling sum.
    # weight_i = d - 1 - i for y_{t-i} (i=0 is current, i=d-1 is oldest)
    # sum_xy = (d-1)*y_t + (d-2)*y_{t-1} + ... + 0*y_{t-d+1}
    #        = (d-1) * sum_y - rolling_sum(cumsum_of_y, d) + sum_y
    # Simpler: use rolling.apply for moderate d values

    def _regress(arr: np.ndarray) -> np.ndarray:
        """Vectorized regression for a 1D array (single stock column)."""
        n = len(arr)
        x = np.arange(d, dtype=np.float64)
        result_slope = np.full(n, np.nan)
        result_rsqr = np.full(n, np.nan)
        result_resi = np.full(n, np.nan)

        for t in range(d - 1, n):
            y = arr[t - d + 1 : t + 1]
            if np.isnan(y).any():
                continue
            sxy = np.dot(x, y)
            sy = y.sum()
            slope = (d * sxy - sum_x * sy) / denom
            intercept = (sy - slope * sum_x) / d
            fitted_last = intercept + slope * (d - 1)
            resi = y[-1] - fitted_last

            # R²
            y_hat = intercept + slope * x
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            rsqr = 1.0 - ss_res / (ss_tot + _EPS)

            result_slope[t] = slope
            result_rsqr[t] = rsqr
            result_resi[t] = resi

        return result_slope, result_rsqr, result_resi

    slope_data = np.full_like(s.values, np.nan)
    rsqr_data = np.full_like(s.values, np.nan)
    resi_data = np.full_like(s.values, np.nan)

    for col_idx in range(s.shape[1]):
        col_vals = s.values[:, col_idx]
        sl, rq, re = _regress(col_vals)
        slope_data[:, col_idx] = sl
        rsqr_data[:, col_idx] = rq
        resi_data[:, col_idx] = re

    slope = pd.DataFrame(slope_data, index=s.index, columns=s.columns)
    rsqr = pd.DataFrame(rsqr_data, index=s.index, columns=s.columns)
    resi = pd.DataFrame(resi_data, index=s.index, columns=s.columns)

    return slope, rsqr, resi


# ---------------------------------------------------------------------------
# Rolling factors: simple pandas operations (55 factors)
# ---------------------------------------------------------------------------

def _compute_rolling_simple(
    C: pd.DataFrame, H: pd.DataFrame, L: pd.DataFrame, V: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """ROC, MA, STD, MAX, MIN, QTLU, QTLD, RSV, VMA, VSTD, WVMA."""
    factors: dict[str, pd.DataFrame] = {}
    c_safe = C + _EPS  # protect against C=0
    for d in _WINDOWS:
        # Price-based (all normalized by close)
        factors[f"ROC{d}"] = C.shift(d) / c_safe
        factors[f"MA{d}"] = C.rolling(d, min_periods=d).mean() / c_safe
        factors[f"STD{d}"] = C.rolling(d, min_periods=d).std() / c_safe
        factors[f"MAX{d}"] = H.rolling(d, min_periods=d).max() / c_safe
        factors[f"MIN{d}"] = L.rolling(d, min_periods=d).min() / c_safe
        factors[f"QTLU{d}"] = C.rolling(d, min_periods=d).quantile(0.8) / c_safe
        factors[f"QTLD{d}"] = C.rolling(d, min_periods=d).quantile(0.2) / c_safe

        max_h = H.rolling(d, min_periods=d).max()
        min_l = L.rolling(d, min_periods=d).min()
        factors[f"RSV{d}"] = (C - min_l) / (max_h - min_l + _EPS)

        # Volume-based
        v_mean = V.rolling(d, min_periods=d).mean()
        v_std = V.rolling(d, min_periods=d).std()
        factors[f"VMA{d}"] = v_mean / (V + _EPS)
        factors[f"VSTD{d}"] = v_std / (V + _EPS)
        factors[f"WVMA{d}"] = v_std / (v_mean + _EPS)

    return factors


# ---------------------------------------------------------------------------
# Rolling factors: complex operations (75 factors)
# ---------------------------------------------------------------------------

def _compute_rolling_complex(
    C: pd.DataFrame, V: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """RANK, IMAX, IMIN, IMXD, CORR, CORD, CNT*, SUM*, VSUMP/VSUMN/VSUMD."""
    factors: dict[str, pd.DataFrame] = {}

    # Pre-compute daily changes (once, outside loop)
    delta_c = C.diff()
    ret_c = C / C.shift(1)
    log_v = np.log(V + 1)
    delta_v = V.diff()
    log_v_ratio = np.log(V / V.shift(1).replace(0, np.nan) + 1)
    up = (C > C.shift(1)).astype(float)
    down = (C < C.shift(1)).astype(float)
    pos_change = delta_c.clip(lower=0)
    neg_change = (-delta_c).clip(lower=0)
    abs_change = delta_c.abs()
    v_pos = delta_v.clip(lower=0)
    v_neg = (-delta_v).clip(lower=0)
    v_abs = delta_v.abs()

    for d in _WINDOWS:
        # RANK: percentile rank of current close in window
        factors[f"RANK{d}"] = _rolling_rank(C, d)

        # IMAX / IMIN: position of max/min in window, normalized
        imax = _rolling_idxmax(C, d)
        imin = _rolling_idxmin(C, d)
        factors[f"IMAX{d}"] = imax
        factors[f"IMIN{d}"] = imin
        factors[f"IMXD{d}"] = imax - imin

        # CORR: correlation of close with log(volume+1)
        # replace inf with NaN (occurs when price is constant in window → zero variance)
        corr = C.rolling(d, min_periods=d).corr(log_v)
        factors[f"CORR{d}"] = corr.replace([np.inf, -np.inf], np.nan)

        # CORD: correlation of returns with log(volume change ratio + 1)
        cord = ret_c.rolling(d, min_periods=d).corr(log_v_ratio)
        factors[f"CORD{d}"] = cord.replace([np.inf, -np.inf], np.nan)

        # CNTP / CNTN / CNTD: up/down day proportions
        cntp = up.rolling(d, min_periods=d).mean()
        cntn = down.rolling(d, min_periods=d).mean()
        factors[f"CNTP{d}"] = cntp
        factors[f"CNTN{d}"] = cntn
        factors[f"CNTD{d}"] = cntp - cntn

        # SUMP / SUMN / SUMD: positive/negative change ratios
        sum_abs = abs_change.rolling(d, min_periods=d).sum() + _EPS
        sump = pos_change.rolling(d, min_periods=d).sum() / sum_abs
        sumn = neg_change.rolling(d, min_periods=d).sum() / sum_abs
        factors[f"SUMP{d}"] = sump
        factors[f"SUMN{d}"] = sumn
        factors[f"SUMD{d}"] = sump - sumn

        # VSUMP / VSUMN / VSUMD: volume positive/negative change ratios
        v_sum_abs = v_abs.rolling(d, min_periods=d).sum() + _EPS
        vsump = v_pos.rolling(d, min_periods=d).sum() / v_sum_abs
        vsumn = v_neg.rolling(d, min_periods=d).sum() / v_sum_abs
        factors[f"VSUMP{d}"] = vsump
        factors[f"VSUMN{d}"] = vsumn
        factors[f"VSUMD{d}"] = vsump - vsumn

    return factors


# ---------------------------------------------------------------------------
# Rolling factors: linear regression (15 factors)
# ---------------------------------------------------------------------------

def _compute_rolling_regression(
    C: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """BETA, RSQR, RESI for each window via analytical linear regression."""
    factors: dict[str, pd.DataFrame] = {}
    c_safe = C + _EPS
    for d in _WINDOWS:
        slope, rsqr, resi = _rolling_slope_rsqr_resi(C, d)
        factors[f"BETA{d}"] = slope / c_safe
        factors[f"RSQR{d}"] = rsqr
        factors[f"RESI{d}"] = resi / c_safe
    return factors


# ---------------------------------------------------------------------------
# Fundamental factors (8)
# ---------------------------------------------------------------------------

def _compute_fundamentals(
    fundamentals: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute 18 fundamental factors from fundamentals + daily amount.

    Parameters
    ----------
    fundamentals : DataFrame
        Long format with columns: symbol, date, pe_ttm, pb, ps_ttm, pcf_ttm,
        roe_ttm, revenue_yoy, profit_yoy, market_cap, gross_margin,
        net_profit_margin, debt_ratio, asset_turnover, current_ratio,
        equity_yoy, asset_yoy, cashflow_per_share, cashflow_to_profit.
    amount : DataFrame
        Wide format (date × symbol) daily trading amount in yuan.
    """
    fund = fundamentals.copy()
    fund["date"] = pd.to_datetime(fund["date"])
    # Deduplicate (keep latest row per date+symbol)
    fund = fund.drop_duplicates(subset=["date", "symbol"], keep="last")

    factors: dict[str, pd.DataFrame] = {}

    for src_col, out_name in [
        ("pe_ttm", "PE"),
        ("pb", "PB"),
        ("ps_ttm", "PS"),
        ("pcf_ttm", "PCF"),
        ("roe_ttm", "ROE"),
        ("revenue_yoy", "REVENUE_YOY"),
        ("profit_yoy", "PROFIT_YOY"),
        ("gross_margin", "GROSS_MARGIN"),
        ("net_profit_margin", "NET_PROFIT_MARGIN"),
        ("debt_ratio", "DEBT_RATIO"),
        ("asset_turnover", "ASSET_TURNOVER"),
        ("current_ratio", "CURRENT_RATIO"),
        ("equity_yoy", "EQUITY_YOY"),
        ("asset_yoy", "ASSET_YOY"),
        ("cashflow_per_share", "CASHFLOW_PER_SHARE"),
        ("cashflow_to_profit", "CASHFLOW_TO_PROFIT"),
    ]:
        if src_col in fund.columns:
            wide = fund.pivot(index="date", columns="symbol", values=src_col)
            factors[out_name] = wide.reindex(amount.index, method="ffill")

    # LN_MKTCAP = log(market_cap)
    if "market_cap" in fund.columns:
        mktcap_wide = fund.pivot(
            index="date", columns="symbol", values="market_cap",
        )
        mktcap_aligned = mktcap_wide.reindex(amount.index, method="ffill").astype("float64")
        factors["LN_MKTCAP"] = np.log(mktcap_aligned.clip(lower=_EPS))

    # TURNOVER = daily amount / market_cap
    if "market_cap" in fund.columns:
        factors["TURNOVER"] = amount / (mktcap_aligned + _EPS)

    return factors


# ---------------------------------------------------------------------------
# Alpha158Engine
# ---------------------------------------------------------------------------

class Alpha158Engine:
    """QLib Alpha158 full factor set + fundamental factors, vectorized."""

    WINDOWS = _WINDOWS
    WARMUP = _WARMUP

    @staticmethod
    def get_factor_names() -> list[str]:
        """Return ordered list of all 177 factor names (no duplicates)."""
        return list(FACTOR_NAMES)

    def compute(
        self,
        all_bars: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Batch compute all 177 factors.

        Parameters
        ----------
        all_bars : DataFrame (long format)
            Columns: symbol, date, open, high, low, close, volume, amount,
            adj_factor, preclose.
            Unadjusted prices; forward adjustment done internally.
        fundamentals : DataFrame (long format)
            Columns: symbol, date, pe_ttm, pb, ps_ttm, pcf_ttm, roe_ttm,
            revenue_yoy, profit_yoy, market_cap, gross_margin,
            net_profit_margin, debt_ratio, asset_turnover, current_ratio,
            equity_yoy, asset_yoy, cashflow_per_share, cashflow_to_profit.

        Returns
        -------
        DataFrame with MultiIndex (date, symbol) and 177 factor columns, dtype=float32.
        First ~59 rows per stock will have NaN for rolling(60) factors.
        """
        # Pivot to wide format + forward-adjust prices
        wide = _prepare_wide_tables(all_bars)
        O, H, L, C = wide["O"], wide["H"], wide["L"], wide["C"]  # noqa: E741
        V, amount, VWAP = wide["V"], wide["amount"], wide["VWAP"]

        # 1. KBar (9)
        factors = _compute_kbar(O, H, L, C)

        # 2. Price (5) — includes OVERNIGHT_RET from preclose
        factors.update(_compute_price(
            O, H, L, C, VWAP,
            raw_open=wide.get("raw_open"),
            preclose=wide.get("preclose"),
        ))

        # 3. Rolling — simple (55)
        factors.update(_compute_rolling_simple(C, H, L, V))

        # 4. Rolling — complex (75)
        factors.update(_compute_rolling_complex(C, V))

        # 5. Rolling — regression (15)
        factors.update(_compute_rolling_regression(C))

        # 6. Fundamental (18) — PE/PB/PS/PCF from fundamentals table
        factors.update(_compute_fundamentals(fundamentals, amount))

        # Stack to panel: MultiIndex (date, symbol)
        panel = pd.concat(
            {name: factors[name].stack(future_stack=True) for name in FACTOR_NAMES if name in factors},
            axis=1,
        )
        panel.index.names = ["date", "symbol"]
        panel = panel.astype("float32")

        return panel

    def compute_latest(
        self,
        storage: "DataStorage",
        target_date: str,
        pool: list[str],
    ) -> pd.DataFrame:
        """Compute factors for a single date (online inference).

        Loads 90 calendar days of history (≈60+ trading days) to satisfy
        rolling(60) warmup, then returns only the target_date row.

        Returns
        -------
        DataFrame (index=symbol, columns=177 factors).
        """

        target_dt = pd.Timestamp(target_date)
        lookback_start = (target_dt - timedelta(days=90)).strftime("%Y-%m-%d")

        # Load bars for all pool stocks
        bar_frames = []
        fund_frames = []
        for sym in pool:
            bars = storage.load_daily_bars(sym, lookback_start, target_date)
            if bars is not None and not bars.empty:
                bar_frames.append(bars)
            fund = storage.load_fundamentals_filled(sym, lookback_start, target_date)
            if fund is not None and not fund.empty:
                fund_frames.append(fund)

        if not bar_frames:
            return pd.DataFrame(columns=FACTOR_NAMES)

        all_bars = pd.concat(bar_frames, ignore_index=True)
        fundamentals = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()

        panel = self.compute(all_bars, fundamentals)

        # Filter to target_date only
        if target_dt in panel.index.get_level_values("date"):
            return panel.loc[target_dt].astype("float32")
        return pd.DataFrame(columns=FACTOR_NAMES)
