"""Cross-sectional factor matrix builder for the LLM rebalance evidence pack.

Public helper consumed by ``pangu.tasks.generate_signals._build_candidate_factors``
to surface per-stock factor snapshots to the LLM-TopkDropout debate prompt.
"""

from __future__ import annotations

import pandas as pd

_OHLCV_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def build_factor_matrix(
    pool: list[str],
    tech_df: dict[str, pd.DataFrame],
    fund_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a (rows=symbols, cols=factors) DataFrame for cross-sectional view.

    Args:
        pool: symbols whose factor snapshot we want.
        tech_df: ``symbol → DataFrame`` of technical factors (last row used).
        fund_df: DataFrame indexed by symbol with fundamental columns
            (PE/PB/ROE/...). May be empty or ``None``.

    Returns:
        DataFrame indexed by symbol; columns are technical + fundamental
        factor names. Empty DataFrame if no symbol has technical bars.
    """
    rows: list[dict[str, float]] = []
    valid_symbols: list[str] = []

    for sym in pool:
        bars = tech_df.get(sym)
        if bars is None or bars.empty:
            continue

        tech_last: dict[str, float] = {}
        for col in bars.columns:
            if col in _OHLCV_COLUMNS:
                continue
            val = bars[col].iloc[-1]
            tech_last[col] = float(val) if val is not None else float("nan")

        fund_row: dict[str, float] = {}
        if fund_df is not None and not fund_df.empty and sym in fund_df.index:
            for col in fund_df.columns:
                val = fund_df.loc[sym, col]
                fund_row[col] = float(val) if val is not None else float("nan")

        rows.append({**tech_last, **fund_row})
        valid_symbols.append(sym)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=valid_symbols)
