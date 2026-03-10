"""Simple backtest engine for weekly rebalancing TopN strategies.

Usage::

    from pangu.backtest.engine import BacktestEngine

    engine = BacktestEngine(top_n=10)
    result = engine.run(scores, open_prices, close_prices, benchmark)
    print(result.metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Leave 1% cash buffer to avoid floating-point precision causing negative cash
_CASH_USAGE_LIMIT = 0.99


def _lot_size(symbol: str) -> int:
    """Return minimum trading lot size for a stock."""
    # 科创板 (688/689): 200 shares
    if symbol.startswith("688") or symbol.startswith("689"):
        return 200
    # 主板 + 创业板: 100 shares
    return 100


def _price_limit_ratio(symbol: str) -> float:
    """Return price limit ratio (涨跌停幅度) for a stock."""
    # 科创板 (688/689) and 创业板 (300/301): ±20%
    if (symbol.startswith("688") or symbol.startswith("689")
            or symbol.startswith("300") or symbol.startswith("301")):
        return 0.20
    # 主板: ±10%
    return 0.10


@dataclass
class BacktestResult:
    """Container for backtest output."""

    nav: pd.Series                          # daily portfolio net asset value
    benchmark_nav: pd.Series                # daily benchmark NAV
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)  # daily holdings
    rebalance_log: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """Weekly-rebalance, equal-weight, TopN backtest engine.

    Rebalances on the first trading day of each ISO week (handles holiday Mondays).

    Parameters
    ----------
    top_n : int
        Number of stocks to hold.
    initial_capital : float
        Starting capital.
    """

    def __init__(
        self,
        *,
        top_n: int = 10,
        initial_capital: float = 1_000_000.0,
        stamp_tax: float = 0.001,
        commission: float = 0.0003,
        slippage: float = 0.001,
        min_commission: float = 5.0,
    ) -> None:
        self._top_n = top_n
        self._capital = initial_capital
        self._stamp_tax = stamp_tax
        self._commission = commission
        self._slippage = slippage
        self._min_commission = min_commission

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------

    def _buy_cost(self, trade_value: float) -> float:
        """Commission + slippage for a buy order."""
        commission = max(trade_value * self._commission, self._min_commission)
        return commission + trade_value * self._slippage

    def _sell_cost(self, trade_value: float) -> float:
        """Commission + stamp tax + slippage for a sell order."""
        commission = max(trade_value * self._commission, self._min_commission)
        return commission + trade_value * (self._stamp_tax + self._slippage)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        scores: pd.DataFrame,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        benchmark_close: pd.Series,
        start: str | None = None,
        end: str | None = None,
        universe_fn: callable | None = None,
        volume: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run backtest.

        Parameters
        ----------
        scores : DataFrame
            (date × stock) factor scores. Higher = better.
            The score on date T is used for rebalancing on the *next* trading day.
        open_prices : DataFrame
            (date × stock) **unadjusted** (real market) open prices.
        close_prices : DataFrame
            (date × stock) **unadjusted** (real market) close prices.
        benchmark_close : Series
            Benchmark (e.g. CSI300) close prices.
        start, end : str, optional
            Backtest date range (inclusive). Defaults to scores range.
        universe_fn : callable, optional
            Function(date_str) → set[str] returning tradeable stock symbols
            for a given date. Used for point-in-time constituent filtering.
            If None, all stocks in scores are eligible.
        """
        # Align date range
        dates = close_prices.index.sort_values()
        if start:
            dates = dates[dates >= start]
        if end:
            dates = dates[dates <= end]

        if len(dates) < 2:
            raise ValueError(f"Not enough dates for backtest: {len(dates)}")

        # Align all inputs to common dates and stocks
        common_stocks = scores.columns.intersection(
            close_prices.columns
        ).intersection(open_prices.columns)
        scores = scores.reindex(index=dates, columns=common_stocks)
        open_prices = open_prices.reindex(index=dates, columns=common_stocks)
        close_prices = close_prices.reindex(index=dates, columns=common_stocks)
        if volume is not None:
            volume = volume.reindex(index=dates, columns=common_stocks)
        benchmark_close = benchmark_close.sort_index()  # keep pre-start dates for base

        logger.info(
            "Backtest: %s ~ %s, %d days, %d stocks, top_n=%d",
            dates[0].date(), dates[-1].date(), len(dates),
            len(common_stocks), self._top_n,
        )

        # Forward-fill close prices so suspended stocks retain last-known value
        close_prices = close_prices.ffill()

        # Run simulation
        nav_values, rebal_log, holdings_records = self._simulate(
            dates, scores, open_prices, close_prices, universe_fn, volume,
        )

        # Build results
        nav = pd.Series(nav_values, index=dates, name="nav")

        # Benchmark NAV: use the close before backtest start as base
        # (matches JoinQuant: benchmark return includes first-day movement)
        pre_start_bench = benchmark_close[benchmark_close.index < dates[0]]
        if len(pre_start_bench) > 0:
            bench_base = pre_start_bench.iloc[-1]
        else:
            bench_base = benchmark_close.iloc[0]
        if not np.isfinite(bench_base) or bench_base <= 0:
            raise ValueError("Invalid benchmark base price")
        bench_nav = (benchmark_close / bench_base * self._capital).reindex(dates)
        bench_nav.name = "benchmark"

        metrics = self._compute_metrics(nav, bench_nav, rebal_log)

        return BacktestResult(
            nav=nav,
            benchmark_nav=bench_nav,
            holdings=pd.DataFrame(holdings_records),
            rebalance_log=rebal_log,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------

    def _simulate(
        self,
        dates: pd.DatetimeIndex,
        scores: pd.DataFrame,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        universe_fn: callable | None = None,
        volume: pd.DataFrame | None = None,
    ) -> tuple[list[float], list[dict], list[dict]]:
        """Day-by-day simulation loop."""
        cash = self._capital
        holdings: dict[str, int] = {}
        nav_values: list[float] = []
        rebal_log: list[dict] = []
        holdings_records: list[dict] = []

        prev_date = None

        for i, date in enumerate(dates):
            # Rebalance on first trading day of each new ISO week,
            # plus the very first trading day of the backtest.
            is_first_day = (prev_date is None and i == 0)
            is_new_week = (prev_date is not None
                           and date.isocalendar()[1] != prev_date.isocalendar()[1])
            is_rebal = is_first_day or is_new_week

            if is_rebal:
                # First day: use own score (no T-1 available); otherwise T-1
                score_date = date if is_first_day else prev_date
                prev_scores = scores.loc[score_date].dropna()

                if universe_fn is not None:
                    universe = universe_fn(date.strftime("%Y-%m-%d"))
                    prev_scores = prev_scores[prev_scores.index.isin(universe)]

                if len(prev_scores) >= self._top_n:
                    target = prev_scores.nlargest(self._top_n).index.tolist()
                else:
                    target = prev_scores.sort_values(ascending=False).index.tolist()

                today_open = open_prices.loc[date]
                prev_close_prices = (close_prices.loc[prev_date]
                                     if prev_date is not None
                                     else today_open)
                today_volume = volume.loc[date] if volume is not None and date in volume.index else None
                cash, holdings, log_entry = self._rebalance(
                    date, cash, holdings, target, today_open, prev_close_prices,
                    today_volume,
                )
                if log_entry:
                    rebal_log.append(log_entry)

            # Daily NAV + holdings record (single pass)
            today_close = close_prices.loc[date]
            port_value = cash
            for stock, shares in sorted(holdings.items()):
                price = today_close.get(stock, np.nan)
                mv = shares * price if np.isfinite(price) else 0.0
                port_value += mv
                holdings_records.append({
                    "date": date, "symbol": stock, "shares": shares,
                    "market_value": mv,
                    "weight": 0.0,  # filled below
                })

            # Fill weights now that port_value is known
            if holdings:
                start_idx = len(holdings_records) - len(holdings)
                for j in range(start_idx, len(holdings_records)):
                    if port_value > 0:
                        holdings_records[j]["weight"] = (
                            holdings_records[j]["market_value"] / port_value
                        )

            nav_values.append(port_value)
            prev_date = date

        return nav_values, rebal_log, holdings_records

    # ------------------------------------------------------------------
    # Rebalance
    # ------------------------------------------------------------------

    @staticmethod
    def _is_at_limit(
        open_price: float, prev_close: float, direction: str, symbol: str = "",
    ) -> bool:
        """Check if stock is at price limit.

        direction='up':   涨停 → can't buy
        direction='down': 跌停 → can't sell

        Uses ±20% for 科创板 (688/689) and 创业板 (300/301), ±10% for others.
        """
        if not (np.isfinite(prev_close) and np.isfinite(open_price) and prev_close > 0):
            return False
        ratio = _price_limit_ratio(symbol)
        if direction == "up":
            return open_price >= round(prev_close * (1 + ratio), 2)
        return open_price <= round(prev_close * (1 - ratio), 2)

    def _rebalance(
        self,
        date: pd.Timestamp,
        cash: float,
        holdings: dict[str, int],
        target: list[str],
        open_prices: pd.Series,
        prev_close: pd.Series,
        volume: pd.Series | None = None,
    ) -> tuple[float, dict[str, int], dict | None]:
        """Execute rebalance: sell non-target, buy target at open prices."""
        sells: list[str] = []
        buys: list[str] = []
        turnover = 0.0
        target_set = set(target)

        def _is_suspended(stock: str) -> bool:
            """Check if stock is suspended (volume=0 or NaN)."""
            if volume is None:
                return False
            v = volume.get(stock, np.nan)
            return not (np.isfinite(v) and v > 0)

        # --- Phase 1: Sell non-target holdings ---
        new_holdings: dict[str, int] = {}
        for stock, shares in sorted(holdings.items()):
            if stock not in target_set:
                price = open_prices.get(stock, np.nan)
                pc = prev_close.get(stock, np.nan)
                if (np.isfinite(price) and price > 0
                        and not _is_suspended(stock)
                        and not self._is_at_limit(price, pc, "down", stock)):
                    sell_value = shares * price
                    cash += sell_value - self._sell_cost(sell_value)
                    turnover += sell_value
                    sells.append(stock)
                else:
                    new_holdings[stock] = shares
            else:
                new_holdings[stock] = shares

        # --- Phase 2: Calculate equal-weight allocation ---
        total_value = cash
        for stock, shares in sorted(new_holdings.items()):
            price = open_prices.get(stock, np.nan)
            if np.isfinite(price):
                total_value += shares * price

        # Filter to tradeable targets (preserve score-descending order from target)
        tradeable = [
            s for s in target
            if np.isfinite(open_prices.get(s, np.nan))
            and open_prices.get(s, np.nan) > 0
            and not _is_suspended(s)
            and not self._is_at_limit(
                open_prices.get(s, np.nan), prev_close.get(s, np.nan), "up", s
            )
        ]
        # Reserve cash before equal-weight split (matches JQ's cash_reserve logic)
        available_value = total_value * _CASH_USAGE_LIMIT
        target_value = available_value / max(len(tradeable), 1)

        # --- Phase 3: Buy / rebalance each target stock ---
        for stock in tradeable:
            price = open_prices.get(stock, np.nan)
            current_shares = new_holdings.get(stock, 0)
            current_value = current_shares * price
            diff = target_value - current_value
            lot = _lot_size(stock)

            if diff > 0:
                # Buy: cap by available cash
                budget = min(diff, cash)
                if budget <= 0:
                    new_holdings.setdefault(stock, current_shares)
                    continue
                # Use open price directly for share calculation
                # (slippage only affects cost deduction, not share count)
                buy_shares = int(budget / price / lot) * lot
                if buy_shares > 0:
                    cost_basis = buy_shares * price
                    cash -= cost_basis + self._buy_cost(cost_basis)
                    new_holdings[stock] = current_shares + buy_shares
                    turnover += cost_basis
                    if current_shares == 0:
                        buys.append(stock)
                else:
                    new_holdings.setdefault(stock, current_shares)

            elif diff < -price * lot:
                # Sell excess, check limit down
                pc = prev_close.get(stock, np.nan)
                if self._is_at_limit(price, pc, "down", stock):
                    continue
                sell_shares = int(-diff / price / lot) * lot
                sell_shares = min(sell_shares, current_shares)
                if sell_shares > 0:
                    sell_value = sell_shares * price
                    cash += sell_value - self._sell_cost(sell_value)
                    new_holdings[stock] = current_shares - sell_shares
                    turnover += sell_value

        log_entry = None
        if sells or buys:
            log_entry = {
                "date": date,
                "sells": sells,
                "buys": buys,
                "n_holdings": len(new_holdings),
                "turnover": turnover,
                "cash_pct": cash / total_value if total_value > 0 else 0,
            }

        return cash, new_holdings, log_entry

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, nav: pd.Series, benchmark_nav: pd.Series, rebal_log: list[dict],
    ) -> dict:
        """Compute standard backtest metrics."""
        ret = nav.pct_change().dropna()

        n_days = len(ret)
        n_years = n_days / 252.0

        # Strategy return based on initial capital (not first-day NAV)
        total_return = nav.iloc[-1] / self._capital - 1
        # Benchmark return: bench_nav already based on pre-start close
        bench_total = benchmark_nav.iloc[-1] / self._capital - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

        cum_max = nav.cummax()
        drawdown = (nav - cum_max) / cum_max
        max_dd = drawdown.min()

        win_rate = (ret > 0).mean()
        excess = total_return - bench_total

        total_turnover = sum(e.get("turnover", 0.0) for e in rebal_log)
        avg_nav = nav.mean()
        annual_turnover = (
            total_turnover / avg_nav / n_years
            if (avg_nav > 0 and n_years > 0) else 0.0
        )

        return {
            "total_return": round(float(total_return), 4),
            "annual_return": round(float(annual_return), 4),
            "benchmark_return": round(float(bench_total), 4),
            "excess_return": round(float(excess), 4),
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(max_dd), 4),
            "win_rate": round(float(win_rate), 3),
            "annual_turnover": round(float(annual_turnover), 2),
            "n_rebalances": len(rebal_log),
            "n_days": n_days,
            "n_years": round(n_years, 1),
        }


# ------------------------------------------------------------------
# Universe helper
# ------------------------------------------------------------------


def make_universe_fn(storage: object, volume_wide: pd.DataFrame) -> callable:
    """Create point-in-time constituent filter with suspended stock exclusion.

    Returns a function ``fn(date_str) -> set[str]`` suitable for passing
    as *universe_fn* to :meth:`BacktestEngine.run`.
    """
    cache: dict[str, set[str]] = {}

    def fn(date_str: str) -> set[str]:
        if date_str not in cache:
            constituents = set(storage.load_constituents_for_date(date_str))
            ts = pd.Timestamp(date_str)
            if ts in volume_wide.index:
                vol = volume_wide.loc[ts]
                constituents -= set(vol[(vol == 0) | vol.isna()].index)
            cache[date_str] = constituents
        return cache[date_str]

    return fn
