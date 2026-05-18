"""Simple backtest engine for periodic TopN rebalancing strategies.

Cadence is controlled by ``pangu.rebalance.RebalanceSchedule`` (default
``weekly:1`` = every Monday, deferring to next trading day if Monday is
non-trading) — equivalent to the historical ISO-week-first behaviour.

Usage::

    from pangu.backtest.engine import BacktestEngine

    engine = BacktestEngine(top_n=10)
    result = engine.run(scores, open_prices, close_prices, benchmark)
    print(result.metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pangu.rebalance import RebalanceSchedule

logger = logging.getLogger(__name__)

# Leave 1% cash buffer to avoid floating-point precision causing negative cash
_CASH_USAGE_LIMIT = 0.99


def _is_star_market(symbol: str) -> bool:
    """Check if a stock is on the STAR Market (科创板)."""
    return symbol.startswith("688") or symbol.startswith("689")


def _lot_size(symbol: str) -> int:
    """Return minimum trading lot size for a stock."""
    if _is_star_market(symbol):
        return 200
    return 100


def _price_limit_ratio(symbol: str, is_st: bool = False) -> float:
    """Return price limit ratio (涨跌停幅度) for a stock.

    STAR (688/689) and GEM (300/301) always use ±20% (even for ST on GEM).
    Main-board ST stocks use ±5%; all other main-board stocks use ±10%.
    """
    if _is_star_market(symbol) or symbol.startswith("300") or symbol.startswith("301"):
        return 0.20
    if is_st:
        return 0.05
    return 0.10


def _round_price(value: float) -> float:
    """Round price to 2 decimal places using 四舍五入 (standard rounding).

    Python's built-in ``round()`` uses banker's rounding which can differ
    from the exchange rule (e.g. ``round(5.885, 2) == 5.88`` instead of
    ``5.89``).  Use ``Decimal`` with ``ROUND_HALF_UP`` to match A-share
    exchange limit-price calculation.
    """
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass
class BacktestResult:
    """Container for backtest output."""

    nav: pd.Series  # daily portfolio net asset value
    benchmark_nav: pd.Series  # daily benchmark NAV
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
        top_n: int = 30,
        n_drop: int = 0,
        initial_capital: float = 1_000_000.0,
        stamp_tax: float = 0.001,
        commission: float = 0.0003,
        slippage: float = 0.001,
        min_commission: float = 5.0,
        max_per_sector: int | None = None,
        schedule: "RebalanceSchedule | None" = None,
    ) -> None:
        from pangu.rebalance import RebalanceSchedule as _RS

        self._top_n = top_n
        self._n_drop = n_drop
        self._capital = initial_capital
        self._stamp_tax = stamp_tax
        self._commission = commission
        self._slippage = slippage
        self._min_commission = min_commission
        self._max_per_sector = max_per_sector
        # Default schedule (weekly:1) ≈ original ISO-week-first behaviour
        self._schedule = schedule if schedule is not None else _RS("weekly", 1)

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
        adj_factor: pd.DataFrame | None = None,
        is_st: pd.DataFrame | None = None,
        exclude_prefixes: tuple[str, ...] = ("688", "689"),
        sector_map: dict[str, str] | None = None,
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
        adj_factor : DataFrame, optional
            (date × stock) adjustment factors for dividend detection.
            When provided, dividend cash is credited to the portfolio.
        is_st : DataFrame, optional
            (date × stock) ST status flags (1 = ST, 0 = normal).
            When provided, ST stocks are excluded from buying and
            held-ST stocks are force-sold. Main-board ST uses ±5% limits.
        exclude_prefixes : tuple[str, ...], optional
            Stock code prefixes to exclude from selection (default: 688/689
            STAR market stocks which cannot be traded on most platforms).
            Set to () to disable.
        sector_map : dict[str, str], optional
            Mapping of symbol → sector classification. Required when
            ``max_per_sector`` is set. Stocks missing from the map are
            grouped under "未知" sector with the same cap.
        """
        from pangu.backtest.target_provider import ScoreBasedProvider

        # Wrap the legacy score-based selection inside the new provider API
        # and delegate to run_with_provider. universe_fn / sector_map / cap
        # are now owned by the provider, not the engine itself.
        common_stocks = scores.columns.intersection(close_prices.columns).intersection(open_prices.columns)
        if exclude_prefixes:
            common_stocks = pd.Index([s for s in common_stocks if not any(s.startswith(p) for p in exclude_prefixes)])
        scores_aligned = scores.reindex(columns=common_stocks)

        provider = ScoreBasedProvider(
            scores_aligned,
            top_n=self._top_n,
            n_drop=self._n_drop,
            universe_fn=universe_fn,
            sector_map=sector_map,
            max_per_sector=self._max_per_sector,
        )

        return self.run_with_provider(
            provider,
            open_prices=open_prices,
            close_prices=close_prices,
            benchmark_close=benchmark_close,
            start=start,
            end=end,
            volume=volume,
            adj_factor=adj_factor,
            is_st=is_st,
            exclude_prefixes=exclude_prefixes,
            tradeable_universe=common_stocks,
        )

    def run_with_provider(
        self,
        target_provider,
        *,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        benchmark_close: pd.Series,
        start: str | None = None,
        end: str | None = None,
        volume: pd.DataFrame | None = None,
        adj_factor: pd.DataFrame | None = None,
        is_st: pd.DataFrame | None = None,
        exclude_prefixes: tuple[str, ...] = ("688", "689"),
        tradeable_universe: pd.Index | None = None,
    ) -> BacktestResult:
        """Run backtest driven by a ``TargetProvider``.

        This is the new public API. ``run(scores=..., ...)`` is preserved
        as a thin wrapper that constructs a ``ScoreBasedProvider``.
        ``ReplayProvider`` callers should use this method directly.
        """
        # Align date range
        dates = close_prices.index.sort_values()
        if start:
            dates = dates[dates >= start]
        if end:
            dates = dates[dates <= end]

        if len(dates) < 2:
            raise ValueError(f"Not enough dates for backtest: {len(dates)}")

        # Build the eligible-stock universe
        if tradeable_universe is not None:
            common_stocks = tradeable_universe
        else:
            common_stocks = close_prices.columns.intersection(open_prices.columns)
            if exclude_prefixes:
                common_stocks = pd.Index(
                    [s for s in common_stocks if not any(s.startswith(p) for p in exclude_prefixes)]
                )

        open_prices = open_prices.reindex(index=dates, columns=common_stocks)
        close_prices = close_prices.reindex(index=dates, columns=common_stocks)
        if volume is not None:
            volume = volume.reindex(index=dates, columns=common_stocks)
        if adj_factor is not None:
            adj_factor = adj_factor.reindex(index=dates, columns=common_stocks).ffill()
        if is_st is not None:
            is_st = is_st.reindex(index=dates, columns=common_stocks).fillna(0)
        benchmark_close = benchmark_close.sort_index()  # keep pre-start dates for base

        logger.info(
            "Backtest: %s ~ %s, %d days, %d stocks, top_n=%d, n_drop=%d%s",
            dates[0].date(),
            dates[-1].date(),
            len(dates),
            len(common_stocks),
            self._top_n,
            self._n_drop,
            f", max_per_sector={self._max_per_sector}" if self._max_per_sector else "",
        )

        close_prices = close_prices.ffill()

        nav_values, rebal_log, holdings_records = self._simulate(
            dates,
            target_provider,
            open_prices,
            close_prices,
            volume,
            adj_factor,
            is_st,
        )

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
        target_provider,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        adj_factor: pd.DataFrame | None = None,
        is_st: pd.DataFrame | None = None,
    ) -> tuple[list[float], list[dict], list[dict]]:
        """Day-by-day simulation loop."""
        cash = self._capital
        holdings: dict[str, int] = {}
        nav_values: list[float] = []
        rebal_log: list[dict] = []
        holdings_records: list[dict] = []

        prev_date = None
        # Trading-day predicate for RebalanceSchedule: dates inside this
        # backtest window are by definition trading days.
        trading_set = {ts.date() for ts in dates}
        is_trading_day = trading_set.__contains__

        for i, date in enumerate(dates):
            # --- Dividend processing ---
            # Detect ex-dividend events via adj_factor changes and credit
            # cash for held stocks. Uses 20% tax rate (holding < 1 month).
            if adj_factor is not None and prev_date is not None and holdings:
                prev_adj = adj_factor.loc[prev_date]
                curr_adj = adj_factor.loc[date]
                prev_cls = close_prices.loc[prev_date]
                for stock, shares in holdings.items():
                    old_af = prev_adj.get(stock, np.nan)
                    new_af = curr_adj.get(stock, np.nan)
                    pc = prev_cls.get(stock, np.nan)
                    if (
                        np.isfinite(old_af)
                        and np.isfinite(new_af)
                        and np.isfinite(pc)
                        and old_af > 0
                        and new_af > old_af > 0
                    ):
                        div_per_share = pc * (1 - old_af / new_af)
                        cash += shares * div_per_share * 0.80

            # Rebalance trigger:
            #   day-0 always rebalances (bootstrap portfolio).
            #   otherwise delegate to RebalanceSchedule (configurable cadence).
            is_first_day = prev_date is None and i == 0
            is_rebal = is_first_day or self._schedule.matches(date.date(), is_trading_day)

            if is_rebal:
                # First day: use own date as score reference (no T-1 available)
                score_date = date if is_first_day else prev_date

                target = target_provider.get_target(date, score_date, dict(holdings))

                today_open = open_prices.loc[date]
                prev_close_prices = close_prices.loc[prev_date] if prev_date is not None else today_open
                today_volume = volume.loc[date] if volume is not None and date in volume.index else None
                today_st = is_st.loc[date] if is_st is not None and date in is_st.index else None
                cash, holdings, log_entry = self._rebalance(
                    date,
                    cash,
                    holdings,
                    target,
                    today_open,
                    prev_close_prices,
                    today_volume,
                    today_st,
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
                holdings_records.append(
                    {
                        "date": date,
                        "symbol": stock,
                        "shares": shares,
                        "market_value": mv,
                        "weight": 0.0,  # filled below
                    }
                )

            # Fill weights now that port_value is known
            if holdings:
                start_idx = len(holdings_records) - len(holdings)
                for j in range(start_idx, len(holdings_records)):
                    if port_value > 0:
                        holdings_records[j]["weight"] = holdings_records[j]["market_value"] / port_value

            nav_values.append(port_value)
            prev_date = date

        return nav_values, rebal_log, holdings_records

    # ------------------------------------------------------------------
    # Rebalance
    # ------------------------------------------------------------------

    @staticmethod
    def _is_at_limit(
        open_price: float,
        prev_close: float,
        direction: str,
        symbol: str = "",
        is_st: bool = False,
    ) -> bool:
        """Check if stock is at price limit.

        direction='up':   涨停 → can't buy
        direction='down': 跌停 → can't sell

        Uses ±20% for 科创板 (688/689) and 创业板 (300/301),
        ±5% for main-board ST, ±10% for other main-board stocks.
        """
        if not (np.isfinite(prev_close) and np.isfinite(open_price) and prev_close > 0):
            return False
        ratio = _price_limit_ratio(symbol, is_st=is_st)
        if direction == "up":
            return open_price >= _round_price(prev_close * (1 + ratio))
        return open_price <= _round_price(prev_close * (1 - ratio))

    def _rebalance(
        self,
        date: pd.Timestamp,
        cash: float,
        holdings: dict[str, int],
        target: list[str],
        open_prices: pd.Series,
        prev_close: pd.Series,
        volume: pd.Series | None = None,
        is_st: pd.Series | None = None,
    ) -> tuple[float, dict[str, int], dict | None]:
        """Execute rebalance using 5-step allocation strategy.

        Steps:
          0. Classify stocks into NeedToSell / NeedToAdjust / NeedToBuy.
             Held ST stocks are force-classified as NeedToSell.
          1. Sell NeedToSell (skip suspended, limit-down, no-price → "stuck").
          2. Build AdjustablePool (NeedToAdjust minus suspended/no-price),
             CanBuy (NeedToBuy minus suspended/limit-up/no-price/ST),
             TargetSet = CanBuy ∪ AdjustablePool (score-descending).
          3. Allocatable = cash + AdjustablePool market value (excludes stuck).
          4. Equal-weight iterate TargetSet: buy/sell with directional limit
             checks; partial-fill on cash shortage.
        """
        sells: list[str] = []
        buys: list[str] = []
        turnover = 0.0
        target_set = set(target)
        holdings_set = set(holdings.keys())

        def _is_suspended(stock: str) -> bool:
            if volume is None:
                return False
            v = volume.get(stock, np.nan)
            return not (np.isfinite(v) and v > 0)

        def _has_valid_price(stock: str) -> bool:
            p = open_prices.get(stock, np.nan)
            return np.isfinite(p) and p > 0

        def _stock_is_st(stock: str) -> bool:
            if is_st is None:
                return False
            v = is_st.get(stock, 0)
            return bool(v == 1)

        # --- Step 0: Classify ---
        # Held ST stocks are force-sold regardless of target membership
        need_to_sell = [s for s in sorted(holdings) if s not in target_set or _stock_is_st(s)]
        need_to_adjust = [s for s in target if s in holdings_set and not _stock_is_st(s)]
        need_to_buy = [s for s in target if s not in holdings_set and not _stock_is_st(s)]

        # --- Step 1: Sell NeedToSell ---
        new_holdings: dict[str, int] = {}
        for stock in need_to_sell:
            shares = holdings[stock]
            price = open_prices.get(stock, np.nan)
            pc = prev_close.get(stock, np.nan)
            if (
                _has_valid_price(stock)
                and not _is_suspended(stock)
                and not self._is_at_limit(price, pc, "down", stock, is_st=_stock_is_st(stock))
            ):
                sell_value = shares * price
                cash += sell_value - self._sell_cost(sell_value)
                turnover += sell_value
                sells.append(stock)
            else:
                # Stuck: can't sell — remains in holdings but excluded from allocation
                new_holdings[stock] = shares

        # Carry over NeedToAdjust holdings
        for stock in need_to_adjust:
            new_holdings[stock] = holdings[stock]

        # --- Step 2: Build allocation pools ---
        adjustable_pool = [s for s in need_to_adjust if _has_valid_price(s) and not _is_suspended(s)]
        can_buy = [
            s
            for s in need_to_buy
            if _has_valid_price(s)
            and not _is_suspended(s)
            and not self._is_at_limit(
                open_prices.get(s, np.nan),
                prev_close.get(s, np.nan),
                "up",
                s,
                is_st=_stock_is_st(s),
            )
        ]
        # TargetSet preserves score-descending order from `target`
        adjustable_set = set(adjustable_pool)
        can_buy_set = set(can_buy)
        target_alloc = [s for s in target if s in adjustable_set or s in can_buy_set]

        # --- Step 3: Compute allocatable total (≠ actual NAV) ---
        allocatable = cash
        for stock in adjustable_pool:
            price = open_prices.get(stock, np.nan)
            allocatable += holdings[stock] * price

        if not target_alloc:
            return cash, new_holdings, None

        available = allocatable * _CASH_USAGE_LIMIT
        per_stock = available / len(target_alloc)

        # --- Step 4: Iterate TargetSet in score order ---
        for stock in target_alloc:
            price = open_prices.get(stock, np.nan)
            pc = prev_close.get(stock, np.nan)
            current_shares = new_holdings.get(stock, 0)
            lot = _lot_size(stock)

            desired_shares = int(per_stock / price / lot) * lot
            share_diff = desired_shares - current_shares

            if share_diff > 0:
                # Buy more
                if self._is_at_limit(price, pc, "up", stock, is_st=_stock_is_st(stock)):
                    new_holdings.setdefault(stock, current_shares)
                    continue
                cost_basis = share_diff * price
                if cost_basis > cash:
                    # Partial fill: buy what we can afford
                    share_diff = int(cash / price / lot) * lot
                    if share_diff <= 0:
                        new_holdings.setdefault(stock, current_shares)
                        continue
                    cost_basis = share_diff * price
                cash -= cost_basis + self._buy_cost(cost_basis)
                new_holdings[stock] = current_shares + share_diff
                turnover += cost_basis
                if current_shares == 0:
                    buys.append(stock)

            elif share_diff < 0:
                # Sell excess
                if self._is_at_limit(price, pc, "down", stock, is_st=_stock_is_st(stock)):
                    new_holdings.setdefault(stock, current_shares)
                    continue
                sell_shares = min(-share_diff, current_shares)
                if sell_shares > 0:
                    sell_value = sell_shares * price
                    cash += sell_value - self._sell_cost(sell_value)
                    remaining = current_shares - sell_shares
                    if remaining > 0:
                        new_holdings[stock] = remaining
                    else:
                        new_holdings.pop(stock, None)
                    turnover += sell_value

            else:
                if current_shares > 0:
                    new_holdings.setdefault(stock, current_shares)

        log_entry = None
        if sells or buys:
            log_entry = {
                "date": date,
                "sells": sells,
                "buys": buys,
                "n_holdings": len(new_holdings),
                "turnover": turnover,
                "cash_pct": cash / allocatable if allocatable > 0 else 0,
            }

        return cash, new_holdings, log_entry

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        nav: pd.Series,
        benchmark_nav: pd.Series,
        rebal_log: list[dict],
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
        annual_turnover = total_turnover / avg_nav / n_years if (avg_nav > 0 and n_years > 0) else 0.0

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


def make_universe_fn(storage: object) -> callable:
    """Create point-in-time constituent filter.

    Returns a function ``fn(date_str) -> set[str]`` suitable for passing
    as *universe_fn* to :meth:`BacktestEngine.run`.

    Note: Suspended stocks are NOT filtered here. T-day suspension status
    is future information at selection time (T-1 scores). Suspension is
    handled at execution time in the rebalance logic.
    """
    cache: dict[str, set[str]] = {}

    def fn(date_str: str) -> set[str]:
        if date_str not in cache:
            cache[date_str] = set(storage.load_constituents_for_date(date_str))
        return cache[date_str]

    return fn
