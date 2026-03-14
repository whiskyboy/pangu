"""Backtest equity curve plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_equity_curve(
    nav: pd.Series,
    benchmark_nav: pd.Series,
    strategy: str,
    output_path: str | Path,
    initial_capital: float = 1_000_000,
) -> Path:
    """Plot strategy vs benchmark cumulative returns with colored fill."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook", palette="muted")

    # Cumulative returns based on initial capital (consistent with metrics)
    strat_ret = nav / initial_capital - 1
    bench_ret = benchmark_nav / initial_capital - 1
    dates = strat_ret.index

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, strat_ret, color="#2563eb", linewidth=1.4, label=f"Strategy ({strategy})")
    ax.plot(dates, bench_ret, color="#dc2626", linewidth=1.4, label="Benchmark (CSI300)")

    # Fill: red where strategy > benchmark, green where strategy < benchmark
    ax.fill_between(
        dates, strat_ret, bench_ret,
        where=strat_ret >= bench_ret,
        interpolate=True, alpha=0.20, color="#dc2626", label="Excess return",
    )
    ax.fill_between(
        dates, strat_ret, bench_ret,
        where=strat_ret < bench_ret,
        interpolate=True, alpha=0.20, color="#16a34a", label="Underperformance",
    )

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")

    # Final return annotations
    ax.annotate(
        f"{strat_ret.iloc[-1]:+.1%}",
        xy=(dates[-1], strat_ret.iloc[-1]), xytext=(8, 0),
        textcoords="offset points", fontsize=9, fontweight="bold", color="#2563eb",
    )
    ax.annotate(
        f"{bench_ret.iloc[-1]:+.1%}",
        xy=(dates[-1], bench_ret.iloc[-1]), xytext=(8, 0),
        textcoords="offset points", fontsize=9, fontweight="bold", color="#dc2626",
    )

    ax.set_title(
        f"Backtest: {strategy}  ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
