"""Diagnostic plots for pairs-trading screening and backtests. Agg backend, saved to outputs/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def plot_spread_and_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    ticker_a: str,
    ticker_b: str,
    signal_config=None,
) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(spread.index, spread.to_numpy(), color="tab:blue")
    ax1.set_ylabel("Spread")
    ax1.set_title(f"{ticker_a} / {ticker_b} spread & z-score")

    ax2.plot(zscore.index, zscore.to_numpy(), color="tab:orange")
    ax2.axhline(0, color="black", linewidth=0.8)
    if signal_config is not None:
        for level, style in ((signal_config.entry_z, "--"), (signal_config.stop_z, ":")):
            ax2.axhline(level, color="red", linestyle=style, linewidth=0.8)
            ax2.axhline(-level, color="red", linestyle=style, linewidth=0.8)
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    out_path = OUTPUTS_DIR / f"spread_zscore_{ticker_a}_{ticker_b}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_equity_curve(equity_curve: pd.Series, label: str = "portfolio") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve.index, equity_curve.to_numpy(), color="tab:green")
    ax.set_title(f"Equity curve — {label}")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    out_path = OUTPUTS_DIR / f"equity_curve_{label}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_cointegration_heatmap(screen_results: pd.DataFrame, title: str = "cointegration_pvalues") -> Path:
    """Heatmap of ADF p-values by pair (lower = stronger cointegration evidence).

    Pass a sector-grouped subset of screen_results for readability on large universes.
    """
    pivot = screen_results.pivot_table(index="ticker_a", columns="ticker_b", values="adf_pvalue")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Engle-Granger ADF p-value by pair")
    fig.colorbar(im, ax=ax, label="ADF p-value")
    fig.tight_layout()
    out_path = OUTPUTS_DIR / f"{title}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
