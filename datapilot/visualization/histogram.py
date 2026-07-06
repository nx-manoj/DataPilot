from __future__ import annotations

from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import polars as pl
import pandas as pd

from ..utils.validation import ensure_polars

# ── DataPilot dark theme ──────────────────────────────────────────────────────
_BG      = "#0f172a"
_BG_AX   = "#1e293b"
_GRID    = "#334155"
_TEXT    = "#e2e8f0"
_ACCENT  = "#3b82f6"
_ACCENT2 = "#8b5cf6"


def _apply_theme() -> None:
    sns.set_theme(style="darkgrid")
    mpl.rcParams.update({
        "figure.facecolor": _BG,
        "axes.facecolor":   _BG_AX,
        "axes.edgecolor":   _GRID,
        "axes.labelcolor":  _TEXT,
        "axes.titlecolor":  _TEXT,
        "xtick.color":      _TEXT,
        "ytick.color":      _TEXT,
        "grid.color":       _GRID,
        "text.color":       _TEXT,
        "legend.facecolor": _BG_AX,
        "legend.edgecolor": _GRID,
        "figure.dpi":       120,
        "font.family":      "sans-serif",
    })


def hist(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    bins: Union[int, str] = "auto",
    hue: Optional[str] = None,
    color: str = _ACCENT,
) -> plt.Axes:
    """Plots a styled distribution histogram with a smooth KDE overlay.

    Args:
        df:     Input DataFrame (Pandas or Polars).
        column: Target numeric column name.
        bins:   Number of bins, or 'auto' for optimal bin width (default).
        hue:    Optional grouping column for colour-split distributions.
        color:  Bar fill colour when hue is not set.
    """
    local_df, _ = ensure_polars(df)

    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    pdf = local_df.to_pandas()
    _apply_theme()

    fig, ax = plt.subplots(figsize=(10, 6))

    if hue:
        sns.histplot(
            data=pdf, x=column, hue=hue, kde=True,
            palette="mako", bins=bins, ax=ax,
            edgecolor="none", alpha=0.75,
        )
    else:
        sns.histplot(
            data=pdf, x=column, kde=True, bins=bins, ax=ax,
            color=color, edgecolor="none", alpha=0.8,
            line_kws={"color": _ACCENT2, "linewidth": 2.5},
        )
        # Overlay mean and median lines
        series = pdf[column].dropna()
        ax.axvline(series.mean(),   color="#f43f5e", linestyle="--",
                   linewidth=1.5, label=f"Mean: {series.mean():.2f}")
        ax.axvline(series.median(), color="#10b981", linestyle="-.",
                   linewidth=1.5, label=f"Median: {series.median():.2f}")
        ax.legend(fontsize=9)

    # Descriptive subtitle
    series   = pdf[column].dropna()
    subtitle = (
        f"n={len(series):,} | mean={series.mean():.3f} | "
        f"std={series.std():.3f} | skew={series.skew():.2f}"
    )
    ax.set_title(f"Distribution of  {column}", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_xlabel(column, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    fig.text(0.5, 0.96, subtitle, ha="center", fontsize=8.5,
             color="#94a3b8", transform=fig.transFigure)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return ax
