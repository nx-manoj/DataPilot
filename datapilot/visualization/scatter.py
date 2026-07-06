from __future__ import annotations

from typing import Union
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import polars as pl
import pandas as pd

from ..utils.validation import ensure_polars

_BG     = "#0f172a"
_BG_AX  = "#1e293b"
_GRID   = "#334155"
_TEXT   = "#e2e8f0"


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
    })


def scatter(
    df: Union[pd.DataFrame, pl.DataFrame],
    x: str,
    y: str,
    hue: str | None = None,
    trendline: bool = True,
) -> plt.Axes:
    """Scatter plot with an optional OLS regression trendline.

    Args:
        df:        Input DataFrame (Pandas or Polars).
        x:         Column name for the horizontal axis.
        y:         Column name for the vertical axis.
        hue:       Optional column for colour grouping.
        trendline: Draw a dashed regression line when hue is not set (default: True).
    """
    local_df, _ = ensure_polars(df)
    for col in [x, y] + ([hue] if hue else []):
        if col not in local_df.columns:
            raise ValueError(f"Column '{col}' not found.")

    pdf = local_df.to_pandas()
    _apply_theme()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=pdf, x=x, y=y, hue=hue,
        palette="mako", ax=ax,
        alpha=0.75, s=45, linewidth=0,
    )

    if trendline and not hue:
        _x = pdf[x].dropna()
        _y = pdf[y].dropna()
        common = _x.index.intersection(_y.index)
        if len(common) > 2:
            z    = np.polyfit(_x[common], _y[common], 1)
            p    = np.poly1d(z)
            xrng = np.linspace(_x.min(), _x.max(), 200)
            ax.plot(xrng, p(xrng), color="#f43f5e",
                    linewidth=2, linestyle="--", label="OLS trend", zorder=5)
            ax.legend(fontsize=9)

    ax.set_title(f"{y}  vs  {x}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(x, fontsize=11)
    ax.set_ylabel(y, fontsize=11)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    return ax
