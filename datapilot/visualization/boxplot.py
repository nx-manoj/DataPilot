from __future__ import annotations

from typing import Union, Optional
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
_ACCENT = "#8b5cf6"


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


def box(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    group_by: Optional[str] = None,
    orient: str = "v",
) -> None:
    """Plots a styled box plot with quartile annotations and outlier highlights.

    Args:
        df:       Input DataFrame (Pandas or Polars).
        column:   Target numeric column name.
        group_by: Optional categorical column to split boxes by group.
        orient:   Orientation — 'v' (vertical, default) or 'h' (horizontal).
    """
    local_df, _ = ensure_polars(df)

    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
    if group_by and group_by not in local_df.columns:
        raise ValueError(f"group_by column '{group_by}' not found in the dataset.")

    pdf = local_df.to_pandas()
    _apply_theme()

    fig, ax = plt.subplots(figsize=(10, 6))

    x_kw = group_by if orient == "v" else column
    y_kw = column   if orient == "v" else group_by

    sns.boxplot(
        data=pdf,
        x=x_kw if group_by else None,
        y=y_kw if group_by else column,
        hue=group_by if group_by else None,
        palette="mako" if group_by else None,
        color=_ACCENT if not group_by else None,
        ax=ax,
        linewidth=1.4,
        flierprops={
            "marker": "o",
            "markerfacecolor": "#f43f5e",
            "markersize": 4,
            "alpha": 0.7,
            "markeredgewidth": 0,
        },
        medianprops={"color": "#10b981", "linewidth": 2.2},
        whiskerprops={"color": _TEXT, "linewidth": 1.2},
        capprops={"color": _TEXT, "linewidth": 1.5},
    )

    # Annotate IQR stats in the subtitle
    series = pdf[column].dropna()
    q1, med, q3 = np.percentile(series, [25, 50, 75])
    iqr = q3 - q1
    subtitle = (
        f"Q1={q1:.2f} | Median={med:.2f} | Q3={q3:.2f} | "
        f"IQR={iqr:.2f} | n={len(series):,}"
    )

    title = f"Box Plot of  {column}"
    if group_by:
        title += f"  grouped by  {group_by}"

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(x_kw or "", fontsize=11)
    ax.set_ylabel(y_kw or column, fontsize=11)
    fig.text(0.5, 0.96, subtitle, ha="center", fontsize=8.5,
             color="#94a3b8", transform=fig.transFigure)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
