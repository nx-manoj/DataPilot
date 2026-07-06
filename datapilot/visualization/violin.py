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


def violin(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    group_by: str | None = None,
) -> plt.Axes:
    """Violin plot — combines a box plot with a KDE for richer distribution insight.

    Args:
        df:       Input DataFrame (Pandas or Polars).
        column:   Target numeric column.
        group_by: Optional categorical column to split violins by group.
    """
    local_df, _ = ensure_polars(df)
    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found.")

    pdf = local_df.to_pandas()
    _apply_theme()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(
        data=pdf,
        x=group_by,
        y=column,
        hue=group_by,
        palette="mako" if group_by else None,
        color="#3b82f6" if not group_by else None,
        ax=ax,
        inner="quartile",
        linewidth=0.9,
        legend=False,
    )

    title = f"Violin Plot of  {column}"
    if group_by:
        title += f"  by  {group_by}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(group_by or "", fontsize=11)
    ax.set_ylabel(column, fontsize=11)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    return ax
