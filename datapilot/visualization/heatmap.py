from __future__ import annotations

from typing import Union
import math
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


def heatmap(df: Union[pd.DataFrame, pl.DataFrame]) -> None:
    """Plots a lower-triangle Pearson correlation heatmap for numeric features.

    Renders only the lower triangle to reduce visual clutter.  Cell values
    are annotated and colour-scaled from -1 (cool) to +1 (warm).
    """
    local_df, _ = ensure_polars(df)

    numeric_cols = [
        col for col, dtype in zip(local_df.columns, local_df.dtypes)
        if dtype.is_numeric()
    ]
    if len(numeric_cols) < 2:
        print("⚠️  Not enough numeric columns to generate a correlation heatmap.")
        return

    pdf = local_df.select(numeric_cols).to_pandas()

    # Drop columns that are all-NaN after conversion
    pdf = pdf.dropna(axis=1, how="all")
    if pdf.shape[1] < 2:
        print("⚠️  Not enough valid numeric columns after dropping all-null columns.")
        return

    corr  = pdf.corr()
    n     = len(corr)

    # Dynamically size the figure based on column count
    size  = max(7, n * 0.9)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    _apply_theme()

    # Lower-triangle mask — cleaner than full matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cbar_kws = {
        "shrink": 0.8,
        "label": "Pearson r",
        "orientation": "vertical",
    }

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor=_BG,
        annot_kws={"size": max(8, 11 - n // 3), "weight": "bold"},
        cbar_kws=cbar_kws,
        square=True,
    )

    # Style the colour-bar text
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(_TEXT)
    cbar.ax.tick_params(colors=_TEXT)

    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.patch.set_facecolor(_BG)
    plt.tight_layout()
    plt.show()


def scatter(
    df: Union[pd.DataFrame, pl.DataFrame],
    x: str,
    y: str,
    hue: str | None = None,
    trendline: bool = True,
) -> None:
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
    plt.show()


def violin(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    group_by: str | None = None,
) -> None:
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
    plt.show()
