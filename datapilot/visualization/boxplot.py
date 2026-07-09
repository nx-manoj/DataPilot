from __future__ import annotations

from typing import Union, Optional
import numpy as np
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..utils.validation import ensure_polars

def box(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    group_by: Optional[str] = None,
    orient: str = "v",
):
    """Plots a styled box plot with quartile annotations and outlier highlights using Plotly.

    Args:
        df:       Input DataFrame (Pandas or Polars).
        column:   Target numeric column name.
        group_by: Optional categorical column to split boxes by group.
        orient:   Orientation — 'v' (vertical, default) or 'h' (horizontal).
        
    Returns:
        Interactive Plotly Figure.
    """
    local_df, _ = ensure_polars(df)

    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
    if group_by and group_by not in local_df.columns:
        raise ValueError(f"group_by column '{group_by}' not found in the dataset.")

    pdf = local_df.to_pandas()
    series = pdf[column].dropna()

    x_kw = group_by if orient == "v" else column
    y_kw = column   if orient == "v" else group_by

    if group_by:
        fig = px.box(
            pdf, x=x_kw, y=y_kw, color=group_by,
            orientation=orient,
            title=f"Box Plot of {column} grouped by {group_by}"
        )
    else:
        fig = px.box(
            pdf, x=x_kw if orient == "h" else None, y=y_kw if orient == "v" else None,
            orientation=orient, color_discrete_sequence=["#8b5cf6"],
            title=f"Box Plot of {column}"
        )
    
    # Calculate stats for subtitle
    q1, med, q3 = np.percentile(series, [25, 50, 75])
    iqr = q3 - q1
    subtitle = (
        f"Q1={q1:.2f} | Median={med:.2f} | Q3={q3:.2f} | "
        f"IQR={iqr:.2f} | n={len(series):,}"
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        xaxis_title=x_kw or "",
        yaxis_title=y_kw or column,
        margin=dict(t=50, l=20, r=20, b=50)
    )
    
    # Add subtitle using annotation
    fig.add_annotation(
        text=subtitle,
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=11, color="#94a3b8")
    )

    return fig
