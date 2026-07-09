from __future__ import annotations

from typing import Union
import polars as pl
import pandas as pd
import plotly.express as px

from ..utils.validation import ensure_polars

def violin(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    group_by: str | None = None,
):
    """Violin plot — combines a box plot with a KDE for richer distribution insight using Plotly.

    Args:
        df:       Input DataFrame (Pandas or Polars).
        column:   Target numeric column.
        group_by: Optional categorical column to split violins by group.
        
    Returns:
        Interactive Plotly Figure.
    """
    local_df, _ = ensure_polars(df)
    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found.")

    pdf = local_df.to_pandas()

    if group_by:
        fig = px.violin(
            pdf, x=group_by, y=column, color=group_by,
            box=True, points="all",
            title=f"Violin Plot of {column} by {group_by}"
        )
    else:
        fig = px.violin(
            pdf, y=column, box=True, points="all",
            color_discrete_sequence=["#3b82f6"],
            title=f"Violin Plot of {column}"
        )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        margin=dict(t=50, l=20, r=20, b=50),
        xaxis_title=group_by or "",
        yaxis_title=column
    )

    return fig
