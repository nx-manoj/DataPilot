from __future__ import annotations

from typing import Union, Optional
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..utils.validation import ensure_polars

def hist(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    bins: Union[int, str] = "auto",
    hue: Optional[str] = None,
    color: str = "#3b82f6",
):
    """Plots a styled distribution histogram using Plotly.

    Args:
        df:     Input DataFrame (Pandas or Polars).
        column: Target numeric column name.
        bins:   Number of bins.
        hue:    Optional grouping column for colour-split distributions.
        color:  Bar fill colour when hue is not set.
    
    Returns:
        Interactive Plotly Figure.
    """
    local_df, _ = ensure_polars(df)

    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    pdf = local_df.to_pandas()
    series = pdf[column].dropna()

    if hue:
        fig = px.histogram(
            pdf, x=column, color=hue, nbins=bins if isinstance(bins, int) else None,
            marginal="violin", opacity=0.75, barmode="overlay",
            title=f"Distribution of {column} by {hue}"
        )
    else:
        fig = px.histogram(
            pdf, x=column, nbins=bins if isinstance(bins, int) else None,
            marginal="violin", opacity=0.8, color_discrete_sequence=[color],
            title=f"Distribution of {column}"
        )
        # Overlay mean and median lines
        mean_val = series.mean()
        median_val = series.median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#f43f5e", annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dot", line_color="#10b981", annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")

    subtitle = (
        f"n={len(series):,} | mean={series.mean():.3f} | "
        f"std={series.std():.3f} | skew={series.skew():.2f}"
    )

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        xaxis_title=f"{column}<br><sup>{subtitle}</sup>",
        yaxis_title="Count",
        margin=dict(t=50, l=20, r=20, b=50)
    )

    return fig
