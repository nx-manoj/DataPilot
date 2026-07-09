from __future__ import annotations

from typing import Union
import numpy as np
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..utils.validation import ensure_polars

def scatter(
    df: Union[pd.DataFrame, pl.DataFrame],
    x: str,
    y: str,
    hue: str | None = None,
    trendline: bool = True,
):
    """Scatter plot with an optional OLS regression trendline using Plotly.

    Args:
        df:        Input DataFrame (Pandas or Polars).
        x:         Column name for the horizontal axis.
        y:         Column name for the vertical axis.
        hue:       Optional column for colour grouping.
        trendline: Draw an OLS regression line (default: True).
        
    Returns:
        Interactive Plotly Figure.
    """
    local_df, _ = ensure_polars(df)
    for col in [x, y] + ([hue] if hue else []):
        if col not in local_df.columns:
            raise ValueError(f"Column '{col}' not found.")

    pdf = local_df.to_pandas()

    fig = px.scatter(
        pdf, x=x, y=y, color=hue,
        opacity=0.75,
        title=f"{y} vs {x}"
    )
    
    if trendline and not hue:
        _x = pdf[x].dropna()
        _y = pdf[y].dropna()
        common = _x.index.intersection(_y.index)
        if len(common) > 2:
            z = np.polyfit(_x[common], _y[common], 1)
            p = np.poly1d(z)
            xrng = np.linspace(_x.min(), _x.max(), 200)
            
            fig.add_trace(go.Scatter(
                x=xrng, y=p(xrng),
                mode='lines',
                line=dict(color="#f43f5e", width=2, dash='dash'),
                name="OLS trend"
            ))

    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        margin=dict(t=50, l=20, r=20, b=20)
    )

    return fig
