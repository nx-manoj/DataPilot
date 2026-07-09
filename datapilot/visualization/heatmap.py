from __future__ import annotations

from typing import Union
import polars as pl
import pandas as pd
import plotly.express as px

from ..utils.validation import ensure_polars

def heatmap(df: Union[pd.DataFrame, pl.DataFrame]):
    """Plots a Pearson correlation heatmap for numeric features using Plotly.

    Returns an interactive Plotly Figure.
    """
    local_df, _ = ensure_polars(df)

    numeric_cols = [
        col for col, dtype in zip(local_df.columns, local_df.dtypes)
        if dtype.is_numeric()
    ]
    if len(numeric_cols) < 2:
        print("⚠️  Not enough numeric columns to generate a correlation heatmap.")
        return None

    pdf = local_df.select(numeric_cols).to_pandas()
    # Drop columns that are all-NaN after conversion
    pdf = pdf.dropna(axis=1, how="all")
    if pdf.shape[1] < 2:
        print("⚠️  Not enough valid numeric columns after dropping all-null columns.")
        return None

    corr  = pdf.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        template="plotly_dark",
        title_x=0.5,
        margin=dict(t=50, l=20, r=20, b=20)
    )

    return fig
