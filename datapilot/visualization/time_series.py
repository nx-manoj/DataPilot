import pandas as pd
import polars as pl
from typing import Union, Optional
from ..utils.validation import ensure_polars

def plot_time_series(
    df: Union[pd.DataFrame, pl.DataFrame],
    time_col: str,
    value_col: str,
    title: Optional[str] = None,
    color: str = "#3b82f6"
):
    """Generates an interactive Plotly time-series line chart.
    
    Automatically handles dark-mode styling and sorts the data chronologically.
    
    Args:
        df: Input DataFrame
        time_col: Name of the datetime column
        value_col: Name of the numeric value column
        title: Optional chart title
        color: Line color (default blue)
        
    Returns:
        A Plotly Figure object
    """
    import plotly.express as px

    # Ensure Polars and prepare data
    local_df, _ = ensure_polars(df)
    
    if time_col not in local_df.columns:
        raise ValueError(f"Time column '{time_col}' not found.")
    if value_col not in local_df.columns:
        raise ValueError(f"Value column '{value_col}' not found.")

    # Sort chronologically and drop nulls in time column
    plot_df = local_df.drop_nulls(subset=[time_col]).sort(time_col)
    
    # Cast to pandas for plotly
    pdf = plot_df.to_pandas()
    
    if title is None:
        title = f"{value_col} over Time"
        
    fig = px.line(
        pdf,
        x=time_col,
        y=value_col,
        title=title,
        template="plotly_dark",
    )
    
    fig.update_traces(line_color=color, line_width=2)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        hovermode="x unified"
    )
    
    return fig
