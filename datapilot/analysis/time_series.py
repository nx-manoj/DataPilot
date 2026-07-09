import polars as pl
import pandas as pd
from typing import Union, Dict, Any, Optional
from ..utils.validation import ensure_polars

def time_series_profile(
    df: Union[pd.DataFrame, pl.DataFrame],
    time_col: str,
    value_col: Optional[str] = None
) -> Dict[str, Any]:
    """Profiles a time-series dataset.
    
    Automatically detects the primary interval (e.g. daily, monthly),
    flags missing dates/gaps in the sequence, and provides a statistical summary.
    
    Args:
        df: Input DataFrame
        time_col: Name of the column containing dates/timestamps
        value_col: (Optional) Name of a numeric value column to compute trends on
        
    Returns:
        Dict containing time-series metadata and gap analysis.
    """
    local_df, _ = ensure_polars(df)
    
    if time_col not in local_df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame.")
        
    # Attempt to cast to Datetime if it's not already a temporal type
    if not local_df[time_col].dtype.is_temporal():
        try:
            # Let Polars try to auto-parse standard string dates
            local_df = local_df.with_columns(pl.col(time_col).str.to_datetime(strict=False))
        except Exception:
            raise ValueError(f"Could not convert column '{time_col}' to Datetime.")

    # Drop null dates and sort
    clean_df = local_df.drop_nulls(subset=[time_col]).sort(time_col)
    
    if clean_df.height < 2:
        return {"error": "Not enough data points after dropping nulls."}

    # Calculate differences between consecutive dates
    diffs = clean_df.with_columns(
        diff=pl.col(time_col).diff()
    )["diff"].drop_nulls()
    
    # Infer frequency (mode of differences)
    modes = diffs.mode()
    if modes.is_empty():
        inferred_freq = None
    else:
        inferred_freq = modes[0]

    # Detect Gaps
    gap_count = 0
    if inferred_freq is not None:
        # Find rows where difference is strictly greater than inferred freq
        gap_mask = diffs > inferred_freq
        gap_count = gap_mask.sum()
        
    start_date = clean_df[time_col].min()
    end_date = clean_df[time_col].max()
    
    duration = end_date - start_date if start_date is not None and end_date is not None else None
    
    report = {
        "time_column": time_col,
        "start_date": start_date,
        "end_date": end_date,
        "duration": duration,
        "inferred_frequency": inferred_freq,
        "gap_count": gap_count,
        "total_rows": clean_df.height
    }
    
    # Trend analysis if value_col is provided
    if value_col and value_col in clean_df.columns:
        if clean_df[value_col].dtype.is_numeric():
            start_val = clean_df[value_col].drop_nulls().first()
            end_val = clean_df[value_col].drop_nulls().last()
            
            trend = "stable"
            if start_val is not None and end_val is not None:
                if end_val > start_val:
                    trend = "upward"
                elif end_val < start_val:
                    trend = "downward"
            
            report["value_column"] = value_col
            report["trend"] = trend
            report["start_value"] = start_val
            report["end_value"] = end_val
            
    # Print nice console output
    print("=" * 58)
    print("         DATAPILOT TIME-SERIES PROFILER           ")
    print("=" * 58)
    print(f"  Time Column : {time_col}")
    print(f"  Start Date  : {start_date}")
    print(f"  End Date    : {end_date}")
    print(f"  Duration    : {duration}")
    print(f"  Frequency   : {inferred_freq}")
    print(f"  Missing Gaps: {gap_count} gap(s) detected")
    if "trend" in report:
        print(f"  Overall Trend: {report['trend'].upper()} ({report['start_value']} -> {report['end_value']})")
    print("=" * 58)

    return report
