from ..utils.validation import ensure_polars
import polars as pl
from typing import Union, Dict, Any
import pandas as pd

def duplicates(df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
    """Calculates duplicate row counts and percentages using multi-threaded hashing."""
    local_df, _ = ensure_polars(df)
    total_rows = local_df.height
    
    if total_rows == 0:
        return {"duplicate_count": 0, "duplicate_percentage": 0.0}
        
    # Polars checks structural row uniqueness instantly across CPU cores
    unique_rows = local_df.unique().height
    dup_count = total_rows - unique_rows
    dup_pct = round((dup_count / total_rows) * 100, 2)
    
    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": dup_pct
    }
