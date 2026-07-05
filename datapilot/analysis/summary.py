from ..utils.validation import ensure_polars
import polars as pl
from typing import Union, Dict, Any
import pandas as pd

def summary(df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
    """Calculates lightning fast structural overview metrics of the dataset."""
    # 1. Silently handle engine normalization
    local_df, original_engine = ensure_polars(df)
    
    # 2. Extract metrics via Polars engine
    total_missing = sum(local_df.null_count().row(0))
    memory_mb = local_df.estimated_size("mb")
    
    return {
        "engine_detected": original_engine,
        "rows": local_df.height,
        "columns": local_df.width,
        "datatypes": {col: str(dtype) for col, dtype in zip(local_df.columns, local_df.dtypes)},
        "memory_usage_mb": round(memory_mb, 4),
        "total_missing_values": total_missing
    }
