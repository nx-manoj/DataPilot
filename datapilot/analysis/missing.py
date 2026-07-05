from ..utils.validation import ensure_polars
import polars as pl
from typing import Union
import pandas as pd

def missing(df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
    """Generates a lightning-fast report detailing missing count and percentage."""
    local_df, original_engine = ensure_polars(df)
    total_rows = local_df.height
    
    if total_rows == 0:
        report = pl.DataFrame({"column": [], "missing_count": [], "missing_percentage": []})
        return report.to_pandas() if original_engine == "pandas" else report

    null_counts = local_df.null_count()
    report_data = []
    for col in local_df.columns:
        count = null_counts[col][0]
        if count > 0:
            percentage = round((count / total_rows) * 100, 2)
            report_data.append({
                "column": col, 
                "missing_count": count, 
                "missing_percentage": percentage
            })
            
    if not report_data:
        report = pl.DataFrame({"column": [], "missing_count": [], "missing_percentage": []})
    else:
        report = pl.DataFrame(report_data).sort("missing_percentage", descending=True)
        
    return report.to_pandas() if original_engine == "pandas" else report
