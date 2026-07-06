from ..utils.validation import ensure_polars
import polars as pl
from typing import Union, Dict, Any
import pandas as pd
import math

def correlation(df: Union[pd.DataFrame, pl.DataFrame], threshold: float = 0.6) -> Dict[str, Any]:
    """Calculates a high-speed Pearson correlation matrix and flags strong pairs."""
    local_df, original_engine = ensure_polars(df)

    # Filter down to only numeric types using Polars type properties
    numeric_cols = [col for col, dtype in zip(local_df.columns, local_df.dtypes) if dtype.is_numeric()]

    if len(numeric_cols) < 2:
        return {"matrix": None, "strong_positive": [], "strong_negative": []}

    num_df = local_df.select(numeric_cols)
    corr_matrix = num_df.corr()

    strong_pos = []
    strong_neg = []
    cols = corr_matrix.columns

    # Extract distinct variable pairs matching our relationship thresholds
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix[i, j]
            # Guard against both Python None and IEEE-754 NaN — the latter
            # is what Polars returns for degenerate pairs (e.g. constant cols).
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                pair = f"{cols[i]} ↔ {cols[j]}"
                if val >= threshold:
                    strong_pos.append((pair, round(val, 3)))
                elif val <= -threshold:
                    strong_neg.append((pair, round(val, 3)))

    # If the user passed Pandas, convert the matrix back to Pandas for consistency
    final_matrix = corr_matrix.to_pandas() if original_engine == "pandas" else corr_matrix

    return {
        "matrix": final_matrix,
        "strong_positive": strong_pos,
        "strong_negative": strong_neg
    }
