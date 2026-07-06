from ..utils.validation import ensure_polars
import polars as pl
from typing import Union, Dict, Any, List, Tuple
import pandas as pd
import math


def _pearson_pairwise(series_a: pl.Series, series_b: pl.Series) -> float:
    """Compute Pearson r for two Polars Series, dropping rows where either is null.

    Polars' global .corr() propagates NaN for any column that has missing values,
    making the whole row/column unusable.  This helper does a clean pairwise
    drop-nulls so every valid pair gets a real correlation value.
    """
    # Stack into a 2-col frame and drop any row that has a null in either column
    combined = pl.DataFrame({"a": series_a, "b": series_b}).drop_nulls()
    if combined.height < 2:
        return float("nan")

    a = combined["a"].cast(pl.Float64)
    b = combined["b"].cast(pl.Float64)

    n = combined.height
    mean_a = a.mean()
    mean_b = b.mean()

    cov   = ((a - mean_a) * (b - mean_b)).sum() / (n - 1)
    std_a = a.std()
    std_b = b.std()

    if std_a == 0 or std_b == 0:
        return float("nan")

    return float(cov / (std_a * std_b))


def correlation(
    df: Union[pd.DataFrame, pl.DataFrame],
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """Calculates a pairwise Pearson correlation matrix and flags strong pairs.

    Uses a pairwise drop-nulls strategy so columns with missing values (e.g. Age)
    still yield valid correlation values against all other numeric columns, instead
    of propagating NaN across the whole row/column.
    """
    local_df, original_engine = ensure_polars(df)

    # Filter down to only numeric columns
    numeric_cols = [
        col for col, dtype in zip(local_df.columns, local_df.dtypes)
        if dtype.is_numeric()
    ]

    if len(numeric_cols) < 2:
        return {"matrix": None, "strong_positive": [], "strong_negative": []}

    n_cols = len(numeric_cols)
    strong_pos: List[Tuple[str, float]] = []
    strong_neg: List[Tuple[str, float]] = []

    # Build a symmetric n×n matrix of pairwise correlations
    matrix_data: List[List[float]] = []
    for i in range(n_cols):
        row = []
        for j in range(n_cols):
            if i == j:
                row.append(1.0)
            elif j < i:
                # Already computed — mirror from upper triangle
                row.append(matrix_data[j][i])
            else:
                r = _pearson_pairwise(
                    local_df[numeric_cols[i]],
                    local_df[numeric_cols[j]],
                )
                row.append(r)
                # Flag strong pairs (upper triangle only to avoid duplicates)
                if not math.isnan(r):
                    pair = f"{numeric_cols[i]} ↔ {numeric_cols[j]}"
                    if r >= threshold:
                        strong_pos.append((pair, round(r, 3)))
                    elif r <= -threshold:
                        strong_neg.append((pair, round(r, 3)))
        matrix_data.append(row)

    # Build a Polars DataFrame for the correlation matrix
    corr_polars = pl.DataFrame(
        {numeric_cols[i]: [matrix_data[i][j] for j in range(n_cols)]
         for i in range(n_cols)}
    )

    # Return in the same engine the user passed in
    final_matrix = corr_polars.to_pandas() if original_engine == "pandas" else corr_polars

    return {
        "matrix": final_matrix,
        "strong_positive": strong_pos,
        "strong_negative": strong_neg,
    }
