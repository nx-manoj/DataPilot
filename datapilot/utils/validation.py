import polars as pl
import pandas as pd
from typing import Union, Tuple

def ensure_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> Tuple[pl.DataFrame, str]:
    """Silently normalizes input dataframes to Polars using zero-copy Arrow memory

    sharing. Tracks the original data type to prevent user friction.
    """
    if isinstance(df, pl.DataFrame):
        return df, "polars"
    elif isinstance(df, pd.DataFrame):
        # Zero-copy conversion from Pandas to Polars via PyArrow
        return pl.from_pandas(df), "pandas"
    else:
        raise TypeError(
            f"Unsupported data type: {type(df)}. "
            "DataPilot accepts either pandas.DataFrame or polars.DataFrame."
        )
