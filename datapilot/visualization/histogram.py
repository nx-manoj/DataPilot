from ..utils.validation import ensure_polars
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import pandas as pd

def hist(df: Union[pd.DataFrame, pl.DataFrame], column: str, bins: int = 10) -> None:
    """Generates a clean distribution histogram for a selected numerical feature."""
    local_df, _ = ensure_polars(df)
    
    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
        
    # Extract data and drop null values so matplotlib doesn't complain
    plot_data = local_df[column].drop_nulls().to_numpy()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(plot_data, bins=bins, kde=True, color="#4A90E2")
    plt.title(f"Distribution of {column}", fontsize=14, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
