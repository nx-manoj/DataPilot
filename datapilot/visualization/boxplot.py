from ..utils.validation import ensure_polars
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import pandas as pd

def box(df: Union[pd.DataFrame, pl.DataFrame], column: str) -> None:
    """Generates a clean box plot to showcase distribution quartiles and outliers."""
    local_df, _ = ensure_polars(df)
    
    if column not in local_df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")
        
    plot_data = local_df[column].drop_nulls().to_numpy()
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=plot_data, color="#50E3C2")
    plt.title(f"Box Plot of {column}", fontsize=14, fontweight='bold')
    plt.ylabel(column)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
