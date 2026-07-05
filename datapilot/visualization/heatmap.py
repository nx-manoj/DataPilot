from ..utils.validation import ensure_polars
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
import pandas as pd

def heatmap(df: Union[pd.DataFrame, pl.DataFrame]) -> None:
    """Generates a visual correlation matrix heatmap for all numerical features."""
    local_df, _ = ensure_polars(df)
    
    # Isolate only numerical fields
    numeric_cols = [col for col, dtype in zip(local_df.columns, local_df.dtypes) if dtype.is_numeric()]
    if len(numeric_cols) < 2:
        print("⚠️ Not enough numerical columns to generate a correlation heatmap.")
        return
        
    # Compute using Polars' high-speed engine
    corr_matrix = local_df.select(numeric_cols).corr().to_pandas()
    corr_matrix.index = numeric_cols  # Re-apply structural index labels for seaborn
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Dataset Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
