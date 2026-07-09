import polars as pl
from datapilot.analysis.correlations import correlation

df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 6, 8, 10],
    "z": [5, 4, 3, 2, 1]
})

res = correlation(df)
print(res)
