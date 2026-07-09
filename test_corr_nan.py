import polars as pl
from datapilot.analysis.correlations import correlation
import math

df = pl.DataFrame({
    "a": [1.0, 2.0, 3.0, 4.0, None],
    "b": [2.0, 4.0, 6.0, None, 10.0],
    "c": [None, 4.0, 3.0, 2.0, 1.0]
})

print("Testing custom correlation function")
res = correlation(df)
print(res)

print("Testing polars built-in corr")
print(df.corr())
