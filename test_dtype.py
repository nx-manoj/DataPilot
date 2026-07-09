import polars as pl
df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
print([dtype.is_numeric() for dtype in df.dtypes])
