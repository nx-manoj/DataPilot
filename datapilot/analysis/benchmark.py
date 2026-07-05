from ..utils.validation import ensure_polars
import polars as pl
import pandas as pd
import time
from typing import Union, Dict, Any


def benchmark(df: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
    """Benchmarks DataPilot (Polars) operations against equivalent Pandas operations.

    Runs the same analytical operations using both engines and reports the
    elapsed time and speedup factor, so users can see the concrete performance
    advantage of the Polars-native architecture.

    Args:
        df: Input DataFrame (Pandas or Polars). The function will internally
            convert to both engines for a fair side-by-side comparison.

    Returns:
        Dict mapping operation name → {'datapilot_ms', 'pandas_ms', 'speedup'}.
        Also prints a formatted benchmark report to stdout.
    """
    # Ensure we have both engines
    local_polars, _ = ensure_polars(df)
    pandas_df = local_polars.to_pandas()

    results: Dict[str, Any] = {}
    numeric_cols = [
        col for col, dtype in zip(local_polars.columns, local_polars.dtypes)
        if dtype.is_numeric()
    ]

    benchmarks = []

    # ── 1. Null count ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(10):
        local_polars.null_count()
    polars_ms = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        pandas_df.isnull().sum()
    pandas_ms = (time.perf_counter() - t0) / 10 * 1000

    benchmarks.append(("Null Count", polars_ms, pandas_ms))

    # ── 2. Duplicate detection ────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(10):
        local_polars.is_duplicated().sum()
    polars_ms = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        pandas_df.duplicated().sum()
    pandas_ms = (time.perf_counter() - t0) / 10 * 1000

    benchmarks.append(("Duplicate Detection", polars_ms, pandas_ms))

    # ── 3. Descriptive statistics ─────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(10):
        local_polars.describe()
    polars_ms = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        pandas_df.describe()
    pandas_ms = (time.perf_counter() - t0) / 10 * 1000

    benchmarks.append(("Describe / Summary Stats", polars_ms, pandas_ms))

    # ── 4. Correlation matrix (numeric only) ──────────────────────────────────
    if len(numeric_cols) >= 2:
        t0 = time.perf_counter()
        for _ in range(10):
            local_polars.select(numeric_cols).corr()
        polars_ms = (time.perf_counter() - t0) / 10 * 1000

        t0 = time.perf_counter()
        for _ in range(10):
            pandas_df[numeric_cols].corr()
        pandas_ms = (time.perf_counter() - t0) / 10 * 1000

        benchmarks.append(("Correlation Matrix", polars_ms, pandas_ms))

    # ── 5. Group-by mean ──────────────────────────────────────────────────────
    string_cols = [
        col for col, dtype in zip(local_polars.columns, local_polars.dtypes)
        if dtype == pl.Utf8 or dtype == pl.String
    ]
    if string_cols and numeric_cols:
        g_col = string_cols[0]
        n_col = numeric_cols[0]

        t0 = time.perf_counter()
        for _ in range(10):
            local_polars.group_by(g_col).agg(pl.col(n_col).mean())
        polars_ms = (time.perf_counter() - t0) / 10 * 1000

        t0 = time.perf_counter()
        for _ in range(10):
            pandas_df.groupby(g_col)[n_col].mean()
        pandas_ms = (time.perf_counter() - t0) / 10 * 1000

        benchmarks.append(("Group-By Mean", polars_ms, pandas_ms))

    # ── Print report ──────────────────────────────────────────────────────────
    print("=" * 66)
    print("          DATAPILOT BENCHMARK  —  Polars vs Pandas          ")
    print(f"  Dataset: {local_polars.height:,} rows × {local_polars.width} columns")
    print("=" * 66)
    print(f"  {'Operation':<26} {'DataPilot':>12} {'Pandas':>10} {'Speedup':>10}")
    print("-" * 66)

    total_speedup = 0.0
    for name, dp_ms, pd_ms in benchmarks:
        speedup = pd_ms / dp_ms if dp_ms > 0 else float("inf")
        total_speedup += speedup
        bar = "▓" * min(int(speedup), 20)
        print(f"  {name:<26} {dp_ms:>9.2f}ms  {pd_ms:>7.2f}ms  {speedup:>7.1f}× {bar}")
        results[name] = {
            "datapilot_ms": round(dp_ms, 3),
            "pandas_ms":    round(pd_ms, 3),
            "speedup":      round(speedup, 2),
        }

    avg_speedup = total_speedup / len(benchmarks) if benchmarks else 1
    print("-" * 66)
    print(f"  {'Average Speedup':<26} {'':>12} {'':>10} {avg_speedup:>7.1f}×")
    print("=" * 66)
    print(f"\n  ⚡ DataPilot is on average {avg_speedup:.1f}× faster than Pandas on this dataset.")
    print(f"  📈 Speedup grows significantly on larger datasets (1M+ rows).")
    print("=" * 66)

    return results
