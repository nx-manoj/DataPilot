from ..utils.validation import ensure_polars
import polars as pl
import pandas as pd
from typing import Union, List, Dict, Any


def suggest(df: Union[pd.DataFrame, pl.DataFrame]) -> List[Dict[str, Any]]:
    """Analyses dataset structure and returns a list of actionable preprocessing suggestions.

    Detects common data quality patterns such as potential ID columns, high-null
    columns, unenconded categoricals, date strings, constant columns, and
    imbalanced binary columns — without requiring any AI or internet access.

    Args:
        df: Input DataFrame (Pandas or Polars).

    Returns:
        List of dicts, each containing 'column', 'issue', 'severity', and 'suggestion'.
        Also prints a formatted report to stdout.
    """
    local_df, _ = ensure_polars(df)
    total_rows = local_df.height
    suggestions: List[Dict[str, Any]] = []

    null_counts = local_df.null_count()

    for col in local_df.columns:
        dtype = local_df[col].dtype
        series = local_df[col]
        null_count = null_counts[col][0]
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        n_unique = series.n_unique()

        # ── High-null columns ─────────────────────────────────────────────────
        if null_pct >= 60:
            suggestions.append({
                "column": col,
                "issue": f"Very high missing rate ({null_pct:.1f}%)",
                "severity": "🔴 HIGH",
                "suggestion": f"Consider dropping '{col}' — over 60% nulls will degrade model quality.",
            })
        elif null_pct >= 20:
            suggestions.append({
                "column": col,
                "issue": f"High missing rate ({null_pct:.1f}%)",
                "severity": "🟡 MEDIUM",
                "suggestion": f"Impute '{col}' with median (numeric) or mode (categorical) before training.",
            })

        # ── Potential ID / key columns ─────────────────────────────────────────
        if n_unique == total_rows and total_rows > 10:
            suggestions.append({
                "column": col,
                "issue": "100% unique values — likely an ID or key column",
                "severity": "🔴 HIGH",
                "suggestion": f"Drop '{col}' — unique identifiers leak no signal and inflate dimensionality.",
            })

        # ── Constant columns ──────────────────────────────────────────────────
        if n_unique == 1:
            suggestions.append({
                "column": col,
                "issue": "Constant column — all values are identical",
                "severity": "🔴 HIGH",
                "suggestion": f"Drop '{col}' — zero variance columns carry no predictive information.",
            })

        # ── String columns that should be encoded ─────────────────────────────
        if dtype == pl.Utf8 or dtype == pl.String:
            if 2 <= n_unique <= 20:
                suggestions.append({
                    "column": col,
                    "issue": f"Unencoded categorical string ({n_unique} unique values)",
                    "severity": "🟡 MEDIUM",
                    "suggestion": f"Encode '{col}' using One-Hot Encoding (low cardinality) before modelling.",
                })
            elif n_unique > 20:
                suggestions.append({
                    "column": col,
                    "issue": f"High-cardinality string column ({n_unique} unique values)",
                    "severity": "🟡 MEDIUM",
                    "suggestion": f"Use Target Encoding or embeddings for '{col}' — One-Hot will explode dimensionality.",
                })

        # ── Date-like string detection ─────────────────────────────────────────
        if dtype == pl.Utf8 or dtype == pl.String:
            sample = series.drop_nulls().head(5).to_list()
            date_hints = ["2020", "2021", "2022", "2023", "2024", "2025", "2026",
                          "-01-", "-02-", "/01/", "jan", "feb", "mar"]
            if any(any(hint in str(s).lower() for hint in date_hints) for s in sample):
                suggestions.append({
                    "column": col,
                    "issue": "Looks like a date stored as string",
                    "severity": "🟡 MEDIUM",
                    "suggestion": f"Parse '{col}' to datetime and extract features: year, month, day_of_week.",
                })

        # ── Imbalanced binary columns ──────────────────────────────────────────
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16,
                     pl.UInt32, pl.UInt64) and n_unique == 2:
            counts = series.drop_nulls().value_counts()
            if counts.height == 2:
                vals = counts["count"].to_list()
                ratio = min(vals) / max(vals) if max(vals) > 0 else 1
                if ratio < 0.2:
                    suggestions.append({
                        "column": col,
                        "issue": f"Highly imbalanced binary column (minority class: {ratio*100:.1f}%)",
                        "severity": "🟡 MEDIUM",
                        "suggestion": f"'{col}' may be a target with class imbalance — consider SMOTE, class_weight, or stratified splits.",
                    })

    # ── Print formatted report ─────────────────────────────────────────────────
    print("=" * 56)
    print("       DATAPILOT SMART COLUMN SUGGESTIONS        ")
    print("=" * 56)
    if not suggestions:
        print("✅ No issues detected — dataset looks clean and ready!")
    else:
        for i, s in enumerate(suggestions, 1):
            print(f"\n[{i}] {s['severity']}  —  {s['column']}")
            print(f"     Issue:      {s['issue']}")
            print(f"     Fix:        {s['suggestion']}")
    print("=" * 56)

    return suggestions
