from ..utils.validation import ensure_polars
from ..config import get_config
import polars as pl
import pandas as pd
from typing import Union, Tuple, Dict, Any, List, Optional


def auto_clean(
    df: Union[pd.DataFrame, pl.DataFrame],
    drop_null_threshold: float = 0.6,
    impute_strategy: str = "auto",
    encode_categoricals: Optional[str] = None,
    scale_numerics: Optional[str] = None,
    drop_id_columns: bool = True,
    drop_constant_columns: bool = True,
    use_ai: bool = False,
    ai_provider: Optional[str] = None,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], List[Dict[str, Any]]]:
    """Automatically cleans a dataset by applying common preprocessing steps.

    Performs the following operations (in order):
    1. Drops constant columns (zero variance)
    2. Drops ID-like columns (100% unique values)
    3. Drops columns exceeding the null threshold
    4. Imputes remaining numeric nulls (median or KNN)
    5. Imputes remaining categorical/string nulls with mode
    6. Encodes categoricals (onehot or label)
    7. Scales numerics (standard or minmax)

    Args:
        df: Input DataFrame (Pandas or Polars).
        drop_null_threshold: Columns with null % above this are dropped (default: 0.60 = 60%).
        impute_strategy: 'auto', 'knn', 'median', or 'mode'.
        encode_categoricals: 'onehot', 'label', or None.
        scale_numerics: 'standard', 'minmax', or None.
        drop_id_columns: Whether to drop columns where all values are unique (default: True).
        drop_constant_columns: Whether to drop columns with a single unique value (default: True).

    Returns:
        Tuple of (cleaned_df, change_log) where change_log is a list of dicts
        describing every action taken. The returned DataFrame matches the input type.
    """
    local_df, original_engine = ensure_polars(df)
    total_rows = local_df.height
    change_log: List[Dict[str, Any]] = []
    cols_to_drop: List[str] = []

    print("=" * 58)
    print("         DATAPILOT AUTO-CLEAN ENGINE              ")
    print("=" * 58)
    print(f"  Input:  {total_rows:,} rows × {local_df.width} columns")

    null_counts = local_df.null_count()

    for col in local_df.columns:
        series  = local_df[col]
        n_unique = series.n_unique()
        null_ct  = null_counts[col][0]
        null_pct = null_ct / total_rows if total_rows > 0 else 0

        # ── 1. Constant columns ──────────────────────────────────────────────
        if drop_constant_columns and n_unique == 1:
            cols_to_drop.append(col)
            change_log.append({
                "action": "DROPPED",
                "column": col,
                "reason": "Constant column — all values are identical (zero variance)",
            })
            print(f"\n  🗑️  DROP   '{col}'  →  constant column (1 unique value)")
            continue

        # ── 2. ID-like columns ───────────────────────────────────────────────
        if drop_id_columns and n_unique == total_rows and total_rows > 10:
            cols_to_drop.append(col)
            change_log.append({
                "action": "DROPPED",
                "column": col,
                "reason": "ID/key column — 100% unique values carry no predictive signal",
            })
            print(f"\n  🗑️  DROP   '{col}'  →  ID-like column (all {n_unique} values unique)")
            continue

        # ── 3. High-null columns ──────────────────────────────────────────────
        if null_pct >= drop_null_threshold:
            cols_to_drop.append(col)
            change_log.append({
                "action": "DROPPED",
                "column": col,
                "reason": f"High null rate ({null_pct*100:.1f}%) exceeds threshold ({drop_null_threshold*100:.0f}%)",
            })
            print(f"\n  🗑️  DROP   '{col}'  →  {null_pct*100:.1f}% nulls (above {drop_null_threshold*100:.0f}% threshold)")

    # Apply drops
    clean_df = local_df.drop(cols_to_drop) if cols_to_drop else local_df.clone()

    # ── 4 & 5. Impute remaining nulls ─────────────────────────────────────────
    # If strategy is KNN, check if scikit-learn is available
    if impute_strategy == "knn":
        try:
            from sklearn.impute import KNNImputer
            import numpy as np
            
            num_cols = [c for c in clean_df.columns if clean_df[c].dtype.is_numeric()]
            if num_cols:
                # Find columns that actually need imputation
                null_cols = [c for c in num_cols if clean_df[c].null_count() > 0]
                if null_cols:
                    imputer = KNNImputer(n_neighbors=5)
                    X = clean_df.select(num_cols).to_numpy()
                    X_imputed = imputer.fit_transform(X)
                    
                    imputed_df = pl.DataFrame(X_imputed, schema=num_cols)
                    clean_df = clean_df.with_columns([pl.Series(c, imputed_df[c]) for c in num_cols])
                    
                    for col in null_cols:
                        null_ct = null_counts[col][0]
                        change_log.append({
                            "action": "IMPUTED",
                            "column": col,
                            "reason": f"Filled {null_ct} null(s) using KNN Imputer (k=5)",
                        })
                        print(f"\n  🔧  IMPUTE '{col}'  →  filled {null_ct} null(s) with KNN")
        except ImportError:
            print("\n  ⚠️  scikit-learn is required for KNN imputation. Falling back to median.")
            impute_strategy = "median"

    impute_exprs = []
    for col in clean_df.columns:
        dtype    = clean_df[col].dtype
        null_ct  = clean_df[col].null_count()
        if null_ct == 0:
            continue

        if dtype.is_numeric():
            if impute_strategy in ("auto", "median"):
                median_val = clean_df[col].median()
                if median_val is not None:
                    impute_exprs.append(pl.col(col).fill_null(median_val))
                    change_log.append({
                        "action": "IMPUTED",
                        "column": col,
                        "reason": f"Filled {null_ct} null(s) with median ({round(median_val, 4)})",
                    })
                    print(f"\n  🔧  IMPUTE '{col}'  →  filled {null_ct} null(s) with median={round(median_val, 4)}")
        else:
            if impute_strategy in ("auto", "mode", "knn"): # knn doesn't do strings, so we still mode impute strings
                modes = clean_df[col].drop_nulls().mode()
                if modes.len() > 0:
                    mode_val = str(modes[0])
                    impute_exprs.append(pl.col(col).fill_null(mode_val))
                    change_log.append({
                        "action": "IMPUTED",
                        "column": col,
                        "reason": f"Filled {null_ct} null(s) with mode ('{mode_val}')",
                    })
                    print(f"\n  🔧  IMPUTE '{col}'  →  filled {null_ct} null(s) with mode='{mode_val}'")

    if impute_exprs:
        clean_df = clean_df.with_columns(impute_exprs)

    # ── 6. Encode Categoricals ────────────────────────────────────────────────
    if encode_categoricals:
        cat_cols = [c for c in clean_df.columns if clean_df[c].dtype in (pl.Utf8, pl.Categorical, pl.Enum)]
        if cat_cols:
            if encode_categoricals == "onehot":
                clean_df = clean_df.to_dummies(columns=cat_cols)
                change_log.append({
                    "action": "ENCODED",
                    "column": "Multiple",
                    "reason": f"One-hot encoded {len(cat_cols)} categorical columns.",
                })
                print(f"\n  🔢  ENCODE → One-hot encoded {len(cat_cols)} column(s)")
            elif encode_categoricals == "label":
                # Convert string to categorical and then to integer (physical)
                exprs = [pl.col(c).cast(pl.Categorical).to_physical().alias(c) for c in cat_cols]
                clean_df = clean_df.with_columns(exprs)
                change_log.append({
                    "action": "ENCODED",
                    "column": "Multiple",
                    "reason": f"Label encoded {len(cat_cols)} categorical columns.",
                })
                print(f"\n  🔢  ENCODE → Label encoded {len(cat_cols)} column(s)")

    # ── 7. Scale Numerics ─────────────────────────────────────────────────────
    if scale_numerics:
        num_cols = [c for c in clean_df.columns if clean_df[c].dtype.is_numeric()]
        if num_cols:
            if scale_numerics == "standard":
                # Handle standard deviation being 0 by keeping the column or filling with 0
                exprs = [
                    pl.when(pl.col(c).std() == 0).then(0.0)
                    .otherwise((pl.col(c) - pl.col(c).mean()) / pl.col(c).std())
                    .alias(c) for c in num_cols
                ]
                clean_df = clean_df.with_columns(exprs)
                change_log.append({
                    "action": "SCALED",
                    "column": "Multiple",
                    "reason": f"Standard scaled (Z-score) {len(num_cols)} numeric columns.",
                })
                print(f"\n  📏  SCALE  → Standard scaled {len(num_cols)} numeric column(s)")
            elif scale_numerics == "minmax":
                exprs = [
                    pl.when(pl.col(c).max() == pl.col(c).min()).then(0.0)
                    .otherwise((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min()))
                    .alias(c) for c in num_cols
                ]
                clean_df = clean_df.with_columns(exprs)
                change_log.append({
                    "action": "SCALED",
                    "column": "Multiple",
                    "reason": f"Min-Max scaled {len(num_cols)} numeric columns.",
                })
                print(f"\n  📏  SCALE  → Min-Max scaled {len(num_cols)} numeric column(s)")

    print("\n" + "-" * 58)
    print(f"  Output: {clean_df.height:,} rows × {clean_df.width} columns")
    dropped = local_df.width - clean_df.width
    imputed = sum(1 for c in change_log if c["action"] == "IMPUTED")
    print(f"  ✅ {dropped} column(s) dropped  |  {imputed} column(s) imputed")
    print("=" * 58)

    # Return in original format
    result_df = clean_df.to_pandas() if original_engine == "pandas" else clean_df

    # ── Optional AI Commentary ──────────────────────────────────────────────
    if use_ai and change_log:
        cfg           = get_config()
        provider_name = ai_provider or cfg["ai_provider"]
        model_name    = ai_model    or cfg["ai_model"]
        key           = api_key     or cfg["api_key"]

        log_text = "\n".join(
            f"- [{e['action']}] {e['column']}: {e['reason']}"
            for e in change_log
        )
        system_prompt = (
            "You are DataPilot, a data cleaning expert. "
            "Given a cleaning change log, explain in 3-4 bullet points "
            "why these changes improve ML model quality and what the user "
            "should do next (feature engineering, encoding, splitting). "
            "Be concise and direct."
        )
        user_prompt = (
            f"Input: {total_rows:,} rows × {local_df.width} cols → "
            f"Output: {clean_df.height:,} rows × {clean_df.width} cols\n"
            f"Change log:\n{log_text}"
        )
        try:
            from ..ai.factory import get_provider
            provider = get_provider(
                ai_provider=provider_name,
                ai_model=model_name,
                api_key=key,
            )
            ai_response = provider._call_with_raw_prompts(system_prompt, user_prompt)
            print(f"\n🤖 AI Summary  [{provider_name.upper()}]:")
            print(ai_response)
        except Exception as e:
            import logging
            logging.error(f"AI error: {e}", exc_info=True)
            print(f"\n⚠️  AI error: {e}")

    return result_df, change_log
