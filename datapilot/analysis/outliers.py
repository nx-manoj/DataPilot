from ..utils.validation import ensure_polars
from ..config import get_config
import polars as pl
import pandas as pd
from typing import Union, Dict, Any, List, Optional


def outliers(
    df: Union[pd.DataFrame, pl.DataFrame],
    method: str = "both",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    use_ai: bool = False,
    ai_provider: Optional[str] = None,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Detects outliers in all numeric columns using IQR and/or Z-score methods.

    Args:
        df: Input DataFrame (Pandas or Polars).
        method: Detection strategy — 'iqr', 'zscore', or 'both' (default).
                'both' flags values that are outliers in either method.
        z_threshold: Number of standard deviations for Z-score method (default: 3.0).
        iqr_multiplier: IQR fence multiplier (default: 1.5; use 3.0 for extreme outliers only).

    Returns:
        Dict mapping column name → dict with 'count', 'percentage', 'values', 'severity'.
        Also prints a formatted report to stdout.
    """
    local_df, original_engine = ensure_polars(df)
    numeric_cols = [
        col for col, dtype in zip(local_df.columns, local_df.dtypes)
        if dtype.is_numeric()
    ]

    if not numeric_cols:
        print("⚠️  No numeric columns found for outlier detection.")
        return {}

    result: Dict[str, Any] = {}

    print("=" * 56)
    print("        DATAPILOT OUTLIER DETECTION REPORT        ")
    print(f"  Method: {method.upper()} | Z-threshold: {z_threshold} | IQR×{iqr_multiplier}")
    print("=" * 56)

    for col in numeric_cols:
        series = local_df[col].drop_nulls()
        if series.len() < 4:
            continue

        values = series.to_list()
        n = len(values)
        mean = series.mean()
        std  = series.std() or 1e-9
        q1   = series.quantile(0.25)
        q3   = series.quantile(0.75)
        iqr  = q3 - q1
        lower_iqr = q1 - iqr_multiplier * iqr
        upper_iqr = q3 + iqr_multiplier * iqr

        outlier_set = set()

        if method in ("iqr", "both"):
            for v in values:
                if v < lower_iqr or v > upper_iqr:
                    outlier_set.add(v)

        if method in ("zscore", "both"):
            for v in values:
                if abs((v - mean) / std) > z_threshold:
                    outlier_set.add(v)

        if outlier_set:
            count = len(outlier_set)
            pct   = round(count / n * 100, 2)
            severity = (
                "🔴 HIGH"   if pct > 5 else
                "🟡 MEDIUM" if pct > 1 else
                "🟢 LOW"
            )
            top_vals = sorted(outlier_set, key=abs, reverse=True)[:5]

            result[col] = {
                "count": count,
                "percentage": pct,
                "severity": severity,
                "values": top_vals,
                "lower_fence": round(lower_iqr, 4),
                "upper_fence": round(upper_iqr, 4),
            }

            print(f"\n  {severity}  —  {col}")
            print(f"     Outlier count:  {count} ({pct}% of non-null values)")
            print(f"     IQR fences:     [{round(lower_iqr, 2)}, {round(upper_iqr, 2)}]")
            print(f"     Extreme values: {top_vals}")

    print("\n" + "-" * 56)
    if not result:
        print("  ✅ No outliers detected across numeric columns.")
    else:
        print(f"  🔍 {len(result)} column(s) contain outliers.")
        print("  💡 Tip: Cap outliers with IQR clipping or use robust scalers.")
    print("=" * 56)

    # ── Optional AI Commentary ───────────────────────────────────────────────
    if use_ai and result:
        cfg           = get_config()
        provider_name = ai_provider or cfg["ai_provider"]
        model_name    = ai_model    or cfg["ai_model"]
        key           = api_key     or cfg["api_key"]

        issues_text = "\n".join(
            f"- {col}: {info['count']} outliers ({info['percentage']}%) "
            f"| severity={info['severity']} | extremes={info['values'][:3]}"
            for col, info in result.items()
        )
        system_prompt = (
            "You are DataPilot, an expert data scientist. "
            "Given a list of numeric columns with outlier statistics, "
            "provide 3-4 concise treatment recommendations (capping, log-transform, "
            "robust scalers, domain-specific decisions). Be direct."
        )
        user_prompt = (
            f"Method used: {method.upper()} | z-threshold: {z_threshold} | IQR×{iqr_multiplier}\n"
            f"Outlier report:\n{issues_text}"
        )
        try:
            from ..ai.factory import get_provider
            provider = get_provider(
                ai_provider=provider_name,
                ai_model=model_name,
                api_key=key,
            )
            ai_response = provider._call_with_raw_prompts(system_prompt, user_prompt)
            print(f"\n\ud83e\udd16 AI Treatment Recommendations  [{provider_name.upper()}]:")
            print(ai_response)
        except Exception as e:
            import logging
            logging.error(f"AI error: {e}", exc_info=True)
            print(f"\n⚠️  AI error: {e}")

    return result
