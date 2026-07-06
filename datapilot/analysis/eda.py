from .summary import summary
from .missing import missing
from .duplicates import duplicates
from .correlations import correlation
from ..utils.validation import ensure_polars
from ..ai.insights import generate_insights
from ..config import get_config

import polars as pl
from typing import Union, Optional
import pandas as pd


def analyze(
    df: Union[pd.DataFrame, pl.DataFrame],
    use_ai: bool = False,
    ai_provider: Optional[str] = None,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Runs automated exploratory data analysis pipelines and optionally
    fetches AI-powered recommendations from a local or cloud AI provider.

    DataPilot uses a Metadata-Only AI Pattern — raw data rows are NEVER
    transmitted to any provider. Only lightweight statistical summaries are sent.

    Args:
        df:          Input DataFrame (Pandas or Polars).
        use_ai:      Whether to enable the AI Copilot (default: False).
        ai_provider: AI backend to use when use_ai=True. Options:
                       'ollama'  — local, free, fully private (default)
                       'openai'  — GPT-4o, GPT-4, GPT-3.5-turbo
                       'gemini'  — Google Gemini 1.5 Pro/Flash
                       'claude'  — Anthropic Claude 3.5 Sonnet/Haiku
                       'groq'    — Ultra-fast free-tier Llama3/Mixtral
        ai_model:    Model name within the provider. Falls back to a sensible
                     default per provider if not specified.
        api_key:     API key for cloud providers (openai/gemini/claude/groq).
                     Not required for Ollama.
    """
    local_df, _ = ensure_polars(df)

    # 1. Gather all analytical components
    meta_stats  = summary(local_df)
    miss_report = missing(local_df)
    dup_report  = duplicates(local_df)
    corr_report = correlation(local_df, threshold=0.6)

    meta_stats["duplicates_count"] = dup_report["duplicate_count"]
    miss_list = (
        miss_report.to_dict(orient="records")
        if isinstance(miss_report, pd.DataFrame)
        else miss_report.to_dicts()
    )

    all_corrs = []
    for pair, val in corr_report["strong_positive"]:
        all_corrs.append(f"{pair} (pos: {val})")
    for pair, val in corr_report["strong_negative"]:
        all_corrs.append(f"{pair} (neg: {val})")

    # 2. Print structured findings
    print("=" * 50)
    print("         DATAPILOT AUTOMATED EDA REPORT         ")
    print("=" * 50)
    print(f"📊 Dataset Shape: {meta_stats['rows']} rows × {meta_stats['columns']} columns")
    print(f"💾 Memory Usage:  {meta_stats['memory_usage_mb']} MB")
    print(f"👥 Duplicate Rows: {dup_report['duplicate_count']} ({dup_report['duplicate_percentage']}%)")
    print("-" * 50)

    print("\n🔍 Missing Value Profile:")
    if not miss_list:
        print("   ✅ No missing values detected!")
    else:
        for rec in miss_list:
            print(f"   • {rec['column']}: {rec['missing_count']} missing ({rec['missing_percentage']}%)")

    print("\n🔗 Strong Linear Correlations (|r| >= 0.6):")
    if not all_corrs:
        print("   • No strong linear relationships detected.")
    else:
        for item in all_corrs:
            print(f"   • {item}")

    # 3. Optional AI Copilot
    if use_ai:
        # Resolve settings: per-call overrides take priority over dp.configure() session config
        cfg = get_config()
        resolved_provider = ai_provider or cfg["ai_provider"]
        resolved_model    = ai_model    or cfg["ai_model"]
        resolved_key      = api_key     or cfg["api_key"]

        provider_label = resolved_provider.upper()
        print(f"\n🤖 AI Copilot Insights  [{provider_label}]:")
        ai_output = generate_insights(
            meta=meta_stats,
            missing_list=miss_list,
            strong_relations=all_corrs,
            model_name=resolved_model or "",
            ai_provider=resolved_provider,
            api_key=resolved_key,
        )
        print(ai_output)
        return ai_output

    return None
