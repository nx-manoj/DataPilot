"""
datapilot.ai.chat — dp.ask_ai() conversational assistant.

Computes a rich metadata snapshot of the dataset (shape, dtypes, null
profile, correlations, basic stats) and streams it to the configured
AI provider together with the user's free-form question.  Raw data rows
are never transmitted — only the statistical summary.

Example
-------
    dp.ask_ai(df, "Which features are most likely to cause overfitting?")
    dp.ask_ai(df, "Should I normalise Age or log-transform Fare first?")
"""

from __future__ import annotations

from typing import Union, Optional, Dict, Any, List
import polars as pl
import pandas as pd
import math

from ..utils.validation import ensure_polars
from ..config import get_config
from .factory import get_provider


# ── Helpers ───────────────────────────────────────────────────────────────────

class _SilentStr(str):
    """A str subclass whose __repr__ returns '' so Jupyter doesn't auto-display
    the return value of ask_ai() a second time (it's already printed)."""
    def __repr__(self) -> str:
        return ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_metadata_snapshot(local_df: pl.DataFrame) -> str:
    """Return a compact, human-readable text block with dataset statistics."""
    rows, cols = local_df.height, local_df.width
    null_counts = local_df.null_count()

    dtype_summary: Dict[str, List[str]] = {"numeric": [], "string": [], "other": []}
    missing_parts: List[str] = []
    stats_parts:   List[str] = []

    for col in local_df.columns:
        dtype = local_df[col].dtype
        null_ct  = null_counts[col][0]
        null_pct = round(null_ct / rows * 100, 2) if rows > 0 else 0

        if dtype.is_numeric():
            dtype_summary["numeric"].append(col)
            series = local_df[col].drop_nulls()
            if series.len() > 0:
                mn   = round(series.mean(), 4)
                std  = round(series.std() or 0, 4)
                mn_v = round(series.min(), 4)
                mx_v = round(series.max(), 4)
                stats_parts.append(
                    f"  {col}: mean={mn}, std={std}, min={mn_v}, max={mx_v}"
                    + (f", nulls={null_ct}({null_pct}%)" if null_ct else "")
                )
        elif dtype in (pl.Utf8, pl.String):
            dtype_summary["string"].append(col)
            n_unique = local_df[col].n_unique()
            stats_parts.append(
                f"  {col}: string, {n_unique} unique values"
                + (f", nulls={null_ct}({null_pct}%)" if null_ct else "")
            )
        else:
            dtype_summary["other"].append(col)

        if null_ct > 0:
            missing_parts.append(f"{col}({null_pct}%)")

    # Correlation pairs
    numeric_cols = dtype_summary["numeric"]
    corr_parts: List[str] = []
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = local_df.select(numeric_cols).to_pandas().corr()
            cols_list   = corr_matrix.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    val = corr_matrix.iloc[i, j]
                    if pd.notna(val) and abs(val) >= 0.5:
                        corr_parts.append(f"{cols_list[i]} ↔ {cols_list[j]}: r={round(val, 3)}")
        except Exception:
            pass

    lines = [
        f"Dataset: {rows:,} rows × {cols} columns",
        f"Numeric columns ({len(dtype_summary['numeric'])}): {', '.join(dtype_summary['numeric']) or 'none'}",
        f"String columns ({len(dtype_summary['string'])}): {', '.join(dtype_summary['string']) or 'none'}",
        f"Missing values: {', '.join(missing_parts) if missing_parts else 'none'}",
        f"Correlations (|r|≥0.5): {', '.join(corr_parts) if corr_parts else 'none detected'}",
        "",
        "Column-level stats:",
    ] + stats_parts

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def ask_ai(
    df: Union[pd.DataFrame, pl.DataFrame],
    question: str,
    ai_provider: Optional[str] = None,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Ask a free-form question about your dataset and receive an AI answer.

    DataPilot computes a rich statistical snapshot of the dataset and sends
    it together with your question to the configured AI provider.  Raw data
    rows are NEVER transmitted.

    Args:
        df:          Input DataFrame (Pandas or Polars).
        question:    A plain-English question about the dataset.
        ai_provider: Override the globally configured provider for this call.
        ai_model:    Override the globally configured model for this call.
        api_key:     Override the globally configured API key for this call.

    Returns:
        Plain-text AI response string.

    Example
    -------
        dp.configure(ai_provider="groq", api_key="gsk_...")
        dp.ask_ai(df, "Which columns have the most leakage risk?")
        dp.ask_ai(df, "Should I drop or impute the Age column?")
    """
    cfg = get_config()
    provider_name = ai_provider or cfg["ai_provider"]
    model_name    = ai_model    or cfg["ai_model"]
    key           = api_key     or cfg["api_key"]

    local_df, _ = ensure_polars(df)
    snapshot     = _build_metadata_snapshot(local_df)

    system_prompt = (
        "You are DataPilot, an expert data science assistant. "
        "You will be given a statistical snapshot of a dataset (no raw rows) "
        "and a question from a data scientist. "
        "Give a direct, technically precise answer in 3-5 sentences. "
        "Focus only on what the statistics reveal — do not speculate beyond the data."
    )

    user_prompt = (
        f"=== Dataset Statistical Snapshot ===\n"
        f"{snapshot}\n\n"
        f"=== User Question ===\n"
        f"{question}"
    )

    print(f"\n🤖 DataPilot AI [{provider_name.upper()}] — Answering your question...")
    print(f"   Q: {question}\n")

    try:
        provider = get_provider(
            ai_provider=provider_name,
            ai_model=model_name,
            api_key=key,
        )
        # Temporarily override the standard prompt-building with our chat-specific one
        answer = provider._call_with_raw_prompts(system_prompt, user_prompt)
        print(f"   A: {answer}\n")
        return _SilentStr(answer)
    except AttributeError:
        # Fallback: provider doesn't have _call_with_raw_prompts yet — use generate()
        # Build a minimal meta dict the standard generate() expects
        meta = {
            "rows": local_df.height,
            "columns": local_df.width,
            "memory_usage_mb": 0,
            "engine_detected": "polars",
            "duplicates_count": 0,
        }
        answer = provider.generate(meta, [], [f"User question: {question}\n\nContext:\n{snapshot}"])
        print(f"   A: {answer}\n")
        return _SilentStr(answer)
    except ValueError as e:
        msg = f"⚠️  Configuration error: {e}"
        print(msg)
        return _SilentStr(msg)
    except Exception as e:
        import logging
        logging.error(f"AI error: {e}", exc_info=True)
        msg = f"⚠️  AI error: {e}"
        print(msg)
        return _SilentStr(msg)
