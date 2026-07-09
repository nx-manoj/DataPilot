"""
datapilot.visualization.visualize_ai — dp.visualize_ai()

Natural-language driven visualization.  The user writes a plain-English
prompt; the AI decides which chart type to generate and which columns to
use, then DataPilot builds and shows the chart via Plotly.

The AI returns a compact JSON decision block (chart_type, x, y, hue,
title) which is then executed locally — no plotting code is ever run by
the model.

Example
-------
    dp.visualize_ai(df, "Show me the relation between Age and Survived")
    dp.visualize_ai(df, "Distribution of Fare by passenger class")
    dp.visualize_ai(df, "Correlation heatmap of all numeric columns")
    dp.visualize_ai(df, "Compare Sex and Survived side by side")
"""

from __future__ import annotations

import json
import re
from typing import Union, Optional

import numpy as np
import polars as pl
import pandas as pd
import plotly.express as px

from ..utils.validation import ensure_polars
from ..config import get_config
from ..ai.factory import get_provider


# ── AI decision parsing ───────────────────────────────────────────────────────

def _ask_ai_for_chart_decision(
    columns: list,
    dtypes: dict,
    user_prompt: str,
    provider_name: str,
    model_name: Optional[str],
    api_key: Optional[str],
) -> dict:
    """Ask the AI which chart to draw.  Returns a parsed dict."""
    col_info = "\n".join(
        f"  - {col}: {dtype}" for col, dtype in dtypes.items()
    )
    system_prompt = (
        "You are a data visualization assistant. "
        "Given a list of dataset columns and the user's chart request, "
        "decide exactly what chart to plot. "
        "Respond ONLY with a JSON object — no explanation, no markdown fences. "
        "The JSON must have these fields:\n"
        '  "chart_type": one of ["histogram","boxplot","scatter","bar","violin","heatmap","countplot","line","pairplot"]\n'
        '  "x": column name or null\n'
        '  "y": column name or null\n'
        '  "hue": column name or null (for colour grouping)\n'
        '  "title": a short descriptive chart title\n'
        "Choose the most insightful chart type for the request. "
        "Use null when a field is not applicable."
    )
    user_msg = (
        f"Dataset columns:\n{col_info}\n\n"
        f"User request: {user_prompt}"
    )

    try:
        provider = get_provider(
            ai_provider=provider_name,
            ai_model=model_name,
            api_key=api_key,
        )
        raw = provider._call_with_raw_prompts(system_prompt, user_msg)
    except Exception as e:
        return {"error": str(e)}

    # Strip markdown fences if present
    raw_clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw_clean = re.sub(r"```\s*$", "", raw_clean.strip(), flags=re.MULTILINE)

    # Extract JSON from the response (tolerates extra text)
    match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if not match:
        return {"error": f"AI returned non-JSON: {raw}"}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}\nRaw: {raw}"}


# ── Chart renderer ────────────────────────────────────────────────────────────

def _render_chart(pdf: pd.DataFrame, decision: dict):
    """Render the chart described by the AI decision dict using Plotly."""
    import plotly.graph_objects as go
    
    chart_type = decision.get("chart_type", "").lower()
    x     = decision.get("x")
    y     = decision.get("y")
    hue   = decision.get("hue")
    title = decision.get("title", "DataPilot Visualization")

    # Validate columns exist
    for col_ref in [x, y, hue]:
        if col_ref and col_ref not in pdf.columns:
            print(f"⚠️  Column '{col_ref}' not found — skipping chart.")
            return None

    try:
        fig = None
        if chart_type == "histogram":
            col = x or y
            fig = px.histogram(pdf, x=col, color=hue, barmode="overlay", marginal="violin")

        elif chart_type == "boxplot":
            fig = px.box(pdf, x=x, y=y, color=hue)

        elif chart_type == "violin":
            fig = px.violin(pdf, x=x, y=y, color=hue, box=True, points="all")

        elif chart_type == "scatter":
            fig = px.scatter(pdf, x=x, y=y, color=hue, opacity=0.75)
            # Add a regression line if hue is not set
            if not hue and x and y:
                _x = pdf[x].dropna()
                _y = pdf[y].dropna()
                common_idx = _x.index.intersection(_y.index)
                if len(common_idx) > 2:
                    z = np.polyfit(_x[common_idx], _y[common_idx], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(_x.min(), _x.max(), 200)
                    fig.add_trace(go.Scatter(
                        x=x_line, y=p(x_line),
                        mode='lines',
                        line=dict(color="#f43f5e", width=2, dash='dash'),
                        name="trend"
                    ))

        elif chart_type == "bar":
            fig = px.bar(pdf, x=x, y=y, color=hue)

        elif chart_type == "countplot":
            col = x or y
            fig = px.histogram(pdf, x=col, color=hue, barmode="group")

        elif chart_type == "line":
            fig = px.line(pdf, x=x, y=y, color=hue)

        elif chart_type == "heatmap":
            numeric_cols = pdf.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) < 2:
                print("⚠️  Not enough numeric columns for a heatmap.")
                return None
            corr = pdf[numeric_cols].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto"
            )

        elif chart_type == "pairplot":
            numeric_cols = pdf.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                return None
            
            pdf_sample = pdf.dropna(subset=numeric_cols).head(500)
            
            fig = px.scatter_matrix(
                pdf_sample,
                dimensions=numeric_cols,
                color=hue
            )
            fig.update_traces(diagonal_visible=False)

        else:
            print(f"⚠️  Unknown chart type returned by AI: '{chart_type}'")
            return None
            
        fig.update_layout(
            template="plotly_dark",
            title=title,
            title_x=0.5,
            margin=dict(t=50, l=20, r=20, b=20)
        )
        return fig

    except Exception as e:
        import logging
        logging.error(f"Chart rendering error: {e}", exc_info=True)
        print(f"⚠️  Chart rendering error: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def visualize_ai(
    df: Union[pd.DataFrame, pl.DataFrame],
    prompt: str,
    ai_provider: Optional[str] = None,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Ask the AI to choose and draw the right chart from a plain-English prompt.

    DataPilot sends only the column names and their data types to the AI —
    no raw values are ever transmitted.  The AI decides the chart type and
    columns; the rendering is done 100% locally using Plotly.

    Args:
        df:          Input DataFrame (Pandas or Polars).
        prompt:      Natural language description of what you want to see.
        ai_provider: Override the globally configured provider for this call.
        ai_model:    Override the globally configured model for this call.
        api_key:     Override the globally configured API key for this call.

    Returns:
        Interactive Plotly Figure or None.
        
    Example
    -------
        dp.configure(ai_provider="groq", api_key="gsk_...")

        dp.visualize_ai(df, "Show the relation between Age and Survived")
        dp.visualize_ai(df, "Distribution of Fare for each passenger class")
        dp.visualize_ai(df, "Correlation heatmap of numeric columns")
    """
    cfg           = get_config()
    provider_name = ai_provider or cfg["ai_provider"]
    model_name    = ai_model    or cfg["ai_model"]
    key           = api_key     or cfg["api_key"]

    local_df, _ = ensure_polars(df)
    pdf = local_df.to_pandas()

    columns = local_df.columns
    dtypes  = {col: str(local_df[col].dtype) for col in columns}

    print(f"\n🎨 DataPilot visualize_ai — processing: \"{prompt}\"")
    print(f"   🤖 Asking AI [{provider_name.upper()}] to decide the best chart...\n")

    decision = _ask_ai_for_chart_decision(
        columns=columns,
        dtypes=dtypes,
        user_prompt=prompt,
        provider_name=provider_name,
        model_name=model_name,
        api_key=key,
    )

    if "error" in decision:
        print(f"⚠️  Could not determine chart type: {decision['error']}")
        return None

    chart_type = decision.get("chart_type", "unknown")
    x_col = decision.get("x") or ""
    y_col = decision.get("y") or ""
    hue_col = decision.get("hue") or ""

    print(f"   ✅ Chart decided: {chart_type.upper()}")
    if x_col: print(f"      x  → {x_col}")
    if y_col: print(f"      y  → {y_col}")
    if hue_col: print(f"      hue → {hue_col}")
    print()

    fig = _render_chart(pdf, decision)
    if fig:
        fig.show()
    return fig
