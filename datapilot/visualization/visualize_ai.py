"""
datapilot.visualization.visualize_ai — dp.visualize_ai()

Natural-language driven visualization.  The user writes a plain-English
prompt; the AI decides which chart type to generate and which columns to
use, then DataPilot builds and shows the chart via Seaborn/Matplotlib.

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

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import polars as pl
import pandas as pd

from ..utils.validation import ensure_polars
from ..config import get_config
from ..ai.factory import get_provider


# ── Seaborn theme applied once ────────────────────────────────────────────────
_PALETTE   = "mako"
_BG_COLOR  = "#0f172a"
_GRID_COLOR = "#1e293b"
_TEXT_COLOR = "#e2e8f0"
_ACCENT    = "#3b82f6"


def _apply_dp_theme() -> None:
    """Apply the DataPilot dark seaborn theme."""
    sns.set_theme(style="darkgrid", palette=_PALETTE)
    mpl.rcParams.update({
        "figure.facecolor":  _BG_COLOR,
        "axes.facecolor":    _BG_COLOR,
        "axes.edgecolor":    _GRID_COLOR,
        "axes.labelcolor":   _TEXT_COLOR,
        "axes.titlecolor":   _TEXT_COLOR,
        "xtick.color":       _TEXT_COLOR,
        "ytick.color":       _TEXT_COLOR,
        "grid.color":        _GRID_COLOR,
        "text.color":        _TEXT_COLOR,
        "legend.facecolor":  "#1e293b",
        "legend.edgecolor":  _GRID_COLOR,
        "figure.dpi":        120,
    })


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

    # Extract JSON from the response (tolerates extra text)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"error": f"AI returned non-JSON: {raw}"}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}\nRaw: {raw}"}


# ── Chart renderer ────────────────────────────────────────────────────────────

def _render_chart(pdf: pd.DataFrame, decision: dict) -> None:
    """Render the chart described by the AI decision dict."""
    chart_type = decision.get("chart_type", "").lower()
    x     = decision.get("x")
    y     = decision.get("y")
    hue   = decision.get("hue")
    title = decision.get("title", "DataPilot Visualization")

    # Validate columns exist
    for col_ref in [x, y, hue]:
        if col_ref and col_ref not in pdf.columns:
            print(f"⚠️  Column '{col_ref}' not found — skipping chart.")
            return

    _apply_dp_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if chart_type == "histogram":
            col = x or y
            sns.histplot(data=pdf, x=col, hue=hue, kde=True,
                         palette=_PALETTE, ax=ax, bins="auto",
                         edgecolor="none", alpha=0.8)

        elif chart_type == "boxplot":
            sns.boxplot(data=pdf, x=x, y=y, hue=hue,
                        palette=_PALETTE, ax=ax,
                        linewidth=1.2, flierprops={"marker": "o", "markersize": 3})

        elif chart_type == "violin":
            sns.violinplot(data=pdf, x=x, y=y, hue=hue,
                           palette=_PALETTE, ax=ax, inner="quartile",
                           linewidth=0.8)

        elif chart_type == "scatter":
            sns.scatterplot(data=pdf, x=x, y=y, hue=hue,
                            palette=_PALETTE, ax=ax,
                            alpha=0.75, s=40, linewidth=0)
            # Add a regression line if hue is not set
            if not hue and x and y:
                _x = pdf[x].dropna()
                _y = pdf[y].dropna()
                common_idx = _x.index.intersection(_y.index)
                if len(common_idx) > 2:
                    z = np.polyfit(_x[common_idx], _y[common_idx], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(_x.min(), _x.max(), 200)
                    ax.plot(x_line, p(x_line), color="#f43f5e",
                            linewidth=1.8, linestyle="--", label="trend")
                    ax.legend()

        elif chart_type == "bar":
            sns.barplot(data=pdf, x=x, y=y, hue=hue,
                        palette=_PALETTE, ax=ax,
                        errorbar=None, edgecolor="none")

        elif chart_type == "countplot":
            col = x or y
            sns.countplot(data=pdf, x=col, hue=hue,
                          palette=_PALETTE, ax=ax, edgecolor="none")

        elif chart_type == "line":
            sns.lineplot(data=pdf, x=x, y=y, hue=hue,
                         palette=_PALETTE, ax=ax, linewidth=2)

        elif chart_type == "heatmap":
            numeric_cols = pdf.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) < 2:
                print("⚠️  Not enough numeric columns for a heatmap.")
                plt.close()
                return
            corr = pdf[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                        cmap="coolwarm", vmin=-1, vmax=1,
                        linewidths=0.4, cbar_kws={"shrink": 0.8})

        elif chart_type == "pairplot":
            plt.close()  # pairplot creates its own figure
            g = sns.pairplot(
                pdf.select_dtypes(include="number").dropna().head(500),
                diag_kind="kde", plot_kws={"alpha": 0.5},
                palette=_PALETTE,
            )
            g.figure.patch.set_facecolor(_BG_COLOR)
            g.figure.suptitle(title, y=1.02,
                               color=_TEXT_COLOR, fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.show()
            return

        else:
            print(f"⚠️  Unknown chart type returned by AI: '{chart_type}'")
            plt.close()
            return

        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=_TEXT_COLOR, pad=14)
        if x: ax.set_xlabel(x, fontsize=10)
        if y: ax.set_ylabel(y, fontsize=10)
        fig.patch.set_facecolor(_BG_COLOR)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        plt.close()
        print(f"⚠️  Chart rendering error: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def visualize_ai(
    df: Union[pd.DataFrame, pl.DataFrame],
    prompt: str,
    ai_provider: Optional[str] = None,
    ai_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Ask the AI to choose and draw the right chart from a plain-English prompt.

    DataPilot sends only the column names and their data types to the AI —
    no raw values are ever transmitted.  The AI decides the chart type and
    columns; the rendering is done 100% locally.

    Args:
        df:          Input DataFrame (Pandas or Polars).
        prompt:      Natural language description of what you want to see.
        ai_provider: Override the globally configured provider for this call.
        ai_model:    Override the globally configured model for this call.
        api_key:     Override the globally configured API key for this call.

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
        return

    chart_type = decision.get("chart_type", "unknown")
    x_col = decision.get("x") or ""
    y_col = decision.get("y") or ""
    hue_col = decision.get("hue") or ""

    print(f"   ✅ Chart decided: {chart_type.upper()}")
    if x_col: print(f"      x  → {x_col}")
    if y_col: print(f"      y  → {y_col}")
    if hue_col: print(f"      hue → {hue_col}")
    print()

    _render_chart(pdf, decision)
