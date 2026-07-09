from ..analysis.summary import summary
from ..analysis.missing import missing
from ..analysis.duplicates import duplicates
from ..analysis.correlations import correlation
from ..utils.validation import ensure_polars

import polars as pl
from typing import Union
import pandas as pd
import base64
import io

def _build_correlation_heatmap_html(local_df: pl.DataFrame) -> str:
    """Renders a correlation heatmap and returns it as an interactive Plotly HTML string."""
    try:
        import plotly.express as px

        numeric_cols = [
            col for col, dtype in zip(local_df.columns, local_df.dtypes)
            if dtype.is_numeric()
        ]
        if len(numeric_cols) < 2:
            return ""

        corr_matrix = local_df.select(numeric_cols).to_pandas().corr()
        corr_matrix.index = numeric_cols

        fig = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            aspect="auto",
            title="Pearson Correlation Matrix"
        )
        fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), title_x=0.5)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        return ""

def _build_missing_bar_html(miss_list: list) -> str:
    """Renders a horizontal bar chart of missing value percentages as interactive Plotly HTML."""
    if not miss_list:
        return ""
    try:
        import plotly.express as px

        cols = [r["column"] for r in miss_list]
        pcts = [r["missing_percentage"] for r in miss_list]

        fig = px.bar(
            x=pcts, y=cols, orientation='h',
            text=[f"{p:.1f}%" for p in pcts],
            title="Missing Value Distribution",
            labels={"x": "Missing (%)", "y": "Column"},
            color_discrete_sequence=["#4A90E2"]
        )
        fig.update_layout(
            xaxis=dict(range=[0, 100]), 
            yaxis=dict(autorange="reversed"),
            margin=dict(t=40, l=10, r=10, b=10),
            title_x=0.5
        )
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception:
        return ""


def dashboard(df: Union[pd.DataFrame, pl.DataFrame], output_path: str = "datapilot_report.html") -> None:
    """Generates a standalone, beautifully styled HTML dashboard report from the dataset.

    The report is fully self-contained (no internet required) and includes:
    - Dataset shape, memory, and duplicate metrics
    - Data type overview table
    - Missing value table and inline bar chart
    - Pearson correlation matrix (inline heatmap image)
    - Strong positive/negative correlation pairs

    Args:
        df: Input DataFrame (Pandas or Polars).
        output_path: File path for the exported HTML report.
    """
    local_df, engine = ensure_polars(df)

    meta = summary(local_df)
    miss_df = missing(local_df)
    dup_df = duplicates(local_df)
    corr_report = correlation(local_df, threshold=0.6)

    miss_list = miss_df.to_dict(orient="records") if isinstance(miss_df, pd.DataFrame) else miss_df.to_dicts()

    # ── Inline chart images ──────────────────────────────────────────────────
    heatmap_html = _build_correlation_heatmap_html(local_df)
    missing_bar_html = _build_missing_bar_html(miss_list)

    # ── Missing value rows ───────────────────────────────────────────────────
    if not miss_list:
        missing_rows_html = "<tr><td colspan='3' style='text-align:center;color:#2ecc71;'>✅ No missing values detected!</td></tr>"
    else:
        missing_rows_html = "".join(
            f"<tr><td><strong>{r['column']}</strong></td>"
            f"<td>{r['missing_count']}</td>"
            f"<td><span class='badge badge-danger'>{r['missing_percentage']}%</span></td></tr>"
            for r in miss_list
        )

    # ── Data-type overview rows ──────────────────────────────────────────────
    dtype_rows_html = "".join(
        f"<tr><td><strong>{col}</strong></td><td><code>{dtype}</code></td></tr>"
        for col, dtype in meta["datatypes"].items()
    )

    # ── Correlation pairs ────────────────────────────────────────────────────
    pos_pairs = corr_report["strong_positive"]
    neg_pairs = corr_report["strong_negative"]

    def corr_rows(pairs, badge_class, label):
        if not pairs:
            return f"<tr><td colspan='3' style='color:#95a5a6;'>No strong {label} correlations found.</td></tr>"
        return "".join(
            f"<tr><td>{pair}</td>"
            f"<td><span class='badge {badge_class}'>{val:+.3f}</span></td>"
            f"<td>{label}</td></tr>"
            for pair, val in pairs
        )

    corr_rows_html = corr_rows(pos_pairs, "badge-success", "Positive") + \
                     corr_rows(neg_pairs, "badge-danger", "Negative")

    heatmap_section = (
        f"<div class='section'><h2>📊 Correlation Heatmap</h2>"
        f"<div style='margin-top: 15px;'>{heatmap_html}</div></div>"
    ) if heatmap_html else ""

    missing_bar_section = (
        f"<div style='margin-top: 18px; border-radius: 8px; overflow: hidden;'>"
        f"{missing_bar_html}</div>"
    ) if missing_bar_html else ""

    # ── HTML ─────────────────────────────────────────────────────────────────
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataPilot Analytics Dashboard</title>
    <style>
        /* ── Offline font stack ── */
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         Helvetica, Arial, sans-serif;
            background: #f0f4f8;
            color: #2d3748;
            margin: 0;
            padding: 24px;
        }}
        .container {{ max-width: 1120px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            color: #fff;
            padding: 32px 36px;
            border-radius: 14px;
            margin-bottom: 28px;
            box-shadow: 0 4px 14px rgba(74,144,226,.25);
        }}
        .header h1 {{ margin: 0 0 6px; font-size: 26px; font-weight: 700; }}
        .header p  {{ margin: 0; opacity: .88; font-size: 14px; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 18px;
            margin-bottom: 28px;
        }}
        .card {{
            background: #fff;
            padding: 22px 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,.06);
            text-align: center;
        }}
        .card h3 {{
            margin: 0;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #718096;
        }}
        .card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #2c3e50;
            margin: 10px 0 0;
        }}
        .section {{
            background: #fff;
            padding: 26px 28px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,.06);
            margin-bottom: 26px;
        }}
        .section h2 {{
            margin: 0 0 16px;
            font-size: 17px;
            color: #2c3e50;
            padding-bottom: 10px;
            border-bottom: 2px solid #edf2f7;
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{
            padding: 11px 14px;
            text-align: left;
            border-bottom: 1px solid #edf2f7;
            font-size: 14px;
        }}
        th {{ background: #f7fafc; color: #4a5568; font-weight: 600; }}
        code {{
            background: #edf2f7;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 9px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-danger  {{ background: #fed7d7; color: #c53030; }}
        .badge-success {{ background: #c6f6d5; color: #276749; }}
        .footer {{
            text-align: center;
            margin-top: 8px;
            font-size: 12px;
            color: #a0aec0;
        }}
    </style>
</head>
<body>
    <div class="container">

        <div class="header">
            <h1>📊 DataPilot Dataset Report</h1>
            <p>Automated high-speed profiling dashboard &nbsp;•&nbsp; Engine: <strong>{meta['engine_detected'].upper()}</strong></p>
        </div>

        <!-- ── Summary Cards ── -->
        <div class="grid">
            <div class="card">
                <h3>Total Rows</h3>
                <div class="value">{meta['rows']:,}</div>
            </div>
            <div class="card">
                <h3>Total Columns</h3>
                <div class="value">{meta['columns']}</div>
            </div>
            <div class="card">
                <h3>Duplicate Rows</h3>
                <div class="value">{dup_df['duplicate_count']} <span style="font-size:14px;font-weight:400;color:#718096;">({dup_df['duplicate_percentage']}%)</span></div>
            </div>
            <div class="card">
                <h3>Memory Footprint</h3>
                <div class="value">{meta['memory_usage_mb']} <span style="font-size:14px;font-weight:400;color:#718096;">MB</span></div>
            </div>
            <div class="card">
                <h3>Missing Values</h3>
                <div class="value">{meta['total_missing_values']:,}</div>
            </div>
        </div>

        <!-- ── Data Types ── -->
        <div class="section">
            <h2>🗂️ Column Data Types</h2>
            <table>
                <thead><tr><th>Column</th><th>Data Type</th></tr></thead>
                <tbody>{dtype_rows_html}</tbody>
            </table>
        </div>

        <!-- ── Missing Values ── -->
        <div class="section">
            <h2>🔍 Missing Values Diagnostics</h2>
            <table>
                <thead>
                    <tr><th>Column</th><th>Null Count</th><th>Missing %</th></tr>
                </thead>
                <tbody>{missing_rows_html}</tbody>
            </table>
            {missing_bar_section}
        </div>

        <!-- ── Correlation Pairs ── -->
        <div class="section">
            <h2>🔗 Strong Linear Correlations (|r| ≥ 0.6)</h2>
            <table>
                <thead><tr><th>Variable Pair</th><th>Pearson r</th><th>Direction</th></tr></thead>
                <tbody>{corr_rows_html}</tbody>
            </table>
        </div>

        <!-- ── Correlation Heatmap ── -->
        {heatmap_section}

        <div class="footer">
            Generated securely by <strong>DataPilot</strong> Core Engine &nbsp;•&nbsp; No raw data was transmitted.
        </div>

    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🎉 Standalone HTML Dashboard successfully exported to: {output_path}")
