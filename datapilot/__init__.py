__version__ = "0.4.3"

# ── Session Configuration ──────────────────────────────────────────────────────

def configure(ai_provider="ollama", ai_model=None, api_key=None):
    """Set AI provider, model, and API key once for the entire session.

    After calling configure(), every use_ai=True function call automatically
    uses these settings — no need to pass api_key on each call.

    Example
    -------
        dp.configure(ai_provider="groq", api_key="gsk_...", ai_model="llama3-70b-8192")
    """
    from .config import configure as _configure
    return _configure(ai_provider=ai_provider, ai_model=ai_model, api_key=api_key)


# ── Analysis ───────────────────────────────────────────────────────────────────

def summary(df):
    """Returns a high-level structural overview of the dataset."""
    from .analysis.summary import summary as _summary
    return _summary(df)

def missing(df):
    """Generates a report detailing missing count and percentage per column."""
    from .analysis.missing import missing as _missing
    return _missing(df)

def duplicates(df):
    """Detects duplicate row counts and percentages using multi-threaded hashing."""
    from .analysis.duplicates import duplicates as _duplicates
    return _duplicates(df)

def correlation(df, threshold=0.6):
    """Calculates a high-speed Pearson correlation matrix and flags strong pairs."""
    from .analysis.correlations import correlation as _correlation
    return _correlation(df, threshold)

def analyze(df, use_ai=False, ai_provider=None, ai_model=None, api_key=None):
    """Runs the full automated EDA pipeline with optional AI insights.

    Reads from dp.configure() by default — pass overrides only when needed.

    ai_provider options: 'ollama' (local/default), 'openai', 'gemini', 'claude', 'groq'
    """
    from .analysis.eda import analyze as _analyze
    return _analyze(df, use_ai=use_ai, ai_provider=ai_provider,
                    ai_model=ai_model, api_key=api_key)

def suggest(df, use_ai=False, ai_provider=None, ai_model=None, api_key=None):
    """Analyses dataset structure and returns actionable preprocessing suggestions.

    Pass use_ai=True to append expert AI recommendations on the flagged issues.
    """
    from .analysis.suggest import suggest as _suggest
    return _suggest(df, use_ai=use_ai, ai_provider=ai_provider,
                    ai_model=ai_model, api_key=api_key)

def compare(df_train, df_test, threshold=0.1, use_ai=False,
            ai_provider=None, ai_model=None, api_key=None):
    """Detects distribution drift between training and test/production DataFrames.

    Pass use_ai=True to receive AI-suggested mitigation strategies.
    """
    from .analysis.compare import compare as _compare
    return _compare(df_train, df_test, threshold, use_ai=use_ai,
                    ai_provider=ai_provider, ai_model=ai_model, api_key=api_key)

def outliers(df, method="both", z_threshold=3.0, iqr_multiplier=1.5,
             use_ai=False, ai_provider=None, ai_model=None, api_key=None):
    """Detects outliers across all numeric columns using IQR and/or Z-score.

    Pass use_ai=True to receive AI-suggested treatment recommendations.
    """
    from .analysis.outliers import outliers as _outliers
    return _outliers(df, method=method, z_threshold=z_threshold,
                     iqr_multiplier=iqr_multiplier, use_ai=use_ai,
                     ai_provider=ai_provider, ai_model=ai_model, api_key=api_key)

def auto_clean(df, drop_null_threshold=0.6, impute_strategy="auto",
               drop_id_columns=True, drop_constant_columns=True,
               use_ai=False, ai_provider=None, ai_model=None, api_key=None):
    """Automatically cleans a dataset — drops bad columns, imputes nulls.

    Pass use_ai=True for an AI explanation of the changes and next steps.
    """
    from .analysis.auto_clean import auto_clean as _auto_clean
    return _auto_clean(
        df,
        drop_null_threshold=drop_null_threshold,
        impute_strategy=impute_strategy,
        drop_id_columns=drop_id_columns,
        drop_constant_columns=drop_constant_columns,
        use_ai=use_ai,
        ai_provider=ai_provider,
        ai_model=ai_model,
        api_key=api_key,
    )

def benchmark(df):
    """Benchmarks DataPilot (Polars) operations vs equivalent Pandas operations."""
    from .analysis.benchmark import benchmark as _benchmark
    return _benchmark(df)


# ── AI Copilot ─────────────────────────────────────────────────────────────────

def ask_ai(df, question, ai_provider=None, ai_model=None, api_key=None):
    """Ask a free-form natural-language question about your dataset.

    DataPilot sends only statistical metadata to the AI — raw data rows are
    never transmitted.  Reads from dp.configure() by default.

    Example
    -------
        dp.configure(ai_provider="groq", api_key="gsk_...")
        dp.ask_ai(df, "Which features carry the most risk of data leakage?")
        dp.ask_ai(df, "Should I log-transform Fare before training?")
    """
    from .ai.chat import ask_ai as _ask_ai
    return _ask_ai(df, question, ai_provider=ai_provider,
                   ai_model=ai_model, api_key=api_key)


# ── Visualization ──────────────────────────────────────────────────────────────

def hist(df, column, bins="auto", hue=None, color="#3b82f6"):
    """Distribution histogram with KDE overlay, mean/median lines and dark theme."""
    from .visualization.histogram import hist as _hist
    return _hist(df, column, bins=bins, hue=hue, color=color)

def box(df, column, group_by=None, orient="v"):
    """Box plot with quartile annotations, coloured median/outlier markers."""
    from .visualization.boxplot import box as _box
    return _box(df, column, group_by=group_by, orient=orient)

def heatmap(df):
    """Lower-triangle Pearson correlation heatmap with dark theme and annotations."""
    from .visualization.heatmap import heatmap as _heatmap
    return _heatmap(df)

def scatter(df, x, y, hue=None, trendline=True):
    """Scatter plot with optional OLS regression trendline and dark theme."""
    from .visualization.scatter import scatter as _scatter
    return _scatter(df, x, y, hue=hue, trendline=trendline)

def violin(df, column, group_by=None):
    """Violin plot combining box plot + KDE for richer distribution insight."""
    from .visualization.violin import violin as _violin
    return _violin(df, column, group_by=group_by)

def visualize_ai(df, prompt, ai_provider=None, ai_model=None, api_key=None):
    """Ask the AI to choose and draw the right chart from a plain-English prompt.

    DataPilot sends only column names + dtypes to the AI (no raw data).
    The chart is rendered 100% locally with Seaborn.

    Example
    -------
        dp.configure(ai_provider="groq", api_key="gsk_...")
        dp.visualize_ai(df, "Show the relation between Age and Survived")
        dp.visualize_ai(df, "Distribution of Fare by Pclass")
        dp.visualize_ai(df, "Correlation heatmap of all numeric columns")
    """
    from .visualization.visualize_ai import visualize_ai as _viz_ai
    return _viz_ai(df, prompt, ai_provider=ai_provider,
                   ai_model=ai_model, api_key=api_key)


# ── Machine Learning ───────────────────────────────────────────────────────────

def classification_report(y_true, y_pred, average="auto"):
    """Calculates classification performance metrics (binary or multi-class)."""
    from .ml.classification import classification_report as _cr
    return _cr(y_true, y_pred, average=average)

def regression_report(y_true, y_pred):
    """Calculates comprehensive regression evaluation metrics (MAE, RMSE, R², MAPE)."""
    from .ml.regression import regression_report as _rr
    return _rr(y_true, y_pred)

def diagnose(train_score, test_score, metric_name="Accuracy"):
    """Diagnoses model health — overfitting, underfitting, or optimal state."""
    from .ml.diagnostics import diagnose as _diagnose
    return _diagnose(train_score, test_score, metric_name)


# ── Dashboard ──────────────────────────────────────────────────────────────────

def dashboard(df, output_path="datapilot_report.html"):
    """Generates a standalone, beautifully styled HTML dashboard from the dataset."""
    from .dashboard.dashboard import dashboard as _dashboard
    return _dashboard(df, output_path)


__all__ = [
    "__version__",
    # Config
    "configure",
    # Analysis
    "summary", "missing", "duplicates", "correlation", "analyze",
    "suggest", "compare", "outliers", "auto_clean", "benchmark",
    # AI Copilot
    "ask_ai",
    # Visualization
    "hist", "box", "heatmap", "scatter", "violin", "visualize_ai",
    # ML
    "classification_report", "regression_report", "diagnose",
    # Dashboard
    "dashboard",
]
