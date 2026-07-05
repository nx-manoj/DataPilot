__version__ = "0.3.0"

# ── Analysis ──────────────────────────────────────────────────────────────────

def summary(df):
    """Calculates lightning fast structural overview metrics of the dataset."""
    from .analysis.summary import summary as _summary
    return _summary(df)

def missing(df):
    """Generates a report detailing missing count and percentage per column."""
    from .analysis.missing import missing as _missing
    return _missing(df)

def duplicates(df):
    """Calculates duplicate row counts and percentages using multi-threaded hashing."""
    from .analysis.duplicates import duplicates as _duplicates
    return _duplicates(df)

def correlation(df, threshold=0.6):
    """Calculates a high-speed Pearson correlation matrix and flags strong pairs."""
    from .analysis.correlations import correlation as _correlation
    return _correlation(df, threshold)

def analyze(df, use_ai=False, ai_provider="ollama", ai_model=None, api_key=None):
    """Runs automated EDA with optional AI insights from local or cloud providers.

    ai_provider options: 'ollama' (local/default), 'openai', 'gemini', 'claude', 'groq'
    """
    from .analysis.eda import analyze as _analyze
    return _analyze(df, use_ai=use_ai, ai_provider=ai_provider,
                    ai_model=ai_model, api_key=api_key)

def suggest(df):
    """Analyses dataset structure and returns actionable preprocessing suggestions."""
    from .analysis.suggest import suggest as _suggest
    return _suggest(df)

def compare(df_train, df_test, threshold=0.1):
    """Detects distribution drift between a training and test/production DataFrame."""
    from .analysis.compare import compare as _compare
    return _compare(df_train, df_test, threshold)

def outliers(df, method="both", z_threshold=3.0, iqr_multiplier=1.5):
    """Detects outliers in all numeric columns using IQR and/or Z-score methods."""
    from .analysis.outliers import outliers as _outliers
    return _outliers(df, method=method, z_threshold=z_threshold, iqr_multiplier=iqr_multiplier)

def auto_clean(df, drop_null_threshold=0.6, impute_strategy="auto",
               drop_id_columns=True, drop_constant_columns=True):
    """Automatically cleans a dataset — drops bad columns, imputes nulls."""
    from .analysis.auto_clean import auto_clean as _auto_clean
    return _auto_clean(
        df,
        drop_null_threshold=drop_null_threshold,
        impute_strategy=impute_strategy,
        drop_id_columns=drop_id_columns,
        drop_constant_columns=drop_constant_columns,
    )

def benchmark(df):
    """Benchmarks DataPilot (Polars) operations vs equivalent Pandas operations."""
    from .analysis.benchmark import benchmark as _benchmark
    return _benchmark(df)

# ── Visualization ─────────────────────────────────────────────────────────────

def hist(df, column, bins=10):
    """Generates a clean distribution histogram for a selected numerical feature."""
    from .visualization.histogram import hist as _hist
    return _hist(df, column, bins)

def box(df, column):
    """Generates a clean box plot to showcase distribution quartiles and outliers."""
    from .visualization.boxplot import box as _box
    return _box(df, column)

def heatmap(df):
    """Generates a visual correlation matrix heatmap for all numerical features."""
    from .visualization.heatmap import heatmap as _heatmap
    return _heatmap(df)

# ── Machine Learning ──────────────────────────────────────────────────────────

def classification_report(y_true, y_pred, average="auto"):
    """Calculates key classification performance metrics (binary or multi-class)."""
    from .ml.classification import classification_report as _cr
    return _cr(y_true, y_pred, average=average)

def regression_report(y_true, y_pred):
    """Calculates comprehensive regression evaluation metrics (MAE, RMSE, R², MAPE)."""
    from .ml.regression import regression_report as _rr
    return _rr(y_true, y_pred)

def diagnose(train_score, test_score, metric_name="Accuracy"):
    """Diagnoses model health, highlighting overfitting, underfitting, or optimal states."""
    from .ml.diagnostics import diagnose as _diagnose
    return _diagnose(train_score, test_score, metric_name)

# ── Dashboard ─────────────────────────────────────────────────────────────────

def dashboard(df, output_path="datapilot_report.html"):
    """Generates a standalone, beautifully styled HTML dashboard report from the dataset."""
    from .dashboard.dashboard import dashboard as _dashboard
    return _dashboard(df, output_path)


__all__ = [
    "__version__",
    # Analysis
    "summary", "missing", "duplicates", "correlation", "analyze",
    "suggest", "compare", "outliers", "auto_clean", "benchmark",
    # Visualization
    "hist", "box", "heatmap",
    # ML
    "classification_report", "regression_report", "diagnose",
    # Dashboard
    "dashboard",
]
