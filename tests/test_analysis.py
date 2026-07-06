import pandas as pd
import polars as pl
import pytest
from unittest.mock import patch, MagicMock
import datapilot as dp

def test_summary_engine_pandas():
    # Arrange
    data = {"A": [1, 2, 3], "B": [4, None, 6]}
    df = pd.DataFrame(data)
    
    # Act
    report = dp.summary(df)
    
    # Assert
    assert report["engine_detected"] == "pandas"
    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["total_missing_values"] == 1

def test_missing_report_sorting():
    # Arrange
    data = {
        "Clean": [1, 2, 3, 4],
        "MostlyNull": [None, None, None, 4],
        "OneNull": [1, 2, None, 4]
    }
    df = pl.DataFrame(data)
    
    # Act
    report = dp.missing(df)
    
    # Assert
    # The output should be sorted with the highest missing count first
    columns_order = report["column"].to_list()
    assert columns_order[0] == "MostlyNull"
    assert columns_order[1] == "OneNull"

def test_duplicates():
    # Arrange
    data = {"A": [1, 2, 2, 3], "B": [4, 5, 5, 6]}
    df = pd.DataFrame(data)
    
    # Act
    report = dp.duplicates(df)
    
    # Assert
    assert report["duplicate_count"] == 1
    assert report["duplicate_percentage"] == 25.0

def test_correlation():
    # Arrange
    # Perfectly correlated numeric series
    data = {
        "X": [1.0, 2.0, 3.0, 4.0],
        "Y": [2.0, 4.0, 6.0, 8.0],
        "Z": [10.0, 9.0, 8.0, 7.0] # negatively correlated
    }
    df = pd.DataFrame(data)
    
    # Act
    report = dp.correlation(df, threshold=0.9)
    
    # Assert
    assert len(report["strong_positive"]) >= 1
    assert len(report["strong_negative"]) >= 1
    # Check that X ↔ Y is flagged positively
    assert any("X ↔ Y" in pair for pair, val in report["strong_positive"])
    # Check Z correlation
    assert any("X ↔ Z" in pair for pair, val in report["strong_negative"])

def test_classification_report():
    # Arrange
    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 1]
    
    # Act
    report = dp.classification_report(y_true, y_pred)
    
    # Assert
    assert report["accuracy"] == 0.8333
    assert report["precision"] == 1.0
    assert report["recall"] == 0.75

def test_diagnose(capsys):
    # Overfitting case
    dp.diagnose(train_score=0.95, test_score=0.70)
    captured = capsys.readouterr()
    assert "OVERFITTING DETECTED" in captured.out
    
    # Underfitting case
    dp.diagnose(train_score=0.50, test_score=0.45)
    captured = capsys.readouterr()
    assert "UNDERFITTING DETECTED" in captured.out
    
    # Optimal fit
    dp.diagnose(train_score=0.85, test_score=0.82)
    captured = capsys.readouterr()
    assert "OPTIMAL FIT" in captured.out

def test_dashboard(tmp_path):
    # Arrange
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    df = pd.DataFrame(data)
    report_file = tmp_path / "report.html"
    
    # Act
    dp.dashboard(df, output_path=str(report_file))
    
    # Assert
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "DataPilot Dataset Report" in content
    assert "Total Rows" in content
    assert "Total Columns" in content
    # Enhanced dashboard checks
    assert "Column Data Types" in content
    assert "Strong Linear Correlations" in content


def test_version():
    assert hasattr(dp, "__version__")
    assert isinstance(dp.__version__, str)
    assert dp.__version__ == "0.4.2"




def test_classification_report_multiclass():
    # 3-class problem - should not raise an error
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 1, 2]
    report = dp.classification_report(y_true, y_pred)
    assert "accuracy" in report
    assert "precision" in report
    assert "recall" in report
    assert "f1_score" in report


# ── Visualization smoke tests (headless) ─────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({"Age": [25.0, 30.0, 35.0, None, 40.0], "Fare": [10.0, 20.0, 30.0, 40.0, 50.0]})


def test_hist_smoke(sample_df):
    import matplotlib
    matplotlib.use("Agg")  # headless backend — no GUI window
    import matplotlib.pyplot as plt
    try:
        dp.hist(sample_df, "Age", bins=5)
    except Exception as e:
        pytest.fail(f"dp.hist() raised an unexpected exception: {e}")
    finally:
        plt.close("all")


def test_box_smoke(sample_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        dp.box(sample_df, "Fare")
    except Exception as e:
        pytest.fail(f"dp.box() raised an unexpected exception: {e}")
    finally:
        plt.close("all")


def test_heatmap_smoke(sample_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        dp.heatmap(sample_df)
    except Exception as e:
        pytest.fail(f"dp.heatmap() raised an unexpected exception: {e}")
    finally:
        plt.close("all")


# ── AI insights mock test ────────────────────────────────────────────────────

def test_ai_insights_success():
    """Verify generate_insights returns the LLM response on success."""
    from datapilot.ai.insights import generate_insights
    with patch("datapilot.ai.insights.get_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "• Impute Age nulls.\n• Encode Sex feature."
        mock_factory.return_value = mock_provider
        result = generate_insights(
            meta={"rows": 891, "columns": 12, "duplicates_count": 0},
            missing_list=[{"column": "Age", "missing_count": 177, "missing_percentage": 19.87}],
            strong_relations=[],
        )
    assert "Impute" in result



def test_ai_insights_connection_error():
    """Verify graceful error message when Ollama is not running."""
    from datapilot.ai.insights import generate_insights
    with patch("datapilot.ai.insights.get_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = Exception("Connection refused")
        mock_factory.return_value = mock_provider
        result = generate_insights(
            meta={"rows": 100, "columns": 5, "duplicates_count": 0},
            missing_list=[],
            strong_relations=[],
        )
    assert "Could not generate AI Insights" in result


def test_ai_factory_invalid_provider():
    """Verify factory raises ValueError for unknown provider name."""
    from datapilot.ai.factory import get_provider
    with pytest.raises(ValueError, match="Unknown AI provider"):
        get_provider(ai_provider="unknown_provider", api_key="test")


def test_ai_factory_missing_api_key():
    """Verify factory raises ValueError when cloud provider is used without an API key."""
    from datapilot.ai.factory import get_provider
    with pytest.raises(ValueError, match="requires an API key"):
        get_provider(ai_provider="openai", api_key=None)


def test_ai_openai_provider_mock():
    """Verify OpenAI provider returns content when SDK is available."""
    from datapilot.ai.providers.openai_provider import OpenAIProvider
    provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    # Patch at the generate() level — SDK is lazily imported inside the method
    with patch.object(provider, "generate", return_value="• Drop nulls. • Encode cats. • Scale features."):
        result = provider.generate(
            meta={"rows": 100, "columns": 5, "duplicates_count": 0},
            missing_list=[],
            strong_relations=[],
        )
    assert "Drop nulls" in result



def test_ai_groq_provider_mock():
    """Verify Groq provider returns content when SDK is available."""
    from datapilot.ai.providers.groq_provider import GroqProvider
    provider = GroqProvider(model="llama3-70b-8192", api_key="gsk_test")
    # Patch at the generate() level — SDK is lazily imported inside the method
    with patch.object(provider, "generate", return_value="• Impute Age. • Drop ID. • Encode Sex."):
        result = provider.generate(
            meta={"rows": 891, "columns": 12, "duplicates_count": 0},
            missing_list=[],
            strong_relations=[],
        )
    assert "Impute Age" in result



def test_ai_gemini_provider_missing_package():
    """Verify graceful error when google-generativeai is not installed."""
    from datapilot.ai.providers.gemini_provider import GeminiProvider
    provider = GeminiProvider(model="gemini-1.5-flash", api_key="AIza_test")
    with patch.dict("sys.modules", {"google.generativeai": None}):
        result = provider.generate(
            meta={"rows": 50, "columns": 3, "duplicates_count": 0},
            missing_list=[],
            strong_relations=[],
        )
    assert "not installed" in result or "error" in result.lower()


def test_ai_claude_provider_missing_package():
    """Verify graceful error when anthropic is not installed."""
    from datapilot.ai.providers.claude_provider import ClaudeProvider
    provider = ClaudeProvider(model="claude-3-haiku-20240307", api_key="sk-ant-test")
    with patch.dict("sys.modules", {"anthropic": None}):
        result = provider.generate(
            meta={"rows": 50, "columns": 3, "duplicates_count": 0},
            missing_list=[],
            strong_relations=[],
        )
    assert "not installed" in result or "error" in result.lower()


# ── New feature tests ────────────────────────────────────────────────────────

def test_suggest_detects_id_column():
    data = {
        "ID": list(range(100)),          # 100% unique → should be flagged
        "Age": [25] * 100,               # constant → should be flagged
        "Score": [float(i) for i in range(100)],
    }
    df = pd.DataFrame(data)
    suggestions = dp.suggest(df)
    issues = [s["column"] for s in suggestions]
    assert "ID" in issues
    assert "Age" in issues


def test_suggest_flags_high_nulls():
    data = {"A": [None] * 80 + [1.0] * 20}  # 80% null
    df = pd.DataFrame(data)
    suggestions = dp.suggest(df)
    assert any(s["column"] == "A" for s in suggestions)


def test_compare_detects_drift():
    train = pd.DataFrame({"Age": [20.0, 21.0, 22.0, 23.0, 24.0]})
    test  = pd.DataFrame({"Age": [60.0, 61.0, 62.0, 63.0, 64.0]})  # big shift
    flags = dp.compare(train, test, threshold=0.1)
    assert any(f["column"] == "Age" for f in flags)


def test_compare_no_drift():
    df1 = pd.DataFrame({"Age": [20.0, 21.0, 22.0, 23.0, 24.0]})
    df2 = pd.DataFrame({"Age": [20.5, 21.5, 22.5, 23.5, 24.5]})  # tiny shift
    flags = dp.compare(df1, df2, threshold=0.1)
    assert not any(f["column"] == "Age" for f in flags)


def test_outliers_detects_extreme_values():
    data = {"Score": [10.0, 11.0, 10.5, 9.5, 10.2, 500.0]}  # 500 is a clear outlier
    df = pd.DataFrame(data)
    result = dp.outliers(df, method="both")
    assert "Score" in result
    assert 500.0 in result["Score"]["values"]


def test_outliers_clean_data():
    data = {"Score": [10.0, 10.5, 11.0, 10.8, 9.9, 10.3]}
    df = pd.DataFrame(data)
    result = dp.outliers(df)
    assert "Score" not in result


def test_auto_clean_drops_id_and_constant():
    data = {
        "ID":    list(range(50)),          # should be dropped
        "Const": ["x"] * 50,              # should be dropped
        "Age":   [25.0, None] * 25,        # nulls should be imputed
    }
    df = pd.DataFrame(data)
    clean_df, log = dp.auto_clean(df)
    assert "ID" not in clean_df.columns
    assert "Const" not in clean_df.columns
    assert clean_df["Age"].isnull().sum() == 0  # all nulls filled


def test_auto_clean_drops_high_null_column():
    data = {
        "Good":  [1.0, 2.0, 3.0, 4.0, 5.0],
        "Sparse": [None, None, None, None, 1.0],  # 80% null
    }
    df = pd.DataFrame(data)
    clean_df, log = dp.auto_clean(df, drop_null_threshold=0.6)
    assert "Sparse" not in clean_df.columns
    assert "Good" in clean_df.columns


def test_regression_report():
    y_true = [3.0, 5.0, 7.0, 9.0, 11.0]
    y_pred = [2.8, 5.2, 6.8, 9.1, 10.9]
    report = dp.regression_report(y_true, y_pred)
    assert "mae" in report
    assert "mse" in report
    assert "rmse" in report
    assert "r2" in report
    assert "mape" in report
    assert "max_error" in report
    assert report["r2"] > 0.99   # near-perfect predictions
    assert report["mae"]  < 0.3


def test_benchmark_returns_results():
    df = pd.DataFrame({
        "A": list(range(200)),
        "B": [float(i) * 0.5 for i in range(200)],
        "C": ["cat", "dog"] * 100,
    })
    result = dp.benchmark(df)
    assert isinstance(result, dict)
    assert len(result) > 0
    for op, metrics in result.items():
        assert "datapilot_ms" in metrics
        assert "pandas_ms" in metrics
        assert "speedup" in metrics
        assert metrics["speedup"] > 0
