# 📖 DataPilot — Your Data Science Copilot

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.5.0-brightgreen.svg)](https://github.com/nx-manoj/DataPilot)
[![Core Engine](https://img.shields.io/badge/Powered%20By-Polars%20%7C%20Apache%20Arrow-orange.svg)](https://pola.rs/)
[![AI Providers](https://img.shields.io/badge/AI-Ollama%20%7C%20OpenAI%20%7C%20Gemini%20%7C%20Claude%20%7C%20Groq-purple.svg)](https://github.com/nx-manoj/DataPilot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passed-success.svg)](https://github.com/nx-manoj/DataPilot)

> **The only EDA library that is Polars-native, multi-provider AI-powered, and actually cleans your data.**

DataPilot is an open-source Python library that automates Exploratory Data Analysis, detects data quality issues, cleans datasets, benchmarks performance, and generates intelligent AI recommendations — via Ollama (local), OpenAI, Google Gemini, Anthropic Claude, or Groq — all with minimal code.

---

## 🚀 Why DataPilot?

| | ydata-profiling | sweetviz | dtale | **DataPilot** |
|---|:-:|:-:|:-:|:-:|
| Polars-native (10x faster) | ❌ | ❌ | ❌ | ✅ |
| Local AI (Ollama, private) | ❌ | ❌ | ❌ | ✅ |
| Cloud AI (OpenAI/Gemini/Claude/Groq) | ❌ | ❌ | ❌ | ✅ |
| Auto data cleaning | ❌ | ❌ | ❌ | ✅ |
| Train/test drift detection | ✅ | ✅ | ❌ | ✅ |
| Outlier detection | ⚠️ | ❌ | ✅ | ✅ |
| Smart column suggestions | ❌ | ❌ | ❌ | ✅ |
| Interactive Glassmorphic HTML dashboard | ✅ | ✅ | ❌ | ✅ |
| Regression diagnostics | ❌ | ❌ | ❌ | ✅ |
| ML model diagnostics | ❌ | ❌ | ❌ | ✅ |

---

## 📑 Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Module 1: Global Session Configuration](#module-1-global-session-configuration)
4. [Module 2: Auto EDA Pipeline](#module-2-auto-eda-pipeline)
5. [Module 3: Smart Column Suggestions](#module-3-smart-column-suggestions)
6. [Module 4: Dataset Analysis API](#module-4-dataset-analysis-api)
7. [Module 5: Outlier Detection](#module-5-outlier-detection)
8. [Module 6: Auto Data Cleaning](#module-6-auto-data-cleaning)
9. [Module 7: Train/Test Drift Detection](#module-7-traintest-drift-detection)
10. [Module 8: Conversational AI (Ask AI)](#module-8-conversational-ai-ask-ai)
11. [Module 9: Visualization Engine](#module-9-visualization-engine)
12. [Module 10: Machine Learning Diagnostics](#module-10-machine-learning-diagnostics)
13. [Module 11: Standalone HTML Dashboard](#module-11-standalone-html-dashboard)
13. [Module 12: Performance Benchmark](#module-12-performance-benchmark)
14. [Module 13: DataPilot Web Studio](#module-13-datapilot-web-studio)
15. [Module 14: AI Copilot Providers](#module-14-ai-copilot-providers)
16. [Troubleshooting](#troubleshooting)

---

## Installation

### From PyPI (Recommended)

To install the stable release:
```bash
pip install datapilot-polars
```

To install with optional cloud AI provider dependencies:
```bash
pip install "datapilot-polars[openai]"    # OpenAI support
pip install "datapilot-polars[gemini]"    # Google Gemini support
pip install "datapilot-polars[claude]"    # Anthropic Claude support
pip install "datapilot-polars[groq]"      # Groq support (free tier)
pip install "datapilot-polars[studio]"    # DataPilot Web Studio UI
pip install "datapilot-polars[all-ai]"    # All cloud AI providers at once
```

### From Source (Development)

```bash
# Clone and install in editable mode
git clone https://github.com/nx-manoj/DataPilot.git
cd DataPilot

uv venv && source .venv/bin/activate
uv pip install -e .

# For development / running tests
uv pip install -e .[dev]
```

---

## Quick Start

```python
import pandas as pd
import datapilot as dp

# 1. Configure AI provider once (Optional)
dp.configure(ai_provider="groq", api_key="gsk_...")

df = pd.read_csv("your_dataset.csv")

# 2. Full automated EDA with AI insights
dp.analyze(df, use_ai=True)

# 3. Get preprocessing suggestions with AI recommendations
suggestions = dp.suggest(df, use_ai=True)

# 4. Ask the AI conversational questions about the data
dp.ask_ai(df, "What are the most important preprocessing steps for this dataset?")

# 5. Generate plots with natural language prompts
dp.visualize_ai(df, "Show the relation between Age and Survived")

# 6. Auto-clean the dataset
clean_df, log = dp.auto_clean(df, use_ai=True)

# 7. Export a full offline HTML report
dp.dashboard(df, "report.html")

# 8. Launch the interactive DataPilot Web Studio in your browser
dp.launch_studio()
```

---

## Module 1: Global Session Configuration

### `dp.configure(ai_provider="ollama", ai_model=None, api_key=None)`

Set your credentials once at the beginning of your session. Subsequent calls containing `use_ai=True` will automatically fetch these credentials.

```python
# Configure cloud AI (e.g., Groq)
dp.configure(ai_provider="groq", api_key="gsk_...")

# Subsequent calls don't need credentials repeated
dp.analyze(df, use_ai=True)
dp.suggest(df, use_ai=True)
```

---

## Module 2: Auto EDA Pipeline

### `dp.analyze(df, use_ai=False, ai_provider=None, ai_model=None, api_key=None)`

Runs all structural checks simultaneously and prints a clean console report.

```python
# Standard rule-based analysis
dp.analyze(df)

# With configured AI copilot
dp.analyze(df, use_ai=True)
```

---

## Module 3: Smart Column Suggestions

### `dp.suggest(df, use_ai=False, ai_provider=None, ai_model=None, api_key=None)`

Analyses every column and returns actionable, rule-based preprocessing recommendations. Enables optional `use_ai=True` to append expert AI comments.

```python
suggestions = dp.suggest(df, use_ai=True)
```

---

## Module 4: Dataset Analysis API

### `dp.summary(df)`
Returns a high-level overview dict:
```python
meta = dp.summary(df)
# {'rows': 891, 'columns': 12, 'memory_usage_mb': 0.08, 'engine_detected': 'pandas', ...}
```

### `dp.missing(df)`
Returns sorted DataFrame showing column null counts and percentages.

### `dp.duplicates(df)`
Checks for exact duplicate rows across all CPU cores.

### `dp.correlation(df, threshold=0.6)`
Calculates the Pearson correlation matrix for all numeric columns, flagging strong pairs.

---

## Module 5: Outlier Detection

### `dp.outliers(df, method="both", z_threshold=3.0, iqr_multiplier=1.5, use_ai=False, ...)`

Detects outliers across numeric columns using **IQR fencing** and/or **Z-score**. Set `use_ai=True` to receive AI recommendations on how to handle them.

```python
result = dp.outliers(df, use_ai=True)
```

---

## Module 6: Auto Data Cleaning

### `dp.auto_clean(df, drop_null_threshold=0.6, impute_strategy="auto", encode_categoricals=None, scale_numerics=None, drop_id_columns=True, drop_constant_columns=True, use_ai=False, ...)`

Automatically cleans the dataset and logs changes. Enabling `use_ai=True` appends a conversational explanation of why the actions improve model quality. 

Advanced ML preprocessing is now supported directly within this function:
- `impute_strategy`: Uses median/mode by default, or `"knn"` to use `sklearn.impute.KNNImputer` for numerical columns.
- `encode_categoricals`: Pass `"onehot"` to get dummy variables, or `"label"` for integer encoding of categorical columns.
- `scale_numerics`: Pass `"standard"` (Z-score) or `"minmax"` to scale numerical features, essential for distance-based ML models.

```python
clean_df, change_log = dp.auto_clean(df, impute_strategy="knn", encode_categoricals="onehot", scale_numerics="standard", use_ai=True)
```

---

## Module 7: Train/Test Drift Detection

### `dp.compare(df_train, df_test, threshold=0.1, use_ai=False, ...)`

Detects distribution shift between training and test datasets. Uses Jensen-Shannon divergence for categoricals. Enabling `use_ai=True` yields AI-suggested mitigation strategies.

```python
flags = dp.compare(df_train, df_test, use_ai=True)
```

---

## Module 8: Conversational AI (Ask AI)

### `dp.ask_ai(df, question, ai_provider=None, ai_model=None, api_key=None)`

Ask free-form natural-language questions about your dataset. Only statistical metadata is transmitted to the AI — never raw rows.

```python
dp.ask_ai(df, "Which features carry the most risk of data leakage?")
dp.ask_ai(df, "Should I log-transform Fare or normalise Age first?")
```

---

## Module 9: Visualization Engine

Includes publication-ready, interactive Plotly charts with a dark theme (`plotly_dark`) and automatic statistical overlays. 

**Note:** All manual charting functions return a Plotly `Figure` object. In Jupyter Notebooks, simply returning the figure displays it. In standard Python scripts, you must call `.show()` on the returned figure to view it.

### `dp.hist(df, column, bins="auto", hue=None, color="#3b82f6")`
Interactive histogram with automatic KDE overlay, plus mean and median dashed lines.
```python
fig = dp.hist(df, "Age")
fig.show()  # Required in standard scripts
```

### `dp.box(df, column, group_by=None, orient="v")`
Interactive box plot with median highlights and automatic IQR annotation.

### `dp.heatmap(df)`
Interactive Pearson correlation matrix heatmap with hover tooltips.

### `dp.scatter(df, x, y, hue=None, trendline=True)`
Interactive scatter plot with optional OLS regression trendline.

### `dp.violin(df, column, group_by=None)`
Interactive violin plot combining box plot and KDE for rich distribution insights.

### `dp.visualize_ai(df, prompt, ai_provider=None, ai_model=None, api_key=None)`
Ask the AI to choose and draw the right interactive chart from a plain-English prompt.
```python
dp.visualize_ai(df, "Show the relation between Age and Survived")
dp.visualize_ai(df, "Distribution of Fare for each passenger class")
dp.visualize_ai(df, "Correlation heatmap of numeric columns")
```

---

## Module 10: Machine Learning Diagnostics

### `dp.classification_report(y_true, y_pred, average="auto")`
Calculates binary or multi-class metrics safely.

### `dp.regression_report(y_true, y_pred)`
Calculates MAE, MSE, RMSE, R², MAPE, and Max Error.

### `dp.diagnose(train_score, test_score, metric_name="Accuracy")`
Evaluates train/test performance gaps to diagnose overfitting or underfitting.

---

## Module 11: Standalone HTML Dashboard

### `dp.dashboard(df, output_path="datapilot_report.html")`
Generates a complete, **offline-ready** HTML dashboard report containing metrics, datatype profiles, missing value charts, and correlation heatmap matrix. Features a premium glassmorphic dark-mode UI with embedded interactive Plotly charts.

---

## Module 12: Performance Benchmark

### `dp.benchmark(df)`
Benchmarks DataPilot (Polars core) operations against equivalent Pandas operations.

---

## Module 13: DataPilot Web Studio

### `dp.launch_studio(port=8501)`
Launches a completely interactive, no-code local Streamlit application in your web browser. 
Users can drag-and-drop CSV files, view the glassmorphic dashboard in real-time, and chat with their data natively.

*Requires the `studio` extra: `pip install datapilot-polars[studio]`*

---

## Module 14: AI Copilot Providers

DataPilot uses a **Metadata-Only AI Pattern** — raw data rows are **never** transmitted. Only statistical summaries are sent.

| Provider | Type | Default Model | Requires |
|----------|------|--------------|----------|
| `ollama` | 🔒 Local / Private | `llama3` | Ollama daemon running locally |
| `openai` | ☁️ Cloud | `gpt-4o-mini` | `pip install datapilot-polars[openai]` + API key |
| `gemini` | ☁️ Cloud | `gemini-1.5-flash` | `pip install datapilot-polars[gemini]` + API key |
| `claude` | ☁️ Cloud | `claude-3-haiku-20240307` | `pip install datapilot-polars[claude]` + API key |
| `groq`   | ☁️ Cloud (free tier) | `llama3-70b-8192` | `pip install datapilot-polars[groq]` + API key |

---

## Troubleshooting

### `ModuleNotFoundError` after editing files
```bash
uv pip install -e . --force-reinstall
```

### AI: Ollama Connection Refused
Ensure the local daemon is active:
```bash
ollama serve
```

### Running Tests
```bash
pytest -v
pytest --cov=datapilot
```
