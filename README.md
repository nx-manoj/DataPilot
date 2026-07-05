# 📖 DataPilot — Your Data Science Copilot

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.0-brightgreen.svg)](https://github.com/nx-manoj/DataPilot)
[![Core Engine](https://img.shields.io/badge/Powered%20By-Polars%20%7C%20Apache%20Arrow-orange.svg)](https://pola.rs/)
[![AI Providers](https://img.shields.io/badge/AI-Ollama%20%7C%20OpenAI%20%7C%20Gemini%20%7C%20Claude%20%7C%20Groq-purple.svg)](https://github.com/nx-manoj/DataPilot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-30%20passed-success.svg)](https://github.com/nx-manoj/DataPilot)

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
| Offline HTML dashboard | ✅ | ✅ | ❌ | ✅ |
| Regression diagnostics | ❌ | ❌ | ❌ | ✅ |
| ML model diagnostics | ❌ | ❌ | ❌ | ✅ |

---

## 📑 Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Module 1: Auto EDA Pipeline](#module-1-auto-eda-pipeline)
4. [Module 2: Smart Column Suggestions](#module-2-smart-column-suggestions)
5. [Module 3: Dataset Analysis API](#module-3-dataset-analysis-api)
6. [Module 4: Outlier Detection](#module-4-outlier-detection)
7. [Module 5: Auto Data Cleaning](#module-5-auto-data-cleaning)
8. [Module 6: Train/Test Drift Detection](#module-6-traintest-drift-detection)
9. [Module 7: Visualization Engine](#module-7-visualization-engine)
10. [Module 8: Machine Learning Diagnostics](#module-8-machine-learning-diagnostics)
11. [Module 9: Standalone HTML Dashboard](#module-9-standalone-html-dashboard)
12. [Module 10: Performance Benchmark](#module-10-performance-benchmark)
13. [Module 11: AI Copilot (5 Providers)](#module-11-ai-copilot--5-providers)
14. [Installation & Setup](#installation--setup)
15. [Troubleshooting](#troubleshooting)

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

> **System requirement:** Python 3.9+. Ollama is only required if you use `use_ai=True`.

---

## Quick Start

```python
import pandas as pd
import datapilot as dp

df = pd.read_csv("your_dataset.csv")

# Full automated EDA in one line
dp.analyze(df)

# Smart suggestions on what to fix
dp.suggest(df)

# Auto-clean the dataset
clean_df, log = dp.auto_clean(df)

# Export a full HTML report
dp.dashboard(df, "report.html")
```

---

## Module 1: Auto EDA Pipeline

### `dp.analyze(df, use_ai=False, ai_model="llama3")`

The primary entry point. Runs all structural checks simultaneously and prints a clean console report.

```python
# Standard rule-based analysis
dp.analyze(df)

# With local AI copilot (requires Ollama)
dp.analyze(df, use_ai=True, ai_model="llama3")
```

**Console Output:**
```
==================================================
         DATAPILOT AUTOMATED EDA REPORT         
==================================================
📊 Dataset Shape: 891 rows × 12 columns
💾 Memory Usage:  0.0815 MB
👥 Duplicate Rows: 0 (0.0%)
--------------------------------------------------

🔍 Missing Value Profile:
   • Cabin: 687 missing (77.1%)
   • Age: 177 missing (19.87%)

🔗 Strong Linear Correlations (|r| >= 0.6):
   • Fare ↔ Survived (pos: 0.697)
==================================================
```

---

## Module 2: Smart Column Suggestions

### `dp.suggest(df)`

Analyses every column and returns actionable, rule-based preprocessing recommendations — no AI or internet required.

```python
suggestions = dp.suggest(df)
```

**Detects:**
- 🔴 ID/key columns (100% unique values — no signal, should be dropped)
- 🔴 Constant columns (zero variance — useless for ML)
- 🔴 Columns with >60% missing data
- 🟡 Columns with 20–60% missing data
- 🟡 Unencoded categorical strings (should be One-Hot or Target encoded)
- 🟡 High-cardinality strings (need embedding or target encoding)
- 🟡 Date strings that should be parsed to datetime
- 🟡 Imbalanced binary targets (class ratio < 20%)

**Console Output:**
```
========================================================
       DATAPILOT SMART COLUMN SUGGESTIONS        
========================================================

[1] 🔴 HIGH  —  PassengerId
     Issue:      100% unique values — likely an ID or key column
     Fix:        Drop 'PassengerId' — unique identifiers leak no signal.

[2] 🟡 MEDIUM  —  Cabin
     Issue:      Very high missing rate (77.1%)
     Fix:        Consider dropping 'Cabin' — over 60% nulls will degrade model quality.

[3] 🟡 MEDIUM  —  Sex
     Issue:      Unencoded categorical string (2 unique values)
     Fix:        Encode 'Sex' using One-Hot Encoding before modelling.
========================================================
```

**Returns:** `List[Dict]` with keys `column`, `issue`, `severity`, `suggestion`.

---

## Module 3: Dataset Analysis API

All functions accept both `pandas.DataFrame` and `polars.DataFrame`.

### `dp.summary(df)`

```python
meta = dp.summary(df)
# Returns: {'engine_detected': 'pandas', 'rows': 891, 'columns': 12,
#           'datatypes': {...}, 'memory_usage_mb': 0.08, 'total_missing_values': 866}
```

### `dp.missing(df)`

```python
report = dp.missing(df)
```

| column | missing_count | missing_percentage |
|--------|:---:|:---:|
| Cabin | 687 | 77.10% |
| Age | 177 | 19.87% |

### `dp.duplicates(df)`

```python
dups = dp.duplicates(df)
# Returns: {'duplicate_count': 5, 'duplicate_percentage': 0.56}
```

### `dp.correlation(df, threshold=0.6)`

```python
corr = dp.correlation(df, threshold=0.6)
print(corr['strong_positive'])  # [('Fare ↔ Survived', 0.697)]
print(corr['strong_negative'])  # [('Pclass ↔ Fare', -0.549)]
```

---

## Module 4: Outlier Detection

### `dp.outliers(df, method="both", z_threshold=3.0, iqr_multiplier=1.5)`

Detects outliers across all numeric columns using **IQR fencing** and/or **Z-score** — whichever flags more values when `method="both"`.

```python
result = dp.outliers(df)
result = dp.outliers(df, method="iqr")          # IQR only
result = dp.outliers(df, method="zscore")        # Z-score only
result = dp.outliers(df, iqr_multiplier=3.0)     # extreme outliers only
```

**Console Output:**
```
========================================================
        DATAPILOT OUTLIER DETECTION REPORT        
  Method: BOTH | Z-threshold: 3.0 | IQR×1.5
========================================================

  🔴 HIGH  —  Fare
     Outlier count:  12 (6.7% of non-null values)
     IQR fences:     [-26.18, 65.63]
     Extreme values: [512.3, 263.0, 211.3]
```

**Returns:** `Dict[column → {count, percentage, severity, values, lower_fence, upper_fence}]`

---

## Module 5: Auto Data Cleaning

### `dp.auto_clean(df, drop_null_threshold=0.6, impute_strategy="auto", drop_id_columns=True, drop_constant_columns=True)`

The killer feature. Automatically fixes your dataset and tells you exactly what it changed.

```python
clean_df, change_log = dp.auto_clean(df)
```

**What it does (in order):**
1. Drops columns where **all values are the same** (zero variance)
2. Drops **ID/key columns** where every row is unique
3. Drops columns with **≥60% null values**
4. **Imputes** remaining numeric nulls with **median**
5. **Imputes** remaining categorical nulls with **mode**

**Console Output:**
```
==========================================================
         DATAPILOT AUTO-CLEAN ENGINE              
==========================================================
  Input:  891 rows × 12 columns

  🗑️  DROP   'PassengerId'  →  ID-like column (all 891 values unique)
  🗑️  DROP   'Cabin'        →  High null rate (77.1%) exceeds threshold (60%)
  🔧  IMPUTE 'Age'          →  filled 177 null(s) with median=28.0
  🔧  IMPUTE 'Embarked'     →  filled 2 null(s) with mode='S'

----------------------------------------------------------
  Output: 891 rows × 10 columns
  ✅ 2 column(s) dropped  |  2 column(s) imputed
==========================================================
```

**Returns:** `Tuple[cleaned_DataFrame, List[change_log_dicts]]`

---

## Module 6: Train/Test Drift Detection

### `dp.compare(df_train, df_test, threshold=0.1)`

Detects distribution shift between training and production data — the root cause of silent model degradation.

```python
flags = dp.compare(df_train, df_test)
flags = dp.compare(df_train, df_test, threshold=0.15)  # stricter
```

**How it works:**
- **Numeric columns:** Flags if mean shifts >15% or std shifts >25%
- **Categorical columns:** Computes Jensen-Shannon divergence (0 = identical, 1 = completely different)

**Console Output:**
```
============================================================
       DATAPILOT TRAIN vs TEST DRIFT REPORT       
  Train shape: 712 rows × 11 cols
  Test  shape: 179 rows × 11 cols
  Shared columns analysed: 11
------------------------------------------------------------

  ⚠️  Age  [🟡 MEDIUM]  (numeric drift)
      Mean:  train=29.64  →  test=30.27  (2.1% shift)
      Std:   train=14.52  →  test=15.11

  ⚠️  Embarked  [🔴 HIGH]  (categorical drift)
      JS Divergence: 0.341 (threshold: 0.1)
      Train top value: 'S'  |  Test top value: 'C'
```

**Returns:** `List[Dict]` of flagged columns with drift statistics.

---

## Module 7: Visualization Engine

One-liner plots, no matplotlib syntax required.

```python
dp.hist(df, "Age", bins=15)      # distribution with KDE overlay
dp.box(df, "Fare")               # quartiles and outliers
dp.heatmap(df)                   # annotated Pearson correlation grid
```

---

## Module 8: Machine Learning Diagnostics

### `dp.classification_report(y_true, y_pred, average="auto")`

Auto-detects binary vs multi-class and picks the right averaging strategy.

```python
metrics = dp.classification_report(y_test, predictions)
# Returns: {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85}
```

### `dp.regression_report(y_true, y_pred)`

```python
metrics = dp.regression_report(y_test, predictions)
# Returns: {'mae': 2.14, 'mse': 7.32, 'rmse': 2.71, 'r2': 0.94, 'mape': 4.8, 'max_error': 8.3}
```

**Console Output:**
```
==================================================
     DATAPILOT REGRESSION METRICS REPORT     
==================================================
  MAE        (Mean Absolute Error)   : 2.1400
  MSE        (Mean Squared Error)    : 7.3200
  RMSE       (Root Mean Sq. Error)   : 2.7055
  R²         (Coefficient of Det.)   : 0.9400
  MAPE       (Mean Abs. % Error)     : 4.80%
  Max Error  (Worst single pred.)    : 8.3000
--------------------------------------------------
  Verdict: ✅ EXCELLENT — model explains ≥90% of variance.
==================================================
```

### `dp.diagnose(train_score, test_score, metric_name="Accuracy")`

```python
dp.diagnose(train_score=0.98, test_score=0.72)
# → 🚨 OVERFITTING DETECTED: Add regularization, gather more data.

dp.diagnose(train_score=0.50, test_score=0.48)
# → ⚠️ UNDERFITTING DETECTED: Increase model complexity.
```

---

## Module 9: Standalone HTML Dashboard

### `dp.dashboard(df, output_path="datapilot_report.html")`

Generates a complete, **offline-ready** HTML report with:
- Summary metric cards (rows, columns, memory, duplicates, missing values)
- Data type overview table
- Missing values table + inline bar chart
- Pearson correlation pairs table
- Inline correlation heatmap image (base64 encoded — no external dependencies)

```python
dp.dashboard(df, output_path="my_project_report.html")
# → 🎉 Standalone HTML Dashboard successfully exported to: my_project_report.html
```

> ✅ **No internet required.** No backend server. Just open the `.html` file in any browser.

---

## Module 10: Performance Benchmark

### `dp.benchmark(df)`

Proves DataPilot's Polars-native speed advantage against equivalent Pandas operations.

```python
dp.benchmark(df)
```

**Console Output:**
```
==================================================================
          DATAPILOT BENCHMARK  —  Polars vs Pandas          
  Dataset: 891 rows × 12 columns
==================================================================
  Operation                   DataPilot       Pandas    Speedup
------------------------------------------------------------------
  Null Count                     0.09ms      0.48ms      5.3× ▓▓▓▓▓
  Duplicate Detection            0.11ms      0.91ms      8.3× ▓▓▓▓▓▓▓▓
  Describe / Summary Stats       0.22ms      1.47ms      6.7× ▓▓▓▓▓▓
  Correlation Matrix             0.31ms      2.18ms      7.0× ▓▓▓▓▓▓▓
  Group-By Mean                  0.08ms      0.54ms      6.8× ▓▓▓▓▓▓
------------------------------------------------------------------
  Average Speedup                                         6.8×

  ⚡ DataPilot is on average 6.8× faster than Pandas on this dataset.
  📈 Speedup grows significantly on larger datasets (1M+ rows).
==================================================================
```

---

## Module 11: AI Copilot — 5 Providers

DataPilot uses a **Metadata-Only AI Pattern** — raw data rows are **never** transmitted to any provider. Only lightweight statistical summaries (e.g. `"Age has 19.8% nulls"`) are sent.

### Supported Providers

| Provider | Type | Default Model | Requires |
|----------|------|--------------|----------|
| `ollama` | 🔒 Local / Private | `llama3` | Ollama daemon running locally |
| `openai` | ☁️ Cloud | `gpt-4o-mini` | `pip install datapilot-polars[openai]` + API key |
| `gemini` | ☁️ Cloud | `gemini-1.5-flash` | `pip install datapilot-polars[gemini]` + API key |
| `claude` | ☁️ Cloud | `claude-3-haiku-20240307` | `pip install datapilot-polars[claude]` + API key |
| `groq`   | ☁️ Cloud (free tier) | `llama3-70b-8192` | `pip install datapilot-polars[groq]` + API key |


### Usage

```python
# 🔒 Local — fully private, no API key needed (default)
dp.analyze(df, use_ai=True)
dp.analyze(df, use_ai=True, ai_provider="ollama", ai_model="mistral")

# ☁️ OpenAI — GPT-4o, GPT-4, GPT-3.5
dp.analyze(df, use_ai=True, ai_provider="openai", api_key="sk-...")
dp.analyze(df, use_ai=True, ai_provider="openai", ai_model="gpt-4o", api_key="sk-...")

# ☁️ Google Gemini — Gemini 1.5 Pro / Flash
dp.analyze(df, use_ai=True, ai_provider="gemini", api_key="AIza...")
dp.analyze(df, use_ai=True, ai_provider="gemini", ai_model="gemini-1.5-pro", api_key="AIza...")

# ☁️ Anthropic Claude — Claude 3.5 Sonnet / Haiku
dp.analyze(df, use_ai=True, ai_provider="claude", api_key="sk-ant-...")
dp.analyze(df, use_ai=True, ai_provider="claude", ai_model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")

# ☁️ Groq — Ultra-fast free-tier inference (Llama3-70b, Mixtral, Gemma2)
dp.analyze(df, use_ai=True, ai_provider="groq", api_key="gsk_...")
dp.analyze(df, use_ai=True, ai_provider="groq", ai_model="mixtral-8x7b-32768", api_key="gsk_...")
```

### Install Only What You Need

```bash
pip install datapilot-polars[openai]    # OpenAI only
pip install datapilot-polars[gemini]    # Google Gemini only
pip install datapilot-polars[claude]    # Anthropic Claude only
pip install datapilot-polars[groq]      # Groq only (free tier)
pip install datapilot-polars[all-ai]    # All cloud providers at once
```

### AI Output Example

```
🤖 AI Copilot Insights  [GROQ]:
• The Age column has 19.8% missing values — impute with median before
  training to avoid biased estimates in tree models.
• Sex and Embarked are unencoded strings — apply One-Hot Encoding;
  Sex has only 2 values making it ideal for binary encoding.
• The strong positive correlation between Fare and Survived (r=0.697)
  suggests Fare is a strong predictor — include it and check for
  outliers before normalising.
```

> 💡 **Groq is recommended for getting started** — it has a generous free tier,
> runs Llama3-70b at extremely low latency, and requires just a free account at
> [console.groq.com](https://console.groq.com).

---

## Installation & Setup

### Basic Setup

```bash
git clone https://github.com/nx-manoj/DataPilot.git
cd DataPilot

uv venv
source .venv/bin/activate
uv pip install -e .
```

### With Cloud AI Providers

```bash
uv pip install -e .[openai]     # Add OpenAI support
uv pip install -e .[groq]       # Add Groq support (free)
uv pip install -e .[all-ai]     # Add all cloud AI providers
```

### Development Setup (with tests)

```bash
uv pip install -e .[dev]
pytest
```

### Initializing Local AI (Ollama)

Only needed if you use `ai_provider="ollama"` (the default):

```bash
# Start the Ollama daemon
ollama serve

# Pull a model (first time only)
ollama pull llama3
```

---

## Troubleshooting

### `ModuleNotFoundError` after editing files
```bash
uv pip install -e . --force-reinstall
```

### AI: Ollama Connection Refused
```bash
ollama serve   # make sure the Ollama daemon is running
```

### AI: Missing cloud provider package
```bash
# Install the extra for your chosen provider
pip install datapilot-polars[openai]   # or gemini / claude / groq / all-ai
```

### AI: Invalid or missing API key
```python
# Always pass the key explicitly for cloud providers
dp.analyze(df, use_ai=True, ai_provider="openai", api_key="sk-...")
```

### Running Tests
```bash
pytest -v
pytest --cov=datapilot   # with coverage report
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style guidelines, and the pull request process.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full version history.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
