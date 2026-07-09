# Changelog

All notable changes to DataPilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2026-07-09

### Added
- **DataPilot Web Studio:** A new interactive Streamlit web application. Call `dp.launch_studio()` to launch a local browser app where you can drag-and-drop CSVs, view the dashboard in real-time, and chat with your AI copilot.
- **Interactive UI Upgrade:** Fully migrated all static visualizations (Matplotlib/Seaborn) to interactive Plotly charts (`px.histogram`, `px.scatter`, `px.box`, `px.violin`, `px.imshow`). Charts now support zooming, panning, and hover-tooltips.
- **Premium Dashboard Aesthetics:** Upgraded `dp.dashboard()` HTML export to a modern, SaaS-like dark mode. Features glassmorphism (backdrop-filter blurs), modern typography (Inter font), responsive CSS grid, and hover animations for a stunning presentation.
- `dp.visualize_ai()` now generates interactive Plotly figures based on natural language prompts instead of Matplotlib axes.
- `plotly` is now a core dependency, and `streamlit` is available as an optional `[studio]` dependency in `pyproject.toml`.

---

## [0.4.3] - 2026-07-06

### Fixed
- `dp.analyze()` now correctly reads AI provider, model, and API key from `dp.configure()` instead of ignoring the session config (broke `dp.configure(...); dp.analyze(df, use_ai=True)` workflows).
- `dp.ask_ai()` response was displayed twice in Jupyter notebooks — once via `print()` and once as the cell's return value. Fixed with a `_SilentStr` return wrapper whose `__repr__` suppresses the auto-display.
- `dp.correlation()` now uses pairwise drop-nulls instead of Polars' global `.corr()`, which propagated `NaN` across the entire row/column for any column with missing values (e.g. `Age` in Titanic). All valid pairs now return real correlation values.
- Strong correlation pairs (`strong_positive` / `strong_negative`) were always empty when any numeric column had nulls — fixed as a direct result of the above pairwise approach.

---

## [0.2.0] - 2026-07-05

### Added
- `dp.suggest(df)` — Rule-based smart column analyser. Detects ID columns, constant columns, high-null columns, unencoded categoricals, date strings, and imbalanced binary targets.
- `dp.compare(df_train, df_test)` — Train/test distribution drift detector. Uses mean/std shift for numeric columns and Jensen-Shannon divergence for categorical columns.
- `dp.outliers(df)` — Dual IQR + Z-score outlier detection across all numeric columns with severity ratings.
- `dp.auto_clean(df)` — One-click automated data cleaning. Drops constant, ID-like, and high-null columns; imputes remaining nulls with median/mode. Returns cleaned DataFrame + full change log.
- `dp.regression_report(y_true, y_pred)` — Comprehensive regression metrics: MAE, MSE, RMSE, R², MAPE, Max Error, with automatic verdict.
- `dp.benchmark(df)` — Side-by-side speed comparison of DataPilot (Polars) vs Pandas across 5 operations.
- `__version__` attribute bumped to `0.2.0`.
- Sub-package `__init__.py` files added to `ml/`, `visualization/`, `dashboard/`, `ai/` for proper package structure.

---

## [0.1.0] - 2026-07-05

### Added
- `dp.summary(df)` — High-speed structural overview of datasets via Polars engine.
- `dp.missing(df)` — Null value profiling, sorted by highest missing percentage.
- `dp.duplicates(df)` — Multi-threaded duplicate row detection via Polars hashing.
- `dp.correlation(df, threshold)` — Pearson correlation matrix with significant pair extraction.
- `dp.analyze(df)` — Full automated EDA pipeline with console reporting.
- `dp.hist(df, column)` — KDE-overlay histogram for numerical distributions.
- `dp.box(df, column)` — Box plot for distribution quartiles and outlier visualization.
- `dp.heatmap(df)` — Annotated correlation heatmap for numerical columns.
- `dp.classification_report(y_true, y_pred)` — Safe, multi-class classification metric computation.
- `dp.diagnose(train_score, test_score)` — Automatic ML overfitting/underfitting diagnostic tool.
- `dp.dashboard(df)` — Standalone HTML dashboard export with inline charts and full dataset profiling.
- Local AI copilot integration via `use_ai=True` powered by Ollama (metadata-only, privacy-preserving).
- Zero-copy Pandas ↔ Polars conversion via PyArrow for maximum performance.

### Fixed
- `dp.hist()` was silently generating plots without calling `plt.show()`, making it invisible in scripts.
- `dp.classification_report()` now supports multi-class targets (auto-detects binary vs. multi-class).

### Changed
- `matplotlib`, `seaborn`, and `scikit-learn` are now declared as core package dependencies in `pyproject.toml`.
- Visualization imports are now lazy (loaded on first call) to speed up `import datapilot`.
