# Changelog

All notable changes to DataPilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
