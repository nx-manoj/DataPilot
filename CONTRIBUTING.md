# Contributing to DataPilot

Thank you for considering contributing to DataPilot! This document outlines the process for contributing code, tests, documentation, and bug reports.

---

## Getting Started

### 1. Fork and Clone the Repository

```bash
git clone https://github.com/yourusername/datapilot.git
cd datapilot
```

### 2. Set Up the Development Environment

We use `uv` for fast virtual environment management:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

### 3. Verify Tests Pass

```bash
pytest
```

---

## Contribution Guidelines

### Code Style
- Follow **PEP 8** for Python code formatting.
- Use **type annotations** on all public function signatures.
- Write **docstrings** for every public function.
- Keep functions focused and single-responsibility.

### Testing
- All new features **must** include unit tests in `tests/`.
- All tests must pass before submitting a pull request.
- Run tests with coverage: `pytest --cov=datapilot`.

### Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
```
feat: add regression diagnostics to dp.diagnose()
fix: resolve plt.show() missing in histogram.py
docs: update installation instructions in README
test: add multi-class classification coverage
```

---

## Pull Request Process

1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Write code and tests.
3. Run `pytest` and confirm all tests pass.
4. Push your branch and open a Pull Request against `main`.
5. The PR description should describe **what** changed and **why**.

---

## Reporting Bugs

Please open a GitHub Issue with:
- A **minimal reproduction script**.
- Your Python version and OS.
- The full traceback.

---

## Feature Requests

Open a GitHub Issue tagged `enhancement` with a description of the feature and why it would benefit DataPilot users.
