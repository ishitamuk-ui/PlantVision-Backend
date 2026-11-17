# DataSleuth: Automated Dataset Auditing & ML Benchmarking Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg)](https://plotly.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](https://pytest.org/)

**DataSleuth** is a production-minded tabular dataset auditing and machine learning benchmarking platform. Built with Python, Streamlit, Pandas, Scikit-learn, and Plotly, DataSleuth helps data scientists, machine learning engineers, and researchers rigorously inspect dataset quality, diagnose structural defects and potential data leakage, explore statistical distributions interactively, and train leak-free baseline models before committing to deep modeling cycles.

All computations are 100% genuine and executed in-memory: **no fake metrics, no placeholder dashboards, and no paid API dependencies.**

---

## Key Capabilities

### 1. Robust Dataset Ingestion & Validation
- **Multi-Format CSV Ingestion:** Automatic sniffing for comma, semicolon, tab, and pipe delimiters with fallback encoding resolution (UTF-8, Latin-1, CP1252).
- **Security & Integrity Checks:** File size limit enforcement, empty-dataset detection, malformed line handling, and automatic column deduplication.
- **Built-in Benchmark Datasets:** Includes three offline, genuine datasets for instant exploration:
  - *Heart Disease Clinical Risk* (Binary classification with mixed numerical/categorical attributes and class imbalance).
  - *California Real Estate Pricing* (Continuous regression with right-skewed numerical features and outliers).
  - *Telco Customer Churn* (Mixed-type classification with contractual and behavioral features).

### 2. Comprehensive Data Quality Engine
- **Decoupled Computational Engine:** Fully testable pure-Python logic separated from Streamlit presentation.
- **Exhaustive Column Auditing:** Inferred semantic types (numerical, categorical, boolean, datetime, identifier), missingness counts & percentages, distinct ratios, and constant column indicators.
- **Parametric Summary Statistics:** Mean, standard deviation, median, 25th/75th percentiles, min/max range, and Fisher-Pearson skewness coefficient.
- **Tukey's IQR Outlier Detection:** Rigorously detects extreme values using the interquartile range ($[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$), clearly distinguishing empirical facts from heuristic warnings.
- **Automated Data Leakage Detection:** Detects row-level identifier features (e.g., surrogate keys, UUIDs) and features exhibiting near-perfect collinearity ($|r| \ge 0.98$) with the target variable.
- **Target Imbalance Diagnostics:** Evaluates class distribution ratios, highlighting severe minority class under-representation ($< 15\%$).
- **Dataset Health Score (0–100):** Transparent, deterministic scoring based on missingness penalties, duplicate rows, constant columns, outlier severity, and class balance.

### 3. Interactive Exploratory Data Analysis (EDA)
- **Fluid Plotly Visualizations:** Styled with a clean, editorial layout and restrained color palette.
- **Feature Distributions:** Interactive histograms with box plot marginals for numerical features, and frequency distributions for categorical features.
- **Missingness Pattern Analysis:** Feature-by-feature missingness bar plots with severity color-coding.
- **Numerical Correlation Heatmap:** Configurable Pearson and Spearman correlation matrices with threshold filtering.
- **Automated Feature vs. Target Exploration:** Dynamically renders grouped box plots, scatter plots with OLS regression trendlines, or stacked percentage bars based on feature and target types.
- **Bivariate Scatter Explorer:** Multi-axis scatter exploration with hue grouping, sizing dimensions, and transparent sampling notices for datasets exceeding 5,000 points.

### 4. Machine Learning Model Lab
- **Task Inference & Target Selection:** Automatically infers Classification vs. Regression with manual user override.
- **Leak-Free Preprocessing Pipelines:** Built with Scikit-learn's `ColumnTransformer` and `Pipeline`:
  - Numerical: `SimpleImputer(strategy="median")` + `StandardScaler()`
  - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
  - *Preprocessors are fitted strictly on training partitions to guarantee zero test-set leakage.*
- **Baseline Model Suite:**
  - *Classification:* `DummyClassifier` (Stratified), `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`.
  - *Regression:* `DummyRegressor` (Mean), `RidgeRegression`, `RandomForestRegressor`.
- **Comprehensive Evaluation:**
  - Classification: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), ROC-AUC (One-vs-Rest), and interactive Confusion Matrix heatmap.
  - Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), $R^2$ Score, and Residual diagnostics.
  - Optional K-Fold / Stratified K-Fold Cross-Validation.
- **Best Model Deep-Dive:** Feature importances (MDI) or linear model coefficients, along with practical constraints and actionable takeaways.

### 5. Publication-Ready Audit Reports & Exports
- **Standalone HTML Report:** Self-contained, responsive HTML file with inline CSS styling (no CDN or external dependencies required) containing executive summary, health scores, quality tables, and benchmark leaderboards.
- **Data Quality Table (CSV):** Complete export of all column-level audit metrics and warnings.
- **Audit Metadata (JSON):** Machine-readable JSON export for downstream CI/CD or data pipelines.

---

## Architecture & Directory Layout

```
├── app.py                      # Streamlit application entrypoint & session state orchestrator
├── datasleuth/
│   ├── __init__.py
│   ├── config.py               # Constants, thresholds, theme color definitions
│   ├── ingestion.py            # Ingestion, validation, delimiter sniffing, sample datasets
│   ├── quality.py              # Pure computational Data Quality Engine
│   ├── eda.py                  # Interactive Plotly chart generators
│   ├── modeling.py             # Preprocessing pipelines, baselines, evaluation, explainability
│   ├── reporting.py            # Standalone HTML and JSON report generators
│   └── ui/
│       ├── __init__.py
│       ├── styles.py           # Editorial design system and modern CSS styling
│       ├── components.py       # Reusable UI widgets (cards, badges, metric displays)
│       ├── view_overview.py    # Overview tab (ingestion, preview, schema, health summary)
│       ├── view_quality.py     # Data Quality tab (column inventory, diagnostic findings)
│       ├── view_eda.py         # Exploratory Data Analysis tab (interactive Plotly charts)
│       ├── view_model_lab.py   # Machine Learning Model Lab tab (benchmarks & leaderboard)
│       └── view_reports.py     # Reports tab (downloadable HTML, CSV, JSON)
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py       # Validation, size limits, duplicate columns, sample datasets
│   ├── test_quality.py         # Column audits, IQR outliers, data leakage, health score
│   ├── test_modeling.py        # ML pipelines, classification & regression baselines, metrics
│   └── test_reporting.py      # HTML and JSON report validation
├── pyproject.toml              # Ruff and Pytest configuration
├── requirements.txt            # Streamlit Community Cloud production dependencies
└── README.md
```

---

## Quickstart & Local Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ishitamuk-ui/PlantVision-Backend.git
   cd PlantVision-Backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the DataSleuth platform:**
   ```bash
   streamlit run app.py
   ```
   The application will open automatically at `http://localhost:8501`.

---

## Running the Automated Test Suite & Linter

### Run Unit Tests (`pytest`)
```bash
pytest
```
*Executes all 24 automated unit tests covering ingestion, quality auditing, pipeline execution, metric computation, and report rendering.*

### Run Linter (`ruff`)
```bash
ruff check .
```
*Ensures compliance with PEP 8 and clean formatting.*

---

## Deployment to Streamlit Community Cloud

DataSleuth is designed for zero-configuration deployment to [Streamlit Community Cloud](https://share.streamlit.io):

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**.
4. Select your repository and branch (`main`).
5. Set the Main file path to `app.py`.
6. Click **Deploy!**

No environment variables, API keys, or external databases are required.

---

## License

This project is licensed under the MIT License.
