"""Data Quality Engine: Pure computational audits for tabular datasets."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from datasleuth.config import CONFIG


@dataclass
class ColumnQuality:
    """Detailed audit metrics for a single DataFrame column."""
    name: str
    dtype: str
    semantic_type: str  # 'numerical', 'categorical', 'boolean', 'datetime', 'identifier'
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    # Numerical statistics
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max_val: float | None = None
    skewness: float | None = None
    outlier_count: int = 0
    outlier_pct: float = 0.0
    # Categorical statistics
    mode: str | None = None
    mode_freq: int | None = None
    # Warnings & heuristic flags
    is_high_cardinality: bool = False
    is_id_like: bool = False
    warnings: list[str] | None = None


@dataclass
class QualityDiagnostic:
    """Dataset-level finding, categorized by severity and nature."""
    kind: str  # 'fact' (confirmed metric) or 'heuristic' (rule-of-thumb warning)
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    affected_columns: list[str]


@dataclass
class DatasetQualityReport:
    """Comprehensive dataset quality audit containing column & dataset diagnostics."""
    row_count: int
    column_count: int
    total_cells: int
    total_missing_cells: int
    overall_missing_pct: float
    duplicate_rows_count: int
    duplicate_rows_pct: float
    constant_columns: list[str]
    empty_columns: list[str]
    column_metrics: list[ColumnQuality]
    diagnostics: list[QualityDiagnostic]
    health_score: int  # 0 to 100
    target_info: dict[str, Any] | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert column metrics into a formatted Pandas DataFrame for display or export."""
        records = []
        for col in self.column_metrics:
            rec = {
                "Column": col.name,
                "Type": col.semantic_type.capitalize(),
                "Dtype": col.dtype,
                "Missing Count": col.missing_count,
                "Missing %": round(col.missing_pct, 2),
                "Unique Values": col.unique_count,
                "Unique Ratio": round(col.unique_ratio, 3),
                "Constant": "Yes" if col.is_constant else "No",
                "Outliers (IQR)": col.outlier_count if col.semantic_type == "numerical" else "-",
                "Outlier %": f"{col.outlier_pct:.1f}%" if col.semantic_type == "numerical" else "-",
                "Mean": round(col.mean, 2) if col.mean is not None else "-",
                "Median": round(col.median, 2) if col.median is not None else "-",
                "Std Dev": round(col.std, 2) if col.std is not None else "-",
                "Min": round(col.min_val, 2) if col.min_val is not None else "-",
                "Max": round(col.max_val, 2) if col.max_val is not None else "-",
                "Skewness": round(col.skewness, 2) if col.skewness is not None else "-",
                "Top / Mode": str(col.mode) if col.mode is not None else "-",
                "Warnings": "; ".join(col.warnings) if col.warnings else "None",
            }
            records.append(rec)
        return pd.DataFrame(records)

    def to_dict(self) -> dict[str, Any]:
        """Convert entire report into serializable dictionary."""
        return {
            "summary": {
                "row_count": self.row_count,
                "column_count": self.column_count,
                "total_cells": self.total_cells,
                "total_missing_cells": self.total_missing_cells,
                "overall_missing_pct": self.overall_missing_pct,
                "duplicate_rows_count": self.duplicate_rows_count,
                "duplicate_rows_pct": self.duplicate_rows_pct,
                "constant_columns": self.constant_columns,
                "empty_columns": self.empty_columns,
                "health_score": self.health_score,
            },
            "columns": [asdict(c) for c in self.column_metrics],
            "diagnostics": [asdict(d) for d in self.diagnostics],
            "target_info": self.target_info,
        }


def detect_outliers_iqr(series: pd.Series, multiplier: float = CONFIG.IQR_MULTIPLIER) -> tuple[int, float]:
    """Detect outliers using Tukey's Interquartile Range (IQR) method.
    
    A value is flagged as a potential outlier if:
      value < Q1 - multiplier * IQR  OR  value > Q3 + multiplier * IQR
    Returns:
      (outlier_count, outlier_percentage)
    """
    clean_series = series.dropna()
    if len(clean_series) < 4:
        return 0, 0.0

    q25 = clean_series.quantile(0.25)
    q75 = clean_series.quantile(0.75)
    iqr = q75 - q25

    if iqr == 0:
        return 0, 0.0

    lower_bound = q25 - multiplier * iqr
    upper_bound = q75 + multiplier * iqr
    is_outlier = (clean_series < lower_bound) | (clean_series > upper_bound)
    count = int(is_outlier.sum())
    pct = (count / len(clean_series)) * 100.0 if len(clean_series) > 0 else 0.0
    return count, pct


def infer_semantic_type(col_name: str, series: pd.Series) -> str:
    """Infer the practical statistical role of a column."""
    name_lower = col_name.lower()
    
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
        
    n_unique = series.nunique(dropna=True)
    n_total = len(series.dropna())
    
    # Check for ID-like column
    if n_unique > 0 and n_unique == n_total and n_total > 50:
        if any(term in name_lower for term in ["id", "uuid", "key", "index", "code", "guid"]):
            return "identifier"

    if pd.api.types.is_numeric_dtype(series):
        # Treat low-cardinality integers with <= 2 values as boolean-like
        if n_unique == 2 and set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return "boolean"
        return "numerical"

    return "categorical"


def calculate_dataset_health_score(
    missing_pct: float,
    duplicate_pct: float,
    constant_col_count: int,
    total_cols: int,
    avg_outlier_pct: float,
    has_severe_imbalance: bool = False,
) -> int:
    """Calculate an objective dataset health score from 0 to 100 based on standard defect penalties."""
    score = 100.0

    # 1. Missing data penalty: up to -35 points
    if missing_pct > 0:
        score -= min(35.0, missing_pct * 1.5)

    # 2. Duplicate rows penalty: up to -20 points
    if duplicate_pct > 0:
        score -= min(20.0, duplicate_pct * 2.0)

    # 3. Constant/useless columns penalty: up to -15 points
    if total_cols > 0 and constant_col_count > 0:
        const_ratio = constant_col_count / total_cols
        score -= min(15.0, const_ratio * 30.0)

    # 4. Outlier severity penalty: up to -15 points
    if avg_outlier_pct > 5.0:
        score -= min(15.0, (avg_outlier_pct - 5.0) * 1.0)

    # 5. Class imbalance penalty: -10 points if severe
    if has_severe_imbalance:
        score -= 10.0

    return max(0, min(100, int(round(score))))


def audit_column(col_name: str, series: pd.Series, n_rows: int) -> ColumnQuality:
    """Compute exhaustive quality metrics and heuristic flags for a single column."""
    missing_count = int(series.isna().sum())
    missing_pct = (missing_count / n_rows * 100.0) if n_rows > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = (unique_count / n_rows) if n_rows > 0 else 0.0
    is_constant = (unique_count <= 1) and (n_rows > 1)
    
    semantic_type = infer_semantic_type(col_name, series)
    is_id_like = (semantic_type == "identifier")
    warnings: list[str] = []

    # Missingness warnings
    if missing_pct >= CONFIG.SEVERE_MISSING_WARN_PCT:
        warnings.append(f"Severe missingness ({missing_pct:.1f}% missing)")
    elif missing_pct >= CONFIG.HIGH_MISSING_WARN_PCT:
        warnings.append(f"Moderate missingness ({missing_pct:.1f}% missing)")

    if is_constant:
        warnings.append("Constant feature (zero variance, offers no predictive signal)")

    col_quality = ColumnQuality(
        name=col_name,
        dtype=str(series.dtype),
        semantic_type=semantic_type,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        is_constant=is_constant,
        is_id_like=is_id_like,
    )

    clean_series = series.dropna()

    if pd.api.types.is_numeric_dtype(series) and semantic_type == "numerical":
        if len(clean_series) > 0:
            col_quality.mean = float(clean_series.mean())
            col_quality.std = float(clean_series.std(ddof=1)) if len(clean_series) > 1 else 0.0
            col_quality.min_val = float(clean_series.min())
            col_quality.q25 = float(clean_series.quantile(0.25))
            col_quality.median = float(clean_series.median())
            col_quality.q75 = float(clean_series.quantile(0.75))
            col_quality.max_val = float(clean_series.max())
            
            # Skewness
            if len(clean_series) > 2 and col_quality.std > 0:
                skew_val = float(stats.skew(clean_series))
                col_quality.skewness = skew_val
                if abs(skew_val) > 1.5:
                    warnings.append(f"High skewness ({skew_val:+.2f}); consider power or log transform")

            # Outlier detection via IQR
            out_cnt, out_pct = detect_outliers_iqr(clean_series)
            col_quality.outlier_count = out_cnt
            col_quality.outlier_pct = out_pct
            if out_pct > 8.0:
                warnings.append(f"Substantial outliers ({out_pct:.1f}% outside 1.5×IQR)")

    elif semantic_type in ["categorical", "boolean"]:
        if len(clean_series) > 0:
            mode_val = clean_series.mode()
            if not mode_val.empty:
                col_quality.mode = str(mode_val.iloc[0])
                col_quality.mode_freq = int((clean_series == mode_val.iloc[0]).sum())
                
        # High cardinality check
        if unique_count > CONFIG.HIGH_CARDINALITY_MAX_UNIQUE and unique_ratio > CONFIG.HIGH_CARDINALITY_RATIO_WARN:
            col_quality.is_high_cardinality = True
            warnings.append(f"High cardinality ({unique_count} distinct categories); risk of sparse encoding")

    if is_id_like:
        warnings.append("Appears to be a unique identifier; exclude from ML model training to avoid data leakage")

    col_quality.warnings = warnings
    return col_quality


def detect_data_leakage_risks(df: pd.DataFrame, target_col: str | None = None) -> list[QualityDiagnostic]:
    """Detect potential data leakage risks such as IDs or features perfectly collinear with the target."""
    diagnostics: list[QualityDiagnostic] = []
    
    # 1. Row-level unique IDs
    id_cols = [c for c in df.columns if infer_semantic_type(c, df[c]) == "identifier"]
    if id_cols:
        diagnostics.append(QualityDiagnostic(
            kind="heuristic",
            severity="warning",
            title="Potential Identifier Leakage Detected",
            message=(
                f"Column(s) {id_cols} have near 100% uniqueness and match ID nomenclature. "
                "Training models on row identifiers causes artificial overfitting and false performance."
            ),
            affected_columns=id_cols,
        ))

    # 2. Target correlation leakage
    if target_col and target_col in df.columns:
        target_series = df[target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col
            ]
            leaky_features = []
            for col in numeric_cols:
                valid_mask = df[[col, target_col]].dropna()
                if len(valid_mask) > 10 and valid_mask[col].std() > 0 and valid_mask[target_col].std() > 0:
                    corr = valid_mask[col].corr(valid_mask[target_col])
                    if pd.notna(corr) and abs(corr) >= CONFIG.LEAKAGE_CORRELATION_THRESHOLD:
                        leaky_features.append((col, corr))

            if leaky_features:
                leaky_names = [f"{col} (r={corr:.3f})" for col, corr in leaky_features]
                diagnostics.append(QualityDiagnostic(
                    kind="heuristic",
                    severity="critical",
                    title="Suspicious Target Correlation (Possible Data Leakage)",
                    message=(
                        f"The following feature(s) correlate almost perfectly (|r| >= {CONFIG.LEAKAGE_CORRELATION_THRESHOLD}) "
                        f"with the target '{target_col}': {', '.join(leaky_names)}. "
                        "Verify whether these features are recorded post-event."
                    ),
                    affected_columns=[col for col, _ in leaky_features],
                ))

    return diagnostics


def audit_dataset_quality(df: pd.DataFrame, target_col: str | None = None) -> DatasetQualityReport:
    """Run an exhaustive data quality audit on the entire dataset."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    overall_missing_pct = (total_missing / total_cells * 100.0) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (duplicate_rows / n_rows * 100.0) if n_rows > 0 else 0.0

    # Column audits
    column_metrics: list[ColumnQuality] = [
        audit_column(col, df[col], n_rows) for col in df.columns
    ]

    constant_cols = [c.name for c in column_metrics if c.is_constant]
    empty_cols = [c.name for c in column_metrics if c.missing_count == n_rows]

    # Outlier percentage average across numeric columns
    numeric_outliers = [
        c.outlier_pct for c in column_metrics if c.semantic_type == "numerical"
    ]
    avg_outlier_pct = float(np.mean(numeric_outliers)) if numeric_outliers else 0.0

    diagnostics: list[QualityDiagnostic] = []

    # Fact: Duplicate rows
    if duplicate_rows > 0:
        diagnostics.append(QualityDiagnostic(
            kind="fact",
            severity="warning" if duplicate_pct > 5.0 else "info",
            title=f"Duplicate Rows ({duplicate_rows:,} found)",
            message=f"{duplicate_rows:,} rows ({duplicate_pct:.2f}%) are exact duplicates across all columns.",
            affected_columns=[],
        ))
    else:
        diagnostics.append(QualityDiagnostic(
            kind="fact",
            severity="info",
            title="No Duplicate Rows",
            message="Every row in the dataset is distinct.",
            affected_columns=[],
        ))

    # Fact: Missing values
    if total_missing > 0:
        sev = "critical" if overall_missing_pct > CONFIG.HIGH_MISSING_WARN_PCT else "warning"
        diagnostics.append(QualityDiagnostic(
            kind="fact",
            severity=sev,
            title=f"Missing Values ({overall_missing_pct:.2f}% overall)",
            message=f"{total_missing:,} missing cells out of {total_cells:,} total data cells.",
            affected_columns=[c.name for c in column_metrics if c.missing_count > 0],
        ))
    else:
        diagnostics.append(QualityDiagnostic(
            kind="fact",
            severity="info",
            title="Complete Dataset (0% Missing)",
            message="There are zero missing cells in any column.",
            affected_columns=[],
        ))

    # Constant columns
    if constant_cols:
        diagnostics.append(QualityDiagnostic(
            kind="fact",
            severity="warning",
            title=f"Constant Columns ({len(constant_cols)} found)",
            message=f"Columns {constant_cols} have only 1 unique value across all rows. They provide zero discriminatory signal.",
            affected_columns=constant_cols,
        ))

    # Target-specific diagnostics
    target_info: dict[str, Any] | None = None
    has_severe_imbalance = False

    if target_col and target_col in df.columns:
        target_series = df[target_col].dropna()
        n_unique_target = target_series.nunique()
        val_counts = target_series.value_counts(normalize=True).to_dict()
        raw_counts = target_series.value_counts().to_dict()

        is_classification = (
            not pd.api.types.is_numeric_dtype(target_series)
            or n_unique_target <= 20
        )

        target_info = {
            "target_column": target_col,
            "inferred_task": "classification" if is_classification else "regression",
            "unique_values_count": n_unique_target,
            "distribution": {str(k): round(v * 100.0, 2) for k, v in val_counts.items()},
            "raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
        }

        if is_classification and n_unique_target > 1:
            min_class_pct = min(val_counts.values()) * 100.0
            max_class_pct = max(val_counts.values()) * 100.0
            imbalance_ratio = max_class_pct / (min_class_pct if min_class_pct > 0 else 1.0)
            target_info["min_class_pct"] = round(min_class_pct, 2)
            target_info["imbalance_ratio"] = round(imbalance_ratio, 2)

            if min_class_pct < CONFIG.CLASS_IMBALANCE_MINORITY_PCT or imbalance_ratio >= CONFIG.CLASS_IMBALANCE_RATIO_WARN:
                has_severe_imbalance = True
                diagnostics.append(QualityDiagnostic(
                    kind="heuristic",
                    severity="warning",
                    title="Class Imbalance Detected in Target",
                    message=(
                        f"Target '{target_col}' has a minority class representing only {min_class_pct:.1f}% "
                        f"(Imbalance ratio: {imbalance_ratio:.1f}:1). Baseline accuracy may be misleading; "
                        "evaluate using F1-score and balanced metrics."
                    ),
                    affected_columns=[target_col],
                ))

        # Check for data leakage
        leakage_diags = detect_data_leakage_risks(df, target_col)
        diagnostics.extend(leakage_diags)
    else:
        # Check general ID column warnings
        leakage_diags = detect_data_leakage_risks(df, None)
        diagnostics.extend(leakage_diags)

    # Health score computation
    health_score = calculate_dataset_health_score(
        missing_pct=overall_missing_pct,
        duplicate_pct=duplicate_pct,
        constant_col_count=len(constant_cols),
        total_cols=n_cols,
        avg_outlier_pct=avg_outlier_pct,
        has_severe_imbalance=has_severe_imbalance,
    )

    return DatasetQualityReport(
        row_count=n_rows,
        column_count=n_cols,
        total_cells=total_cells,
        total_missing_cells=total_missing,
        overall_missing_pct=overall_missing_pct,
        duplicate_rows_count=duplicate_rows,
        duplicate_rows_pct=duplicate_pct,
        constant_columns=constant_cols,
        empty_columns=empty_cols,
        column_metrics=column_metrics,
        diagnostics=diagnostics,
        health_score=health_score,
        target_info=target_info,
    )
