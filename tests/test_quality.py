"""Unit tests for the Data Quality Engine."""

import numpy as np
import pandas as pd

from datasleuth.quality import (
    audit_column,
    audit_dataset_quality,
    calculate_dataset_health_score,
    detect_data_leakage_risks,
    detect_outliers_iqr,
    infer_semantic_type,
)


def test_detect_outliers_iqr():
    # Normal data with obvious outliers
    data = pd.Series([10, 11, 12, 10, 11, 12, 10, 11, 100, -50])
    count, pct = detect_outliers_iqr(data, multiplier=1.5)
    assert count == 2
    assert pct == 20.0


def test_infer_semantic_type():
    df = pd.DataFrame({
        "id_col": [f"user_{i}" for i in range(100)],
        "num_col": list(range(100)),
        "bool_col": [True, False] * 50,
        "cat_col": ["red", "blue", "green", "red"] * 25,
    })
    assert infer_semantic_type("id_col", df["id_col"]) == "identifier"
    assert infer_semantic_type("num_col", df["num_col"]) == "numerical"
    assert infer_semantic_type("bool_col", df["bool_col"]) == "boolean"
    assert infer_semantic_type("cat_col", df["cat_col"]) == "categorical"


def test_audit_column_numerical():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    col_q = audit_column("num", series, n_rows=6)
    assert col_q.missing_count == 1
    assert round(col_q.missing_pct, 1) == 16.7
    assert col_q.mean == 3.0
    assert col_q.min_val == 1.0
    assert col_q.max_val == 5.0
    assert col_q.semantic_type == "numerical"


def test_audit_dataset_quality_basic():
    df = pd.DataFrame({
        "customer_id": [f"ID_{i}" for i in range(100)],
        "age": [25, 30, 35, 40] * 25,
        "constant_feature": [1] * 100,
        "target": [0, 1] * 50,
    })
    report = audit_dataset_quality(df, target_col="target")
    assert report.row_count == 100
    assert report.column_count == 4
    assert report.duplicate_rows_count == 0
    assert "constant_feature" in report.constant_columns
    assert report.health_score > 0
    assert report.target_info["inferred_task"] == "classification"


def test_data_leakage_detection():
    # Perfectly correlated feature with target
    n = 100
    target = np.random.randint(0, 2, n)
    leaky = target * 10.0  # perfect correlation r=1.0
    df = pd.DataFrame({
        "uid": [f"U_{i}" for i in range(n)],
        "feature_ok": np.random.randn(n),
        "leaky_feature": leaky,
        "target": target,
    })
    diags = detect_data_leakage_risks(df, target_col="target")
    titles = [d.title for d in diags]
    assert any("Potential Identifier Leakage Detected" in t for t in titles)
    assert any("Suspicious Target Correlation" in t for t in titles)


def test_health_score_penalties():
    perfect_score = calculate_dataset_health_score(0, 0, 0, 10, 0, False)
    assert perfect_score == 100

    flawed_score = calculate_dataset_health_score(
        missing_pct=30.0,
        duplicate_pct=15.0,
        constant_col_count=2,
        total_cols=10,
        avg_outlier_pct=10.0,
        has_severe_imbalance=True,
    )
    assert flawed_score < 50
