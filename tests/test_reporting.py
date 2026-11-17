"""Unit tests for reporting and export generators."""

import json

import pandas as pd

from datasleuth.modeling import run_benchmark_suite
from datasleuth.quality import audit_dataset_quality
from datasleuth.reporting import generate_html_report, generate_json_report


def test_reporting_html_and_json():
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    qr = audit_dataset_quality(df, target_col="target")
    suite = run_benchmark_suite(df, "target", ["feature1", "feature2"], "classification", 0.2, 42)

    # 1. Test HTML Report Generation
    html_output = generate_html_report("Test Dataset", df, qr, suite)
    assert "<!DOCTYPE html>" in html_output
    assert "DataSleuth Audit Report" in html_output
    assert "Test Dataset" in html_output
    assert "Machine Learning Benchmark Leaderboard" in html_output
    assert str(qr.health_score) in html_output

    # 2. Test JSON Report Generation
    json_output = generate_json_report(qr, suite)
    parsed = json.loads(json_output)
    assert parsed["platform"] == "DataSleuth"
    assert "quality_audit" in parsed
    assert "ml_benchmarks" in parsed
    assert parsed["ml_benchmarks"]["best_model"] == suite.best_model_name


def test_overview_feature_distribution_chart():
    from datasleuth.eda import plot_overview_feature_distribution
    df = pd.DataFrame({
        "age": [25, 32, 45, 52, 60, 38, 42, 29, 65, 71],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    })
    fig_num = plot_overview_feature_distribution(df, "age")
    assert fig_num is not None
    assert len(fig_num.data) > 0
    fig_cat = plot_overview_feature_distribution(df, "sex")
    assert fig_cat is not None
    assert len(fig_cat.data) > 0

