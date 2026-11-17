"""Unit tests for the Machine Learning Model Lab."""

import numpy as np
import pandas as pd

from datasleuth.modeling import (
    build_preprocessing_pipeline,
    get_candidate_models,
    run_benchmark_suite,
    suggest_task_type,
)


def test_suggest_task_type():
    df_class = pd.DataFrame({"target": ["cat", "dog", "bird"] * 10})
    df_binary = pd.DataFrame({"target": [0, 1] * 20})
    df_reg = pd.DataFrame({"target": np.linspace(0, 100, 100)})

    assert suggest_task_type(df_class, "target") == "classification"
    assert suggest_task_type(df_binary, "target") == "classification"
    assert suggest_task_type(df_reg, "target") == "regression"


def test_build_preprocessing_pipeline():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "cat1": ["a", "b", "a", np.nan],
    })
    prep, num_cols, cat_cols = build_preprocessing_pipeline(df, ["num1", "cat1"])
    assert num_cols == ["num1"]
    assert cat_cols == ["cat1"]
    
    transformed = prep.fit_transform(df)
    assert transformed.shape[0] == 4
    assert not np.isnan(transformed).any()


def test_candidate_models_classification():
    models = get_candidate_models("classification", random_seed=42)
    assert "Dummy (Stratified)" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models


def test_candidate_models_regression():
    models = get_candidate_models("regression", random_seed=42)
    assert "Dummy (Mean)" in models
    assert "Ridge Regression" in models
    assert "Random Forest" in models


def test_run_benchmark_suite_classification():
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame({
        "feat_num": rng.randn(n),
        "feat_cat": rng.choice(["A", "B", "C"], size=n),
        "target": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
    })

    suite = run_benchmark_suite(
        df=df,
        target_col="target",
        feature_cols=["feat_num", "feat_cat"],
        task="classification",
        test_size=0.25,
        random_seed=42,
        cv_folds=2,
    )

    assert suite.task == "classification"
    assert suite.train_rows == 90
    assert suite.test_rows == 30
    assert len(suite.results) >= 3
    assert suite.best_model_name is not None
    assert suite.primary_metric == "F1-Score (Weighted)"

    lb_df = suite.get_leaderboard_df()
    assert "Model" in lb_df.columns
    assert "F1-Score (Weighted)" in lb_df.columns
    assert "Accuracy" in lb_df.columns


def test_run_benchmark_suite_regression():
    rng = np.random.RandomState(42)
    n = 100
    x = rng.randn(n)
    y = 3.5 * x + rng.randn(n) * 0.5
    df = pd.DataFrame({
        "feat_x": x,
        "feat_cat": rng.choice(["low", "high"], size=n),
        "target": y,
    })

    suite = run_benchmark_suite(
        df=df,
        target_col="target",
        feature_cols=["feat_x", "feat_cat"],
        task="regression",
        test_size=0.20,
        random_seed=42,
        cv_folds=0,
    )

    assert suite.task == "regression"
    assert suite.train_rows == 80
    assert suite.test_rows == 20
    assert suite.primary_metric == "RMSE"
    
    lb_df = suite.get_leaderboard_df()
    assert "RMSE" in lb_df.columns
    assert "R² Score" in lb_df.columns
