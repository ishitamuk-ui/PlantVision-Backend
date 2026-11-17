"""Machine Learning Model Lab: Baseline benchmarking and evaluation pipelines."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from datasleuth.config import CONFIG


@dataclass
class ModelResult:
    """Evaluation metrics and artifacts for a single trained model."""
    model_name: str
    task: str
    fit_time_sec: float
    metrics: dict[str, float]
    cv_mean_score: float | None = None
    cv_std_score: float | None = None
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_pred_proba: np.ndarray | None = None
    feature_importances: dict[str, float] | None = None
    model_object: Any = None


@dataclass
class BenchmarkSuiteResult:
    """Collection of benchmarked models with leaderboard and metadata."""
    task: str
    target_column: str
    feature_columns: list[str]
    train_rows: int
    test_rows: int
    test_split_pct: float
    random_seed: int
    is_stratified: bool
    cv_folds: int
    results: list[ModelResult]
    best_model_name: str
    primary_metric: str

    def get_leaderboard_df(self) -> pd.DataFrame:
        """Construct a formatted leaderboard DataFrame sorted by primary metric."""
        records = []
        for r in self.results:
            rec = {
                "Model": r.model_name,
                "Fit Time (s)": round(r.fit_time_sec, 3),
            }
            if r.cv_mean_score is not None:
                rec["CV Score"] = f"{r.cv_mean_score:.3f} ± {r.cv_std_score:.3f}"
            for k, v in r.metrics.items():
                rec[k] = round(v, 4) if isinstance(v, float) else v
            records.append(rec)

        df = pd.DataFrame(records)
        # Sort by primary metric
        if self.primary_metric in df.columns:
            ascending = self.task == "regression" and self.primary_metric in ["MAE", "MSE", "RMSE"]
            df = df.sort_values(by=self.primary_metric, ascending=ascending).reset_index(drop=True)
        return df


def suggest_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Recommend classification or regression based on target properties."""
    target_series = df[target_col].dropna()
    if not pd.api.types.is_numeric_dtype(target_series):
        return "classification"
    n_unique = target_series.nunique()
    if n_unique <= 20:
        return "classification"
    return "regression"


def get_default_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """Provide a smart default feature selection excluding target and ID/constant columns."""
    from datasleuth.quality import infer_semantic_type

    permitted = []
    for col in df.columns:
        if col == target_col:
            continue
        sem = infer_semantic_type(col, df[col])
        if sem in ["identifier"]:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        permitted.append(col)
    return permitted


def build_preprocessing_pipeline(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a robust, leak-free Scikit-learn preprocessing ColumnTransformer."""
    numeric_features = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_features = [
        c for c in feature_cols if c not in numeric_features
    ]

    transformers = []
    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


def get_candidate_models(task: str, random_seed: int = 42) -> dict[str, Any]:
    """Return dictionary of baseline estimators based on task type."""
    if task == "classification":
        return {
            "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=random_seed),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_seed),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=random_seed),
            "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=random_seed),
        }
    else:
        return {
            "Dummy (Mean)": DummyRegressor(strategy="mean"),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=random_seed),
        }


def extract_feature_names(preprocessor: ColumnTransformer, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    """Extract human-readable feature names post-onehot-encoding."""
    feature_names = []
    try:
        # Scikit-learn 1.0+ get_feature_names_out
        feature_names = list(preprocessor.get_feature_names_out())
        # Clean prefix like 'num__' and 'cat__'
        feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    except Exception:
        feature_names.extend(numeric_cols)
        feature_names.extend(categorical_cols)
    return feature_names


def run_benchmark_suite(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    task: str = "classification",
    test_size: float = 0.20,
    random_seed: int = 42,
    cv_folds: int = 0,
) -> BenchmarkSuiteResult:
    """Execute complete end-to-end baseline benchmarking with proper train/test isolation."""
    # Drop rows where target is missing
    clean_df = df.dropna(subset=[target_col]).copy()
    X = clean_df[feature_cols]
    y = clean_df[target_col]

    # Preprocessing structure
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(clean_df, feature_cols)

    # Splitting strategy
    stratify = None
    is_stratified = False
    if task == "classification":
        val_counts = y.value_counts()
        if (val_counts >= 2).all() and len(val_counts) > 1:
            stratify = y
            is_stratified = True

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=stratify
    )

    models_dict = get_candidate_models(task, random_seed)
    model_results: list[ModelResult] = []

    primary_metric = "F1-Score (Weighted)" if task == "classification" else "RMSE"

    for name, estimator in models_dict.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        start_t = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_t

        y_pred = pipeline.predict(X_test)
        y_prob = None
        if task == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)
            except Exception:
                y_prob = None

        # Calculate metrics
        metrics: dict[str, float] = {}
        if task == "classification":
            metrics["Accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["F1-Score (Weighted)"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["F1-Score (Macro)"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            metrics["Precision (Weighted)"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["Recall (Weighted)"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

            # ROC-AUC if binary or multiclass
            if y_prob is not None:
                try:
                    n_classes = len(np.unique(y_train))
                    if n_classes == 2:
                        metrics["ROC-AUC"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        metrics["ROC-AUC (OVR)"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
                except Exception:
                    pass
        else:
            # Regression metrics
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))
            metrics["RMSE"] = rmse
            metrics["MAE"] = mae
            metrics["MSE"] = mse
            metrics["R² Score"] = r2

        # Optional Cross-Validation on training set
        cv_mean, cv_std = None, None
        if cv_folds > 1:
            try:
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed) if is_stratified else KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
                scoring_metric = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring_metric)
                if task == "regression":
                    cv_scores = -cv_scores
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
            except Exception:
                cv_mean, cv_std = None, None

        # Feature importances or coefficients
        feat_importances = None
        try:
            fitted_prep = pipeline.named_steps["preprocessor"]
            feature_names = extract_feature_names(fitted_prep, num_cols, cat_cols)
            model_est = pipeline.named_steps["model"]

            if hasattr(model_est, "feature_importances_"):
                raw_imp = model_est.feature_importances_
                if len(raw_imp) == len(feature_names):
                    feat_importances = {name: float(val) for name, val in zip(feature_names, raw_imp)}
            elif hasattr(model_est, "coef_"):
                raw_coef = model_est.coef_
                if raw_coef.ndim > 1:
                    raw_coef = np.mean(np.abs(raw_coef), axis=0)
                if len(raw_coef) == len(feature_names):
                    feat_importances = {name: float(abs(val)) for name, val in zip(feature_names, raw_coef)}
        except Exception:
            feat_importances = None

        model_results.append(ModelResult(
            model_name=name,
            task=task,
            fit_time_sec=fit_time,
            metrics=metrics,
            cv_mean_score=cv_mean,
            cv_std_score=cv_std,
            y_test=np.asarray(y_test),
            y_pred=np.asarray(y_pred),
            y_pred_proba=y_prob,
            feature_importances=feat_importances,
            model_object=pipeline,
        ))

    # Determine best model
    if task == "classification":
        best_model = max(model_results, key=lambda m: m.metrics.get(primary_metric, 0.0))
    else:
        best_model = min(model_results, key=lambda m: m.metrics.get(primary_metric, float("inf")))

    return BenchmarkSuiteResult(
        task=task,
        target_column=target_col,
        feature_columns=feature_cols,
        train_rows=len(X_train),
        test_rows=len(X_test),
        test_split_pct=test_size * 100.0,
        random_seed=random_seed,
        is_stratified=is_stratified,
        cv_folds=cv_folds,
        results=model_results,
        best_model_name=best_model.model_name,
        primary_metric=primary_metric,
    )


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """Generate an annotated interactive confusion matrix heatmap."""
    labels = sorted(list(set(y_true) | set(y_pred)))
    label_str = [str(lbl) for lbl in labels]
    
    cm = pd.crosstab(
        pd.Series(y_true, name="Actual"),
        pd.Series(y_pred, name="Predicted"),
        dropna=False
    ).reindex(index=labels, columns=labels, fill_value=0)

    fig = px.imshow(
        cm.values,
        x=label_str,
        y=label_str,
        text_auto=True,
        color_continuous_scale="Purples",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        title=f"Confusion Matrix: {model_name}",
        template="plotly_dark",
    )
    fig.update_layout(
        font={"family": "Inter, sans-serif", "color": "#94A3B8"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 30, "t": 50, "b": 50},
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
    )
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """Plot actual vs predicted values alongside residual distribution."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    # Scatter of actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode="markers",
        marker={"color": CONFIG.COLOR_LAVENDER_BAR, "opacity": 0.75, "size": 7},
        name="Residuals",
        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
    ))
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=CONFIG.COLOR_DANGER, annotation_text="Zero Error")
    
    fig.update_layout(
        font={"family": "Inter, sans-serif", "color": "#94A3B8"},
        title=f"Residual Plot: {model_name} (Predicted vs Residuals)",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 30, "t": 50, "b": 50},
        xaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
        yaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
    )
    return fig


def plot_feature_importance(importances: dict[str, float], top_n: int = 15) -> go.Figure:
    """Plot top feature importances or coefficients."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_items:
        fig = go.Figure()
        fig.add_annotation(text="No feature importances available for this model.", showarrow=False, font={"color": "#94A3B8"})
        return fig

    names = [item[0] for item in reversed(sorted_items)]
    vals = [item[1] for item in reversed(sorted_items)]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color=CONFIG.COLOR_PRIMARY,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        font={"family": "Inter, sans-serif", "color": "#94A3B8"},
        title=f"Top {len(sorted_items)} Feature Importances / Coefficients",
        xaxis_title="Relative Importance / Magnitude",
        yaxis_title="Feature Name",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 100, "r": 40, "t": 50, "b": 50},
        xaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
        yaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
    )
    return fig
