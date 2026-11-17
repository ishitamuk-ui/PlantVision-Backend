"""Benchmark view: Baseline model training, evaluation leaderboard, and explainability."""

import plotly.express as px
import streamlit as st

from datasleuth.config import CONFIG
from datasleuth.modeling import (
    BenchmarkSuiteResult,
    get_default_features,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_residuals,
    run_benchmark_suite,
    suggest_task_type,
)
from datasleuth.quality import audit_dataset_quality
from datasleuth.ui.components import render_metric_card


def render_model_lab_view() -> None:
    """Render the Benchmark page with model training, leaderboard comparison, and diagnostics."""
    df = st.session_state.get("df")
    if df is None:
        st.warning("No active dataset found. Please upload a CSV or select a benchmark dataset from the Overview page.")
        return

    # Section Header
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="font-family: 'Playfair Display', Georgia, serif; font-size: 32px; font-weight: 700; color: #FFFFFF; margin: 0;">
                ML Benchmark Suite
            </h1>
            <div style="font-size: 14px; color: #94A3B8; margin-top: 4px;">
                Train standardized baseline estimators with leak-free preprocessing pipelines and compare evaluation metrics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Target Column Selection & Task Recommendation
    st.markdown("### 1. Target & Task Setup")
    c_tgt, c_task = st.columns([1.5, 1])
    
    all_cols = list(df.columns)
    current_target = st.session_state.get("target_col")
    default_target_idx = all_cols.index(current_target) if current_target in all_cols else len(all_cols) - 1

    with c_tgt:
        target_col = st.selectbox(
            "Select Prediction Target Variable",
            options=all_cols,
            index=default_target_idx,
            key="benchmark_target_select",
        )
        if target_col != st.session_state.get("target_col"):
            st.session_state["target_col"] = target_col
            st.session_state["quality_report"] = audit_dataset_quality(df, target_col=target_col)

    suggested_task = suggest_task_type(df, target_col)
    with c_task:
        task_type = st.selectbox(
            "Machine Learning Task",
            ["classification", "regression"],
            index=0 if suggested_task == "classification" else 1,
            format_func=lambda x: f"{x.capitalize()} (Suggested)" if x == suggested_task else x.capitalize(),
            key="benchmark_task_select",
        )
        st.session_state["task_type"] = task_type

    # 2. Feature Selection
    st.markdown("### 2. Feature Selection")
    default_features = get_default_features(df, target_col)
    available_features = [c for c in all_cols if c != target_col]

    c_b1, c_b2, _ = st.columns([1, 1, 3])
    with c_b1:
        if st.button("Recommended Features"):
            st.session_state["selected_features"] = default_features
            st.rerun()
    with c_b2:
        if st.button("Select All Features"):
            st.session_state["selected_features"] = available_features
            st.rerun()

    active_selected = st.session_state.get("selected_features", default_features)
    active_selected = [f for f in active_selected if f in available_features]

    selected_features = st.multiselect(
        "Included Feature Columns",
        options=available_features,
        default=active_selected,
        key="benchmark_feature_multiselect",
        help="Target, identifiers, and zero-variance columns are excluded by default to avoid artificial leakage.",
    )
    st.session_state["selected_features"] = selected_features

    if not selected_features:
        st.warning("Please select at least one feature column to train baseline models.")
        return

    # 3. Validation & Training Parameters
    st.markdown("### 3. Validation & Splitting Strategy")
    c_split, c_seed, c_cv = st.columns(3)
    with c_split:
        test_size = st.slider("Hold-Out Test Split (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
    with c_seed:
        random_seed = st.number_input("Random Seed (Reproducibility)", min_value=0, max_value=9999, value=42, step=1)
    with c_cv:
        cv_options = {"None (Holdout Only)": 0, "3-Fold Cross-Validation": 3, "5-Fold Cross-Validation": 5}
        cv_choice = st.selectbox("Cross-Validation on Train Split", list(cv_options.keys()), index=0)
        cv_folds = cv_options[cv_choice]

    # Preprocessing pipeline transparency notice
    with st.expander("ℹ️ Preprocessing Pipeline Specification (Leak-Free Protocol)", expanded=False):
        st.markdown("""
        - **Numerical Features:** Imputed using **median** strategy; scaled with **StandardScaler**.
        - **Categorical Features:** Imputed with **most frequent** category; encoded with **OneHotEncoder(handle_unknown='ignore')**.
        - **Data Leakage Safeguard:** All imputers and scalers are fitted exclusively on the training partition and transformed on test data.
        - **Stratified Partitioning:** Applied automatically for classification tasks when class distributions permit.
        """)

    # 4. Train Baselines Action
    st.markdown("---")
    train_clicked = st.button("⚡ Run ML Benchmark Suite", type="primary", use_container_width=True)

    if train_clicked:
        with st.spinner("Executing preprocessing pipelines and training baseline estimators..."):
            try:
                benchmark_result: BenchmarkSuiteResult = run_benchmark_suite(
                    df=df,
                    target_col=target_col,
                    feature_cols=selected_features,
                    task=task_type,
                    test_size=test_size,
                    random_seed=random_seed,
                    cv_folds=cv_folds,
                )
                st.session_state["benchmark_result"] = benchmark_result
                st.success("Baseline benchmarking completed successfully!")
            except Exception as e:
                st.error(f"Benchmarking encountered an error: {str(e)}")
                return

    benchmark_result: BenchmarkSuiteResult | None = st.session_state.get("benchmark_result")
    if benchmark_result is None:
        st.info("Configure settings above and click 'Run ML Benchmark Suite' to generate results.")
        return

    # Leaderboard Section
    st.markdown("### 🏆 Performance Leaderboard")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        render_metric_card("Top Performer", benchmark_result.best_model_name, f"Best on {benchmark_result.primary_metric}", accent_color="#C084FC")
    with col_b2:
        render_metric_card("Primary Metric", benchmark_result.primary_metric, f"Task: {benchmark_result.task.capitalize()}")
    with col_b3:
        strat_txt = "Stratified" if benchmark_result.is_stratified else "Random"
        render_metric_card("Train / Test Split", f"{benchmark_result.train_rows:,} / {benchmark_result.test_rows:,}", f"{strat_txt} ({benchmark_result.test_split_pct:.0f}% Test)")

    leaderboard_df = benchmark_result.get_leaderboard_df()
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    lb_csv = leaderboard_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Leaderboard (CSV)",
        data=lb_csv,
        file_name="datasleuth_benchmark_leaderboard.csv",
        mime="text/csv",
    )

    # Comparative Visualization Chart
    st.markdown("### Comparative Metric Analysis")
    if benchmark_result.task == "classification":
        comp_metrics = ["Accuracy", "F1-Score (Weighted)", "Precision (Weighted)", "Recall (Weighted)"]
        plot_df = leaderboard_df.melt(id_vars=["Model"], value_vars=[m for m in comp_metrics if m in leaderboard_df.columns], var_name="Metric", value_name="Score")
        fig_bar = px.bar(
            plot_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            template="plotly_dark",
            color_discrete_sequence=CONFIG.COLOR_PALETTE,
            title="Classification Baselines Across Key Metrics",
        )
        fig_bar.update_layout(
            yaxis_range=[0, 1.05],
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
            yaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        fig_bar = px.bar(
            leaderboard_df,
            x="Model",
            y="RMSE",
            color="Model",
            template="plotly_dark",
            color_discrete_sequence=CONFIG.COLOR_PALETTE,
            title="Model Comparison by Root Mean Squared Error (Lower is Better)",
        )
        fig_bar.update_layout(
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
            yaxis={"gridcolor": "rgba(255, 255, 255, 0.06)", "linecolor": "rgba(255, 255, 255, 0.15)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Best Model Deep-Dive
    st.markdown("---")
    st.markdown(f"### 🔍 Model Diagnostics: {benchmark_result.best_model_name}")

    best_res = next(r for r in benchmark_result.results if r.model_name == benchmark_result.best_model_name)

    c_dd1, c_dd2 = st.columns(2)

    with c_dd1:
        if benchmark_result.task == "classification":
            st.markdown("##### Confusion Matrix")
            st.caption("Diagonal elements represent correct predictions; off-diagonal elements show classification error.")
            fig_cm = plot_confusion_matrix(best_res.y_test, best_res.y_pred, benchmark_result.best_model_name)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.markdown("##### Residual Diagnostics")
            st.caption("Predicted values vs residual errors. Unbiased predictions should center around the zero line.")
            fig_res = plot_residuals(best_res.y_test, best_res.y_pred, benchmark_result.best_model_name)
            st.plotly_chart(fig_res, use_container_width=True)

    with c_dd2:
        st.markdown("##### Feature Importances / Coefficients")
        st.caption("Relative impact of top features on the model's decision function.")
        if best_res.feature_importances:
            fig_fi = plot_feature_importance(best_res.feature_importances, top_n=12)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance extraction is not available for this baseline estimator.")

    # Practical takeaways and limitations
    st.markdown("### Practical Takeaways & Limitations")
    st.markdown(f"""
    1. **Baseline Context:** The **{benchmark_result.best_model_name}** achieved the strongest baseline performance with a {benchmark_result.primary_metric} of **{best_res.metrics.get(benchmark_result.primary_metric, 0):.4f}**.
    2. **Dummy Guessing Gap:** Notice the gap between `{benchmark_result.results[0].model_name}` and `{benchmark_result.best_model_name}`. This gap measures genuine signal learned beyond trivial class guessing.
    3. **Generalization Constraints:** Baselines use default hyperparameters without hyperparameter search. Consider tuning tree depth, regularization, or trying gradient boosting algorithms.
    4. **Feature Integrity:** Verify the top important features against the Audit Dataset page to ensure no high-cardinality leakage columns or post-event identifiers were used.
    """)
