"""Exploratory Data Analysis view: Interactive Plotly statistical visualizations."""

import numpy as np
import streamlit as st

from datasleuth.eda import (
    plot_bivariate_scatter,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_feature_vs_target,
    plot_missingness_summary,
    plot_numerical_distribution,
    plot_target_distribution,
)
from datasleuth.ui.components import render_section_heading


def render_eda_view() -> None:
    """Render the Exploratory Data Analysis page with interactive charts."""
    df = st.session_state.get("df")
    if df is None:
        st.warning("No active dataset found. Please upload a CSV or select a benchmark dataset in the Overview tab.")
        return

    render_section_heading(
        "Exploratory Data Analysis (EDA)",
        "Interactive visualizations of statistical distributions, missingness patterns, pairwise correlations, and target relationships.",
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    target_col = st.session_state.get("target_col")
    task_type = st.session_state.get("task_type") or "classification"

    eda_tabs = st.tabs([
        "Feature Distributions",
        "Missingness Pattern",
        "Correlation Heatmap",
        "Feature vs Target",
        "Bivariate Explorer",
    ])

    # Tab 1: Feature Distributions
    with eda_tabs[0]:
        st.markdown("#### Feature Distributions")
        f_type = st.radio("Distribution Type", ["Numerical Features", "Categorical Features"], horizontal=True)
        
        if f_type == "Numerical Features":
            if not numeric_cols:
                st.info("No numerical features found in dataset.")
            else:
                c_sel, c_opt = st.columns([2, 1])
                with c_sel:
                    selected_num = st.selectbox("Select Numerical Feature", numeric_cols, key="eda_num_sel")
                with c_opt:
                    show_box = st.checkbox("Include Box Plot Marginal", value=True)

                if selected_num:
                    fig = plot_numerical_distribution(df, selected_num, show_box=show_box)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            if not categorical_cols:
                st.info("No categorical features found in dataset.")
            else:
                c_sel, c_opt = st.columns([2, 1])
                with c_sel:
                    selected_cat = st.selectbox("Select Categorical Feature", categorical_cols, key="eda_cat_sel")
                with c_opt:
                    top_n = st.slider("Top N Categories", min_value=5, max_value=30, value=15)

                if selected_cat:
                    fig = plot_categorical_distribution(df, selected_cat, max_categories=top_n)
                    st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Missingness Pattern
    with eda_tabs[1]:
        st.markdown("#### Missing Values Distribution")
        fig_miss = plot_missingness_summary(df)
        st.plotly_chart(fig_miss, use_container_width=True)

    # Tab 3: Correlation Heatmap
    with eda_tabs[2]:
        st.markdown("#### Numerical Correlation Heatmap")
        if len(numeric_cols) < 2:
            st.info("At least two numerical columns are required to generate a correlation heatmap.")
        else:
            c_meth, c_thresh = st.columns([1, 1])
            with c_meth:
                corr_method = st.selectbox("Correlation Method", ["pearson", "spearman"], index=0)
            with c_thresh:
                corr_thresh = st.slider("Highlight Minimum Absolute Correlation (|r|)", min_value=0.0, max_value=0.9, value=0.0, step=0.05)

            fig_corr = plot_correlation_heatmap(df, method=corr_method, min_threshold=corr_thresh)
            st.plotly_chart(fig_corr, use_container_width=True)

    # Tab 4: Feature vs Target
    with eda_tabs[3]:
        st.markdown("#### Feature vs Target Relationships")
        if not target_col or target_col not in df.columns:
            st.info("No target variable is currently selected. Select a target in the sidebar or Model Lab to inspect target relationships.")
        else:
            st.caption(f"Active Target: **`{target_col}`** ({task_type.capitalize()})")
            
            # Show target distribution first
            with st.expander("Target Variable Distribution", expanded=False):
                fig_target = plot_target_distribution(df, target_col, task_type)
                st.plotly_chart(fig_target, use_container_width=True)

            available_features = [c for c in df.columns if c != target_col]
            if not available_features:
                st.info("No feature columns available to compare.")
            else:
                feat_sel = st.selectbox("Select Feature to Compare Against Target", available_features, key="eda_feat_target_sel")
                if feat_sel:
                    fig_ft = plot_feature_vs_target(df, feat_sel, target_col, task_type)
                    st.plotly_chart(fig_ft, use_container_width=True)

    # Tab 5: Bivariate Explorer
    with eda_tabs[4]:
        st.markdown("#### Bivariate Feature Explorer")
        st.caption("Inspect relationships between any two variables with optional color/grouping encoding.")
        
        c_x, c_y, c_c, c_s = st.columns(4)
        all_cols = list(df.columns)
        with c_x:
            x_var = st.selectbox("X-Axis Feature", all_cols, index=0, key="biv_x")
        with c_y:
            default_y = 1 if len(all_cols) > 1 else 0
            y_var = st.selectbox("Y-Axis Feature", all_cols, index=default_y, key="biv_y")
        with c_c:
            color_var = st.selectbox("Color / Hue (Optional)", ["(None)"] + all_cols, index=0, key="biv_c")
        with c_s:
            size_var = st.selectbox("Marker Size (Optional)", ["(None)"] + numeric_cols, index=0, key="biv_s")

        color_col = None if color_var == "(None)" else color_var
        size_col = None if size_var == "(None)" else size_var

        if x_var and y_var:
            fig_biv, is_sampled, orig_count = plot_bivariate_scatter(
                df, x_var, y_var, color_col=color_col, size_col=size_col
            )
            if is_sampled:
                st.markdown(f"<span class='sample-badge'>Note: Sampled {len(df):,} → 5,000 points for fluid rendering</span>", unsafe_allow_html=True)
            st.plotly_chart(fig_biv, use_container_width=True)
