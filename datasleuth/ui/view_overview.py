"""Overview view: Minimalist, editorial dashboard matching the reference design."""

import streamlit as st

from datasleuth.eda import plot_overview_feature_distribution
from datasleuth.ingestion import (
    get_sample_datasets,
    load_csv_from_bytes,
    load_sample_dataset,
)
from datasleuth.modeling import get_default_features, run_benchmark_suite
from datasleuth.quality import audit_dataset_quality
from datasleuth.ui.components import (
    render_benchmark_summary_card,
    render_editorial_header,
    render_featured_dataset_card,
)


@st.dialog("Start New Audit", width="medium")
def open_new_audit_dialog() -> None:
    """Dialog modal to upload a new CSV or choose a benchmark dataset."""
    st.markdown("<div style='font-size: 14px; color: #94A3B8; margin-bottom: 16px;'>Ingest a new tabular CSV file or switch to an authentic benchmark dataset.</div>", unsafe_allow_html=True)
    
    tab_sample, tab_upload = st.tabs(["Benchmark Datasets", "Upload CSV File"])
    
    with tab_sample:
        samples = get_sample_datasets()
        sample_key = st.selectbox(
            "Select Benchmark Dataset",
            options=list(samples.keys()),
            format_func=lambda k: f"{samples[k]['name']} ({samples[k]['task'].capitalize()})",
            key="modal_sample_select",
        )
        s_info = samples[sample_key]
        st.caption(f"**Description:** {s_info['description']}")
        st.caption(f"**Target:** `{s_info['target']}` &bull; **Task:** {s_info['task'].capitalize()}")
        
        if st.button("Load Benchmark Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading benchmark dataset..."):
                df = load_sample_dataset(sample_key)
                st.session_state["dataset_name"] = s_info["name"]
                st.session_state["df"] = df
                st.session_state["target_col"] = s_info["target"]
                st.session_state["task_type"] = s_info["task"]
                st.session_state["quality_report"] = audit_dataset_quality(df, target_col=s_info["target"])
                
                # Auto-run baselines for immediate overview display
                try:
                    default_feats = get_default_features(df, s_info["target"])
                    suite = run_benchmark_suite(
                        df=df,
                        target_col=s_info["target"],
                        feature_cols=default_feats,
                        task=s_info["task"],
                        test_size=0.20,
                        random_seed=42,
                    )
                    st.session_state["benchmark_result"] = suite
                except Exception:
                    st.session_state["benchmark_result"] = None
                    
                st.session_state["source_mode"] = "sample"
                st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a CSV file (Max 50MB)",
            type=["csv"],
            key="modal_file_uploader",
        )
        if uploaded_file is not None:
            if st.button("Ingest & Audit File", type="primary", use_container_width=True):
                with st.spinner("Auditing uploaded CSV..."):
                    file_bytes = uploaded_file.getvalue()
                    res = load_csv_from_bytes(file_bytes, filename=uploaded_file.name)
                    if not res.success:
                        st.error(res.error_message)
                    else:
                        df = res.df
                        st.session_state["dataset_name"] = uploaded_file.name
                        st.session_state["df"] = df
                        st.session_state["target_col"] = None
                        st.session_state["task_type"] = None
                        st.session_state["quality_report"] = audit_dataset_quality(df, target_col=None)
                        st.session_state["benchmark_result"] = None
                        st.session_state["source_mode"] = "upload"
                        st.rerun()


def render_overview_view() -> None:
    """Render the simplified, editorial Overview dashboard."""
    # 1. Header with headline & '+ New Audit' button
    new_audit_clicked = render_editorial_header()
    if new_audit_clicked:
        open_new_audit_dialog()

    df = st.session_state.get("df")
    
    # Empty state handling
    if df is None:
        st.markdown(
            """
            <div class="ds-card" style="text-align: center; padding: 48px 24px;">
                <h3 style="color: #FFFFFF; font-family: 'Playfair Display', serif; margin-bottom: 8px;">No Active Dataset</h3>
                <p style="color: #94A3B8; font-size: 14px; max-width: 480px; margin: 0 auto 20px auto;">
                    Upload a tabular CSV file or select a built-in benchmark dataset to inspect data quality and benchmark ML baselines.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⚡ Select Benchmark Dataset", type="primary"):
            open_new_audit_dialog()
        return

    # Ensure quality report exists
    quality_report = st.session_state.get("quality_report")
    if quality_report is None:
        quality_report = audit_dataset_quality(df, target_col=st.session_state.get("target_col"))
        st.session_state["quality_report"] = quality_report

    dataset_name = st.session_state.get("dataset_name", "Dataset")

    # 2. Featured Dataset Panel (Wide horizontal anchor)
    render_featured_dataset_card(
        dataset_name=dataset_name,
        row_count=quality_report.row_count,
        feature_count=quality_report.column_count,
        health_score=quality_report.health_score,
    )

    # 3. Two-Column Main Analytics Section
    col_left, col_right = st.columns([1.4, 1.0], gap="large")

    # --- Left Panel: Feature Distribution ---
    with col_left:
        st.markdown('<div class="ds-card" style="min-height: 330px;">', unsafe_allow_html=True)
        
        # Header with inline feature selector dropdown
        header_c1, header_c2 = st.columns([1.5, 1])
        with header_c1:
            st.markdown(
                """
                <div class="panel-title" style="margin-top: 4px;">
                    Feature Distribution
                </div>
                """,
                unsafe_allow_html=True,
            )
        with header_c2:
            # Feature dropdown selector
            all_cols = list(df.columns)
            # Default to 'age' if present, else first column
            default_col_idx = all_cols.index("age") if "age" in all_cols else 0
            selected_feature = st.selectbox(
                "Select Feature",
                options=all_cols,
                index=default_col_idx,
                label_visibility="collapsed",
                key="ov_feature_selector",
            )

        # Distribution Chart
        if selected_feature:
            fig_dist = plot_overview_feature_distribution(df, selected_feature)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Right Panel: ML Benchmark ---
    with col_right:
        st.markdown('<div class="ds-card" style="min-height: 330px;">', unsafe_allow_html=True)
        
        benchmark_result = st.session_state.get("benchmark_result")
        target_col = st.session_state.get("target_col")
        task_type = st.session_state.get("task_type") or "classification"

        def trigger_quick_benchmark():
            with st.spinner("Training baseline estimators..."):
                # If no target specified, choose last column or intelligent target
                tgt = target_col or df.columns[-1]
                default_feats = get_default_features(df, tgt)
                suite = run_benchmark_suite(
                    df=df,
                    target_col=tgt,
                    feature_cols=default_feats,
                    task=task_type,
                    test_size=0.20,
                    random_seed=42,
                )
                st.session_state["benchmark_result"] = suite
                st.session_state["target_col"] = tgt
                st.rerun()

        render_benchmark_summary_card(
            benchmark_result=benchmark_result,
            on_trigger_benchmark=trigger_quick_benchmark,
        )

        st.markdown('</div>', unsafe_allow_html=True)
