"""Audit Dataset view: Deep-dive diagnostic findings and column-level quality inventory."""

import streamlit as st

from datasleuth.quality import DatasetQualityReport, audit_dataset_quality
from datasleuth.reporting import generate_html_report, generate_json_report
from datasleuth.ui.components import render_findings_accordion, render_metric_card


def render_quality_view() -> None:
    """Render the Audit Dataset page with exhaustive diagnostics and feature quality table."""
    df = st.session_state.get("df")
    if df is None:
        st.warning("No active dataset found. Please upload a CSV or select a benchmark dataset from the Overview page.")
        return

    quality_report: DatasetQualityReport = st.session_state.get("quality_report")
    if quality_report is None:
        quality_report = audit_dataset_quality(df, target_col=st.session_state.get("target_col"))
        st.session_state["quality_report"] = quality_report

    # Section Header
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="font-family: 'Playfair Display', Georgia, serif; font-size: 30px; font-weight: 700; color: #FFFFFF; margin: 0;">
                Dataset Quality Audit
            </h1>
            <div style="font-size: 13.5px; color: #94A3B8; margin-top: 3px;">
                Exhaustive structural diagnostics, missingness rates, outlier boundaries, and heuristic anomaly detection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Metric Cards: Aligned in one row, low vertical padding, restrained borders
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        render_metric_card("Dataset Health", f"{quality_report.health_score}/100", "Overall health score", accent_color="#C084FC")
    with col_m2:
        render_metric_card("Duplicate Rows", f"{quality_report.duplicate_rows_count:,}", f"{quality_report.duplicate_rows_pct:.2f}% of rows")
    with col_m3:
        render_metric_card("Missing Cells", f"{quality_report.total_missing_cells:,}", f"{quality_report.overall_missing_pct:.2f}% overall")
    with col_m4:
        render_metric_card("Constant Features", f"{len(quality_report.constant_columns)}", "Zero variance columns")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # 2. Quality Findings Accordion: Compact list with facts vs heuristic warnings
    st.markdown("### Quality Findings & Heuristics")
    st.caption("Confirmed data-quality facts are separated from heuristic warnings. Click any item to inspect details.")

    # Filter Controls
    filter_c1, filter_c2 = st.columns([1, 1])
    with filter_c1:
        kind_filter = st.selectbox(
            "Filter Findings by Type",
            ["All Findings", "Confirmed Facts Only", "Heuristic Warnings Only"],
        )
    with filter_c2:
        sev_filter = st.selectbox(
            "Filter Findings by Severity",
            ["All Severities", "Critical Only", "Warning & Critical Only"],
        )

    filtered_diags = quality_report.diagnostics
    if kind_filter == "Confirmed Facts Only":
        filtered_diags = [d for d in filtered_diags if d.kind == "fact"]
    elif kind_filter == "Heuristic Warnings Only":
        filtered_diags = [d for d in filtered_diags if d.kind == "heuristic"]

    if sev_filter == "Critical Only":
        filtered_diags = [d for d in filtered_diags if d.severity == "critical"]
    elif sev_filter == "Warning & Critical Only":
        filtered_diags = [d for d in filtered_diags if d.severity in ["critical", "warning"]]

    # Render Compact Accordion
    render_findings_accordion(filtered_diags)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # 3. Column Quality Inventory Table
    st.markdown("### Feature Quality Inventory")
    
    col_df = quality_report.to_dataframe()

    c_f1, c_f2, c_f3 = st.columns([1.5, 1, 1])
    with c_f1:
        search_query = st.text_input("Search feature by name", placeholder="e.g. age, cholesterol...", label_visibility="collapsed")
    with c_f2:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All Types"] + sorted(list(col_df["Type"].unique())),
            label_visibility="collapsed",
        )
    with c_f3:
        only_issues = st.checkbox("Show only columns with warnings or missing data", value=False)

    filtered_table = col_df.copy()
    if search_query:
        filtered_table = filtered_table[filtered_table["Column"].str.contains(search_query, case=False, na=False)]
    if type_filter != "All Types":
        filtered_table = filtered_table[filtered_table["Type"] == type_filter]
    if only_issues:
        filtered_table = filtered_table[
            (filtered_table["Warnings"] != "None") | (filtered_table["Missing Count"] > 0)
        ]

    st.dataframe(
        filtered_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Feature", width="medium"),
            "Missing %": st.column_config.ProgressColumn("Missing %", format="%.1f%%", min_value=0, max_value=100),
            "Outliers (IQR)": st.column_config.TextColumn("Outliers (IQR)"),
        }
    )

    # Export Audit Artifacts
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### Export Audit Artifacts")
    c_dl1, c_dl2, c_dl3 = st.columns(3)
    with c_dl1:
        html_report = generate_html_report(
            dataset_name=st.session_state.get("dataset_name", "Dataset"),
            df=df,
            quality_report=quality_report,
            benchmark_result=st.session_state.get("benchmark_result"),
        )
        st.download_button(
            label="📄 Download Full HTML Report",
            data=html_report,
            file_name=f"datasleuth_audit_{st.session_state.get('dataset_name', 'dataset').replace(' ', '_').lower()}.html",
            mime="text/html",
            use_container_width=True,
        )
    with c_dl2:
        csv_bytes = filtered_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Quality Table (CSV)",
            data=csv_bytes,
            file_name="datasleuth_quality_audit.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_dl3:
        json_report = generate_json_report(
            quality_report=quality_report,
            benchmark_result=st.session_state.get("benchmark_result"),
        )
        st.download_button(
            label="💾 Export JSON Metadata",
            data=json_report,
            file_name="datasleuth_audit_metadata.json",
            mime="application/json",
            use_container_width=True,
        )

    # Educational Outlier Methodology Note
    with st.expander("ℹ️ Outlier Detection Methodology: Tukey's IQR Fences"):
        st.markdown("""
        - **Interquartile Range ($IQR$):** $IQR = Q_3 - Q_1$ (75th percentile minus 25th percentile).
        - **Fences:** Values outside $[Q_1 - 1.5 \\times IQR, Q_3 + 1.5 \\times IQR]$ are flagged as *potential* outliers.
        - **Heuristic Interpretation:** These represent statistical extremes rather than confirmed errors. In real-world data (such as health markers or financial distributions), genuine heavy tails should not be automatically purged without domain validation.
        """)
