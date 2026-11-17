"""Reports view: Audit report generation and downloadable artifacts."""

import streamlit as st

from datasleuth.quality import DatasetQualityReport, audit_dataset_quality
from datasleuth.reporting import generate_html_report, generate_json_report
from datasleuth.ui.components import render_health_score, render_metric_card, render_section_heading


def render_reports_view() -> None:
    """Render the Reports page with executive summary and downloadable artifacts."""
    df = st.session_state.get("df")
    if df is None:
        st.warning("No active dataset found. Please upload a CSV or select a benchmark dataset in the Overview tab.")
        return

    quality_report: DatasetQualityReport = st.session_state.get("quality_report")
    if quality_report is None:
        quality_report = audit_dataset_quality(df, target_col=st.session_state.get("target_col"))
        st.session_state["quality_report"] = quality_report

    benchmark_result = st.session_state.get("benchmark_result")
    dataset_name = st.session_state.get("dataset_name", "Dataset")

    render_section_heading(
        "Audit Reports & Exports",
        "Generate publication-ready audit reports, executive summaries, and structured metadata exports.",
    )

    # Executive Overview
    st.markdown("#### Executive Summary Preview")
    
    col_score, col_e1, col_e2, col_e3 = st.columns([1.2, 1, 1, 1])
    with col_score:
        render_health_score(quality_report.health_score)
    with col_e1:
        render_metric_card("Observations", f"{quality_report.row_count:,}", "Total rows")
    with col_e2:
        render_metric_card("Features", f"{quality_report.column_count:,}", "Total columns")
    with col_e3:
        best_name = benchmark_result.best_model_name if benchmark_result else "Not Benchmarked"
        render_metric_card("Top Baseline", best_name, "Best model")

    st.markdown("---")

    # Export Section
    st.markdown("#### Export Artifacts")
    st.caption("All reports are generated client-side from live computations without external dependencies.")

    c_dl1, c_dl2, c_dl3 = st.columns(3)

    # 1. Standalone HTML Report
    html_content = generate_html_report(
        dataset_name=dataset_name,
        df=df,
        quality_report=quality_report,
        benchmark_result=benchmark_result,
    )
    with c_dl1:
        st.markdown("**1. Standalone HTML Audit Report**")
        st.caption("Self-contained publication-ready HTML file with executive summary, quality tables, and benchmark results.")
        st.download_button(
            label="📄 Download HTML Report",
            data=html_content,
            file_name=f"datasleuth_audit_{dataset_name.replace(' ', '_').lower()}.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
        )

    # 2. Quality Audit CSV
    csv_content = quality_report.to_dataframe().to_csv(index=False)
    with c_dl2:
        st.markdown("**2. Data Quality Table (CSV)**")
        st.caption("Tabular inventory of all column metrics, missingness, outlier counts, and heuristic warnings.")
        st.download_button(
            label="📊 Download Quality CSV",
            data=csv_content,
            file_name=f"datasleuth_quality_{dataset_name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 3. Complete Audit JSON Metadata
    json_content = generate_json_report(
        quality_report=quality_report,
        benchmark_result=benchmark_result,
    )
    with c_dl3:
        st.markdown("**3. Machine-Readable Metadata (JSON)**")
        st.caption("Structured JSON object containing full diagnostic details, health scoring, and benchmark results.")
        st.download_button(
            label="💾 Download Audit JSON",
            data=json_content,
            file_name=f"datasleuth_audit_{dataset_name.replace(' ', '_').lower()}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")

    # In-app summary preview
    st.markdown("#### Quality Diagnostics Inventory")
    for diag in quality_report.diagnostics:
        badge = "ℹ️ INFO" if diag.severity == "info" else "⚠️ WARNING" if diag.severity == "warning" else "🚨 CRITICAL"
        st.markdown(f"**{badge} &bull; {diag.title}** ({diag.kind})")
        st.markdown(f"> {diag.message}")
        if diag.affected_columns:
            st.caption(f"Affected Features: {', '.join(diag.affected_columns)}")
        st.write("")
