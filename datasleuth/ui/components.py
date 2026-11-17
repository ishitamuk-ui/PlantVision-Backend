"""Reusable UI components for DataSleuth matching refined modern editorial dark & light design."""

import streamlit as st

from datasleuth.modeling import BenchmarkSuiteResult
from datasleuth.quality import QualityDiagnostic


def render_editorial_header() -> bool:
    """Render the hero headline: 'Audit. Analyze. Benchmark.' and '+ New Audit' button.
    
    Returns:
        True if the user clicked the '+ New Audit' button, else False.
    """
    col_text, col_btn = st.columns([3.5, 1.2])
    
    with col_text:
        header_html = """
        <div class="hero-title-container">
            <h1 class="hero-title">
                Audit. Analyze. <span class="hero-title-italic">Benchmark.</span>
            </h1>
            <div class="hero-subtitle">
                Assess dataset reliability and establish leak-free ML baselines, end to end.
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
        
    with col_btn:
        st.write("")
        clicked = st.button("+ New Audit", key="new_audit_btn", type="primary", use_container_width=True)
        
    return clicked


def render_metric_card(
    label: str,
    value: str | int | float,
    subtext: str | None = None,
    accent_color: str | None = None,
) -> None:
    """Render a compact, refined KPI metric card with 18px squircle radius and micro-labels."""
    color_style = f"color: {accent_color};" if accent_color else ""
    sub_html = f"<div class='metric-subtext'>{subtext}</div>" if subtext else ""
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">
            <span class="metric-label-dot"></span>
            {label}
        </div>
        <div class="metric-value" style="{color_style}">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_featured_dataset_card(
    dataset_name: str,
    row_count: int,
    feature_count: int,
    health_score: int,
) -> None:
    """Render the wide horizontal featured dataset panel with live pulse dot & health score."""
    card_html = f"""
    <div class="ds-featured-card">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 18px;">
            <!-- Left Info Area -->
            <div style="display: flex; align-items: center; gap: 18px;">
                <div class="pulse-icon-box">
                    <svg viewBox="0 0 24 24">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                    </svg>
                </div>
                <div>
                    <div class="featured-header-row">
                        <span class="featured-label">ACTIVE DATASET</span>
                        <span class="live-status-pill">
                            <span class="status-pulse-dot"></span> AUDIT ACTIVE
                        </span>
                    </div>
                    <div class="featured-title">{dataset_name}</div>
                    <div class="featured-meta">
                        <span><strong>{row_count:,}</strong> rows</span>
                        <span>&bull;</span>
                        <span><strong>{feature_count}</strong> features</span>
                    </div>
                </div>
            </div>
            
            <!-- Right Health Score Area -->
            <div style="min-width: 210px; flex-shrink: 0;">
                <div class="health-score-title">Dataset Health Score</div>
                <div class="health-score-val">{health_score}<span>/100</span></div>
                <div class="health-progress-track">
                    <div class="health-progress-fill" style="width: {min(100, max(5, health_score))}%;"></div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_findings_accordion(diagnostics: list[QualityDiagnostic]) -> None:
    """Render quality findings as a compact, readable accordion list with 14px rounded cards."""
    if not diagnostics:
        st.info("No findings matching the selected filters.")
        return

    items_html = ""
    for diag in diagnostics:
        sev_class = f"sev-{diag.severity}"
        sev_label = f"&bull; {diag.severity.capitalize()}"
        
        kind_class = "badge-fact" if diag.kind == "fact" else "badge-heuristic"
        kind_label = "FACT" if diag.kind == "fact" else "HEURISTIC"

        preview_text = diag.message.split(". ")[0].strip()
        if not preview_text.endswith("."):
            preview_text += "."

        aff_html = ""
        if diag.affected_columns:
            chips = "".join([f"<span class='feature-chip'>{c}</span>" for c in diag.affected_columns])
            aff_html = f"<div style='margin-top: 10px;'><span style='font-size: 12px; font-weight: 700;'>Affected Features:</span> {chips}</div>"

        items_html += f"""
        <details class="ds-finding-item">
            <summary class="ds-finding-summary">
                <div class="finding-summary-left">
                    <span class="{kind_class}">{kind_label}</span>
                    <span class="sev-indicator {sev_class}">{sev_label}</span>
                    <span class="finding-title">{diag.title}</span>
                </div>
                <div class="finding-summary-preview">{preview_text}</div>
            </summary>
            <div class="finding-details-content">
                <div>{diag.message}</div>
                {aff_html}
            </div>
        </details>
        """

    st.markdown(items_html, unsafe_allow_html=True)


def render_benchmark_summary_card(
    benchmark_result: BenchmarkSuiteResult | None,
    on_trigger_benchmark=None,
) -> None:
    """Render the right panel in Overview: sleek ML Benchmark ranked capsules."""
    header_html = """
    <div class="panel-header">
        <div class="panel-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            ML Benchmark
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    if benchmark_result is None or not benchmark_result.results:
        empty_html = """
        <div style="padding: 24px 0; text-align: center; color: var(--ds-text-muted);">
            <p style="font-size: 13.5px; margin-bottom: 14px;">No baseline models evaluated yet for the active dataset.</p>
        </div>
        """
        st.markdown(empty_html, unsafe_allow_html=True)
        if st.button("⚡ Run Baseline Benchmarks", key="ov_run_benchmark_btn", use_container_width=True):
            if on_trigger_benchmark:
                on_trigger_benchmark()
        return

    # Render actual evaluated baseline models with rank badges
    rows_html = ""
    is_classification = benchmark_result.task == "classification"
    sorted_results = sorted(
        benchmark_result.results,
        key=lambda r: r.metrics.get(benchmark_result.primary_metric, 0.0),
        reverse=is_classification,
    )

    for idx, r in enumerate(sorted_results):
        score_val = r.metrics.get(benchmark_result.primary_metric, 0.0)
        score_str = f"{score_val * 100:.1f}%" if is_classification else f"{score_val:.2f}"
        display_name = r.model_name
        rank_num = f"#{idx + 1}"
        
        rows_html += f"""
        <div class="benchmark-row">
            <div class="benchmark-left">
                <span class="benchmark-rank-badge">{rank_num}</span>
                <span class="benchmark-model-name">{display_name}</span>
            </div>
            <div class="benchmark-score-pill">{score_str}</div>
        </div>
        """

    container_html = f"<div>{rows_html}</div>"
    st.markdown(container_html, unsafe_allow_html=True)


def render_sidebar_profile(initials: str = "IM", name: str = "Ishita Mukhopadhyay") -> None:
    """Render the compact user profile card at the bottom of the sidebar."""
    profile_html = f"""
    <div class="sidebar-profile">
        <div class="profile-avatar">{initials}</div>
        <div class="profile-name">{name}</div>
        <div class="profile-chevron">&#9662;</div>
    </div>
    """
    st.sidebar.markdown(profile_html, unsafe_allow_html=True)
