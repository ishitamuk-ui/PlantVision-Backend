"""DataSleuth: Automated Dataset Auditing & ML Benchmarking Platform.

Main Streamlit Application Entrypoint.
"""

import streamlit as st

from datasleuth.config import CONFIG
from datasleuth.ingestion import load_sample_dataset
from datasleuth.modeling import get_default_features, run_benchmark_suite
from datasleuth.quality import audit_dataset_quality
from datasleuth.ui.components import render_sidebar_profile
from datasleuth.ui.styles import apply_custom_styles
from datasleuth.ui.view_documentation import render_documentation_view
from datasleuth.ui.view_model_lab import render_model_lab_view
from datasleuth.ui.view_overview import render_overview_view
from datasleuth.ui.view_quality import render_quality_view

# Streamlit Page Setup
st.set_page_config(
    page_title=f"{CONFIG.APP_NAME} — Audit. Analyze. Benchmark.",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme State Initialization (default to Dark)
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Apply Editorial CSS according to active theme
apply_custom_styles(theme=st.session_state["theme"])


def init_session_state() -> None:
    """Initialize default session state with authentic baseline dataset and precomputed benchmarks."""
    if "df" not in st.session_state:
        # Load default benchmark dataset so user immediately sees live data
        sample_df = load_sample_dataset("heart_disease")
        st.session_state["dataset_name"] = "Heart Disease Clinical Risk"
        st.session_state["df"] = sample_df
        st.session_state["target_col"] = "heart_disease"
        st.session_state["task_type"] = "classification"
        st.session_state["quality_report"] = audit_dataset_quality(sample_df, target_col="heart_disease")
        st.session_state["source_mode"] = "sample"

        # Precompute initial baselines so Overview ML Benchmark panel displays genuine scores immediately
        try:
            default_feats = get_default_features(sample_df, "heart_disease")
            suite = run_benchmark_suite(
                df=sample_df,
                target_col="heart_disease",
                feature_cols=default_feats,
                task="classification",
                test_size=0.20,
                random_seed=42,
            )
            st.session_state["benchmark_result"] = suite
        except Exception:
            st.session_state["benchmark_result"] = None


init_session_state()

# Sidebar Navigation matching the refined design
with st.sidebar:
    # 1. Logo and Wordmark
    st.markdown(
        """
        <div class="sidebar-brand-container">
            <div class="sidebar-logo-icon">
                <svg viewBox="0 0 24 24">
                    <circle cx="11" cy="11" r="7"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
            </div>
            <div class="sidebar-brand-text">DataSleuth</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Modern Theme Switcher (Dark / Light)
    active_theme = st.session_state.get("theme", "dark")
    theme_choice = st.radio(
        "Theme",
        options=["🌙 Dark", "☀️ Light"],
        index=0 if active_theme == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="app_theme_toggle",
    )
    chosen_theme = "light" if "Light" in theme_choice else "dark"
    if chosen_theme != active_theme:
        st.session_state["theme"] = chosen_theme
        st.rerun()

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 3. Primary Navigation Items (only 4)
    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Audit Dataset",
            "Benchmark",
            "Documentation",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # Natural spacing for profile
    st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)

    # 4. User Profile at bottom
    render_sidebar_profile(initials="IM", name="Ishita Mukhopadhyay")

# Route to active view
if nav_selection == "Overview":
    render_overview_view()
elif nav_selection == "Audit Dataset":
    render_quality_view()
elif nav_selection == "Benchmark":
    render_model_lab_view()
elif nav_selection == "Documentation":
    render_documentation_view()
