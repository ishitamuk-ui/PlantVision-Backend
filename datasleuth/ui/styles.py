"""Editorial Dark & Lavender design system and refined CSS styles for DataSleuth."""

import streamlit as st


def apply_custom_styles() -> None:
    """Inject polished, restrained editorial dark & lavender CSS into the Streamlit app."""
    custom_css = """
    <style>
        /* Import Inter for UI and Playfair Display for editorial headings */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,600;1,700&display=swap');

        /* Global application background and text colors */
        .stApp {
            background-color: #0C0A14 !important;
            color: #F8FAFC !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        /* Top header bar */
        header[data-testid="stHeader"] {
            background-color: rgba(12, 10, 20, 0.85) !important;
            backdrop-filter: blur(10px) !important;
        }

        /* Sidebar Styling: Compact and visually quiet */
        section[data-testid="stSidebar"] {
            background-color: #090810 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
            padding-top: 0.75rem !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }

        /* Sidebar Brand */
        .sidebar-brand-container {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 10px 18px 10px;
        }
        .sidebar-logo-icon {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: 2px solid #C084FC;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .sidebar-logo-icon::after {
            content: '';
            position: absolute;
            width: 5px;
            height: 2px;
            background: #C084FC;
            transform: rotate(45deg);
            bottom: 3px;
            right: 3px;
        }
        .sidebar-brand-text {
            font-size: 18px;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.02em;
        }

        /* Sidebar Navigation Radio Styling */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
            gap: 4px;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background-color: transparent !important;
            border-radius: 9px !important;
            padding: 8px 12px !important;
            margin: 0 !important;
            border: 1px solid transparent !important;
            color: #94A3B8 !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            transition: all 0.15s ease !important;
            cursor: pointer !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            color: #F8FAFC !important;
            background-color: rgba(255, 255, 255, 0.03) !important;
        }
        /* Active Radio Option Pill */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
            background: rgba(168, 85, 247, 0.12) !important;
            border: 1px solid rgba(168, 85, 247, 0.25) !important;
            color: #F3E8FF !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
            color: inherit !important;
            font-size: 13.5px !important;
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioButtonCustom"] {
            display: none !important;
        }

        /* Sidebar Profile Footer: Compact & quiet */
        .sidebar-profile {
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 10px 12px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 16px;
        }
        .profile-avatar {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: linear-gradient(135deg, #A855F7, #6366F1);
            color: #FFFFFF;
            font-size: 11px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .profile-name {
            font-size: 12.5px;
            color: #CBD5E1;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex-grow: 1;
        }
        .profile-chevron {
            color: #64748B;
            font-size: 11px;
        }

        /* Hero Header Section */
        .hero-title-container {
            margin-bottom: 18px;
        }
        .hero-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 34px;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.01em;
            margin: 0;
            line-height: 1.2;
        }
        .hero-title-italic {
            font-style: italic;
            font-weight: 700;
            color: #C084FC;
        }
        .hero-subtitle {
            font-size: 14px;
            color: #94A3B8;
            margin-top: 4px;
            font-weight: 400;
        }

        /* + New Audit Pill Button */
        button[key="new_audit_btn"], div.stButton > button[kind="primary"] {
            background-color: #DDD6FE !important;
            color: #1E1B4B !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: 20px !important;
            padding: 7px 18px !important;
            font-size: 13.5px !important;
            transition: all 0.15s ease !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15) !important;
        }
        button[key="new_audit_btn"]:hover, div.stButton > button[kind="primary"]:hover {
            background-color: #EDE9FE !important;
            transform: translateY(-1px) !important;
        }

        /* Standardized Card Container (consistent 14-16px radius, subtle borders) */
        .ds-card {
            background-color: #141222;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 16px;
            position: relative;
        }

        /* Featured Dataset Panel */
        .ds-featured-card {
            background-color: #141222;
            border: 1px solid rgba(168, 85, 247, 0.2);
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
        }

        /* Pulse / EKG Squircle Icon */
        .pulse-icon-box {
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background-color: #1D1A2E;
            border: 1px solid rgba(168, 85, 247, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .pulse-icon-box svg {
            width: 22px;
            height: 22px;
            stroke: #C084FC;
            fill: none;
            stroke-width: 2.2;
            stroke-linecap: round;
            stroke-linejoin: round;
        }

        /* Featured Card Typography */
        .featured-label {
            font-size: 10.5px;
            font-weight: 700;
            color: #94A3B8;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .featured-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 21px;
            font-weight: 600;
            color: #FFFFFF;
            letter-spacing: -0.01em;
            margin-bottom: 2px;
        }
        .featured-meta {
            font-size: 12.5px;
            color: #94A3B8;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Health Score Element */
        .health-score-title {
            font-size: 12px;
            color: #94A3B8;
            font-weight: 500;
            margin-bottom: 2px;
        }
        .health-score-val {
            font-size: 26px;
            font-weight: 700;
            color: #C084FC;
            line-height: 1;
            margin-bottom: 8px;
        }
        .health-score-val span {
            color: #64748B;
            font-size: 17px;
            font-weight: 400;
        }
        .health-progress-track {
            width: 100%;
            height: 6px;
            background-color: #231F3A;
            border-radius: 999px;
            overflow: hidden;
            position: relative;
        }
        .health-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #A855F7, #C084FC);
            border-radius: 999px;
            transition: width 0.5s ease;
        }

        /* Refined Metric Card: Reduced vertical padding and clear hierarchy */
        .metric-card {
            background-color: #141222;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 14px;
            padding: 14px 18px;
            transition: border-color 0.15s ease;
        }
        .metric-card:hover {
            border-color: rgba(192, 132, 252, 0.25);
        }
        .metric-label {
            font-size: 11px;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 3px;
        }
        .metric-value {
            font-size: 22px;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
        }
        .metric-subtext {
            font-size: 12px;
            color: #64748B;
            margin-top: 3px;
        }

        /* Quality Findings Accordion: Compact & Readable */
        details.ds-finding-item {
            background-color: #141222;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            margin-bottom: 8px;
            transition: all 0.15s ease;
            overflow: hidden;
        }
        details.ds-finding-item:hover {
            border-color: rgba(192, 132, 252, 0.22);
            background-color: #161427;
        }
        details.ds-finding-item[open] {
            border-color: rgba(192, 132, 252, 0.3);
            background-color: #161427;
        }
        summary.ds-finding-summary {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 11px 16px;
            cursor: pointer;
            user-select: none;
            gap: 12px;
        }
        summary.ds-finding-summary::-webkit-details-marker {
            display: none;
        }
        .finding-summary-left {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-shrink: 0;
        }
        .badge-fact {
            background: rgba(148, 163, 184, 0.12);
            color: #94A3B8;
            border: 1px solid rgba(148, 163, 184, 0.22);
            font-size: 10.5px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 4px;
            letter-spacing: 0.04em;
        }
        .badge-heuristic {
            background: rgba(192, 132, 252, 0.12);
            color: #C084FC;
            border: 1px solid rgba(192, 132, 252, 0.25);
            font-size: 10.5px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 4px;
            letter-spacing: 0.04em;
        }
        .sev-indicator {
            font-size: 12px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .sev-critical { color: #F87171; }
        .sev-warning { color: #FBBF24; }
        .sev-info { color: #818CF8; }
        .finding-title {
            font-size: 13.5px;
            font-weight: 600;
            color: #F8FAFC;
        }
        .finding-summary-preview {
            font-size: 12.5px;
            color: #64748B;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            text-align: right;
            flex-grow: 1;
        }
        .finding-details-content {
            padding: 12px 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            background-color: #12101F;
            font-size: 13px;
            color: #CBD5E1;
            line-height: 1.55;
        }

        /* Two-Column Panel Card Headers */
        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }
        .panel-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 18px;
            font-weight: 600;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Benchmark List Items */
        .benchmark-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 11px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .benchmark-row:last-child {
            border-bottom: none;
            padding-bottom: 2px;
        }
        .benchmark-model-name {
            font-size: 14px;
            font-weight: 500;
            color: #E2E8F0;
        }
        .benchmark-score-pill {
            background-color: rgba(168, 85, 247, 0.12);
            border: 1px solid rgba(168, 85, 247, 0.22);
            color: #F3E8FF;
            font-weight: 600;
            font-size: 12.5px;
            padding: 3px 10px;
            border-radius: 16px;
            letter-spacing: 0.01em;
        }

        /* Streamlit Input / Selectbox Dark Styling */
        div[data-baseweb="select"] > div {
            background-color: #181528 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }
        div[data-baseweb="select"] * {
            color: #F8FAFC !important;
        }
        div[data-baseweb="popover"] > div {
            background-color: #181528 !important;
            border: 1px solid rgba(168, 85, 247, 0.25) !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: rgba(168, 85, 247, 0.18) !important;
        }

        /* Streamlit Dataframe Dark Refinements */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }

        /* Feature tags */
        .feature-chip {
            display: inline-block;
            background-color: rgba(168, 85, 247, 0.1);
            color: #DDD6FE;
            border: 1px solid rgba(168, 85, 247, 0.2);
            padding: 2px 7px;
            border-radius: 5px;
            font-size: 11.5px;
            font-family: monospace;
            margin-right: 4px;
            margin-bottom: 4px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
