"""Editorial Dark & Lavender design system with Light Mode support and refined modern UI for DataSleuth."""

import streamlit as st


def apply_custom_styles(theme: str = "dark") -> None:
    """Inject polished, modern editorial CSS supporting both Dark & Light themes into the Streamlit app.
    
    Args:
        theme: 'dark' (obsidian & lavender) or 'light' (porcelain & violet).
    """
    is_light = theme.lower() == "light"

    # Theme token mappings
    bg_color = "#F8FAFC" if is_light else "#0A0914"
    sidebar_bg = "#FFFFFF" if is_light else "#07060F"
    sidebar_border = "rgba(226, 232, 240, 0.9)" if is_light else "rgba(255, 255, 255, 0.06)"
    text_primary = "#0F172A" if is_light else "#F8FAFC"
    text_secondary = "#334155" if is_light else "#CBD5E1"
    text_muted = "#64748B" if is_light else "#94A3B8"
    
    card_bg = "#FFFFFF" if is_light else "linear-gradient(180deg, #151325 0%, #110F20 100%)"
    card_border = "rgba(226, 232, 240, 0.95)" if is_light else "rgba(192, 132, 252, 0.14)"
    card_hover_border = "rgba(124, 58, 237, 0.35)" if is_light else "rgba(192, 132, 252, 0.35)"
    card_shadow = (
        "0 10px 25px -4px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.03)"
        if is_light
        else "0 12px 32px -4px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.06)"
    )
    
    accent_primary = "#7C3AED" if is_light else "#C084FC"
    accent_secondary = "#8B5CF6" if is_light else "#A855F7"
    accent_gradient = (
        "linear-gradient(135deg, #7C3AED 0%, #9333EA 100%)"
        if is_light
        else "linear-gradient(135deg, #A855F7 0%, #C084FC 100%)"
    )
    
    pill_btn_bg = "#7C3AED" if is_light else "#DDD6FE"
    pill_btn_text = "#FFFFFF" if is_light else "#1E1B4B"
    pill_btn_hover = "#6D28D9" if is_light else "#EDE9FE"
    
    select_bg = "#FFFFFF" if is_light else "#161327"
    select_border = "#CBD5E1" if is_light else "rgba(255, 255, 255, 0.12)"
    track_bg = "#E2E8F0" if is_light else "#231F3A"
    finding_details_bg = "#F8FAFC" if is_light else "#12101F"
    border_subtle = "rgba(226, 232, 240, 0.8)" if is_light else "rgba(255, 255, 255, 0.06)"

    custom_css = f"""
    <style>
        /* Modern Typography: Plus Jakarta Sans for UI + Playfair Display for editorial headers + JetBrains Mono for metrics */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,600;1,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {{
            --ds-bg: {bg_color};
            --ds-sidebar-bg: {sidebar_bg};
            --ds-text-primary: {text_primary};
            --ds-text-secondary: {text_secondary};
            --ds-text-muted: {text_muted};
            --ds-card-bg: {card_bg};
            --ds-card-border: {card_border};
            --ds-card-hover-border: {card_hover_border};
            --ds-card-shadow: {card_shadow};
            --ds-card-radius: 18px;
            --ds-accent-primary: {accent_primary};
            --ds-accent-secondary: {accent_secondary};
            --ds-accent-gradient: {accent_gradient};
            --ds-pill-btn-bg: {pill_btn_bg};
            --ds-pill-btn-text: {pill_btn_text};
            --ds-pill-btn-hover: {pill_btn_hover};
            --ds-select-bg: {select_bg};
            --ds-select-border: {select_border};
            --ds-track-bg: {track_bg};
            --ds-finding-details-bg: {finding_details_bg};
            --ds-border-subtle: {border_subtle};
        }}

        /* Global application background and text colors */
        .stApp {{
            background-color: var(--ds-bg) !important;
            color: var(--ds-text-primary) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }}

        /* Headings standard text colors */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--ds-text-primary) !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            letter-spacing: -0.02em !important;
        }}
        p, span, label, div {{
            color: inherit;
        }}

        /* Top header bar */
        header[data-testid="stHeader"] {{
            background-color: var(--ds-bg) !important;
            backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid var(--ds-border-subtle) !important;
        }}

        /* Sidebar Styling: Sleek, compact and responsive */
        section[data-testid="stSidebar"] {{
            background-color: var(--ds-sidebar-bg) !important;
            border-right: 1px solid {sidebar_border} !important;
            padding-top: 0.5rem !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 0.75rem !important;
            padding-bottom: 1rem !important;
        }}

        /* Sidebar Brand */
        .sidebar-brand-container {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 10px 16px 10px;
        }}
        .sidebar-logo-icon {{
            width: 32px;
            height: 32px;
            border-radius: 10px;
            background: var(--ds-accent-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3);
        }}
        .sidebar-logo-icon svg {{
            width: 17px;
            height: 17px;
            stroke: #FFFFFF;
            fill: none;
            stroke-width: 2.4;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}
        .sidebar-brand-text {{
            font-size: 19px;
            font-weight: 800;
            color: var(--ds-text-primary);
            letter-spacing: -0.03em;
        }}

        /* Theme Switcher Segmented Control */
        .theme-switcher-box {{
            padding: 4px;
            margin-bottom: 14px;
            border-radius: 12px;
            background-color: var(--ds-card-bg);
            border: 1px solid var(--ds-border-subtle);
        }}

        /* Sidebar Navigation Radio Styling */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
            gap: 5px;
        }}
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
            background-color: transparent !important;
            border-radius: 10px !important;
            padding: 9px 13px !important;
            margin: 0 !important;
            border: 1px solid transparent !important;
            color: var(--ds-text-muted) !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            cursor: pointer !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
            color: var(--ds-text-primary) !important;
            background-color: {"rgba(0,0,0,0.04)" if is_light else "rgba(255, 255, 255, 0.04)"} !important;
        }}
        /* Active Radio Option Pill */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {{
            background: {"rgba(124, 58, 237, 0.09)" if is_light else "rgba(168, 85, 247, 0.14)"} !important;
            border: 1px solid {"rgba(124, 58, 237, 0.28)" if is_light else "rgba(168, 85, 247, 0.3)"} !important;
            color: var(--ds-accent-primary) !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px {"rgba(124, 58, 237, 0.08)" if is_light else "rgba(0, 0, 0, 0.25)"} !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {{
            color: inherit !important;
            font-size: 13.5px !important;
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0 !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioButtonCustom"] {{
            display: none !important;
        }}

        /* Sidebar Profile Footer */
        .sidebar-profile {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            background-color: var(--ds-card-bg);
            border: 1px solid var(--ds-border-subtle);
            margin-top: 18px;
            box-shadow: var(--ds-card-shadow);
        }}
        .profile-avatar {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--ds-accent-gradient);
            color: #FFFFFF;
            font-size: 11px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            position: relative;
        }}
        .profile-name {{
            font-size: 13px;
            color: var(--ds-text-primary);
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex-grow: 1;
        }}
        .profile-chevron {{
            color: var(--ds-text-muted);
            font-size: 11px;
        }}

        /* Hero Header Section */
        .hero-title-container {{
            margin-bottom: 20px;
        }}
        .hero-title {{
            font-family: 'Playfair Display', Georgia, serif !important;
            font-size: 36px;
            font-weight: 700;
            color: var(--ds-text-primary) !important;
            letter-spacing: -0.01em;
            margin: 0;
            line-height: 1.15;
        }}
        .hero-title-italic {{
            font-style: italic;
            font-weight: 700;
            color: var(--ds-accent-primary) !important;
        }}
        .hero-subtitle {{
            font-size: 14.5px;
            color: var(--ds-text-muted) !important;
            margin-top: 5px;
            font-weight: 400;
        }}

        /* + New Audit Pill Button */
        button[key="new_audit_btn"], div.stButton > button[kind="primary"] {{
            background: var(--ds-pill-btn-bg) !important;
            color: var(--ds-pill-btn-text) !important;
            border: none !important;
            font-weight: 600 !important;
            border-radius: 9999px !important;
            padding: 8px 22px !important;
            font-size: 13.5px !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 2px 10px {"rgba(124, 58, 237, 0.25)" if is_light else "rgba(0, 0, 0, 0.25)"} !important;
        }}
        button[key="new_audit_btn"]:hover, div.stButton > button[kind="primary"]:hover {{
            background: var(--ds-pill-btn-hover) !important;
            transform: translateY(-1.5px) !important;
            box-shadow: 0 4px 14px {"rgba(124, 58, 237, 0.35)" if is_light else "rgba(0, 0, 0, 0.4)"} !important;
        }}

        /* Modern Rounded Card Container: 18px squircle radius with top hairline */
        .ds-card {{
            background: var(--ds-card-bg);
            border: 1px solid var(--ds-card-border);
            border-radius: var(--ds-card-radius);
            padding: 20px 22px;
            margin-bottom: 18px;
            box-shadow: var(--ds-card-shadow);
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease;
        }}
        .ds-card:hover {{
            border-color: var(--ds-card-hover-border);
            transform: translateY(-2px);
        }}

        /* Featured Dataset Panel: High visual distinction, gradient top border, pulse dot */
        .ds-featured-card {{
            background: var(--ds-card-bg);
            border: 1px solid var(--ds-card-border);
            border-radius: var(--ds-card-radius);
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: var(--ds-card-shadow);
            position: relative;
            overflow: hidden;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease;
        }}
        .ds-featured-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3.5px;
            background: var(--ds-accent-gradient);
        }}
        .ds-featured-card:hover {{
            border-color: var(--ds-card-hover-border);
            transform: translateY(-2px);
        }}

        /* Pulse / EKG Squircle Icon */
        .pulse-icon-box {{
            width: 46px;
            height: 46px;
            border-radius: 14px;
            background: {"rgba(124, 58, 237, 0.08)" if is_light else "#1A172B"};
            border: 1px solid {"rgba(124, 58, 237, 0.2)" if is_light else "rgba(168, 85, 247, 0.25)"};
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 4px 12px {"rgba(124, 58, 237, 0.12)" if is_light else "rgba(0, 0, 0, 0.2)"};
        }}
        .pulse-icon-box svg {{
            width: 24px;
            height: 24px;
            stroke: var(--ds-accent-primary);
            fill: none;
            stroke-width: 2.2;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}

        /* Featured Card Typography */
        .featured-header-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 3px;
        }}
        .featured-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--ds-text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .live-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 2px 8px;
            background: {"rgba(16, 185, 129, 0.1)" if is_light else "rgba(52, 211, 153, 0.12)"};
            border: 1px solid {"rgba(16, 185, 129, 0.25)" if is_light else "rgba(52, 211, 153, 0.25)"};
            color: {"#059669" if is_light else "#34D399"};
            border-radius: 999px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}
        .status-pulse-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: {"#10B981" if is_light else "#34D399"};
            display: inline-block;
            box-shadow: 0 0 8px {"#10B981" if is_light else "#34D399"};
        }}
        .featured-title {{
            font-family: 'Playfair Display', Georgia, serif !important;
            font-size: 23px;
            font-weight: 600;
            color: var(--ds-text-primary) !important;
            letter-spacing: -0.01em;
            margin-bottom: 4px;
        }}
        .featured-meta {{
            font-size: 13px;
            color: var(--ds-text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Health Score Element */
        .health-score-title {{
            font-size: 12px;
            color: var(--ds-text-muted);
            font-weight: 600;
            margin-bottom: 3px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .health-score-val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 28px;
            font-weight: 700;
            color: var(--ds-accent-primary);
            line-height: 1;
            margin-bottom: 9px;
        }}
        .health-score-val span {{
            color: var(--ds-text-muted);
            font-size: 18px;
            font-weight: 400;
        }}
        .health-progress-track {{
            width: 100%;
            height: 7px;
            background-color: var(--ds-track-bg);
            border-radius: 999px;
            overflow: hidden;
            position: relative;
        }}
        .health-progress-fill {{
            height: 100%;
            background: var(--ds-accent-gradient);
            border-radius: 999px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        /* Refined Metric Card: 18px rounded corners, micro-labels and elevation */
        .metric-card {{
            background: var(--ds-card-bg);
            border: 1px solid var(--ds-card-border);
            border-radius: var(--ds-card-radius);
            padding: 16px 20px;
            box-shadow: var(--ds-card-shadow);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }}
        .metric-card:hover {{
            border-color: var(--ds-card-hover-border);
            transform: translateY(-2px);
        }}
        .metric-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--ds-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .metric-label-dot {{
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: var(--ds-accent-primary);
        }}
        .metric-value {{
            font-family: 'JetBrains Mono', 'Plus Jakarta Sans', monospace;
            font-size: 24px;
            font-weight: 700;
            color: var(--ds-text-primary);
            line-height: 1.2;
        }}
        .metric-subtext {{
            font-size: 12.5px;
            color: var(--ds-text-muted);
            margin-top: 4px;
        }}

        /* Quality Findings Accordion: 14px rounded cards with smooth disclosure */
        details.ds-finding-item {{
            background: var(--ds-card-bg);
            border: 1px solid var(--ds-card-border);
            border-radius: 14px;
            margin-bottom: 9px;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            overflow: hidden;
            box-shadow: var(--ds-card-shadow);
        }}
        details.ds-finding-item:hover {{
            border-color: var(--ds-card-hover-border);
        }}
        details.ds-finding-item[open] {{
            border-color: var(--ds-card-hover-border);
        }}
        summary.ds-finding-summary {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 13px 18px;
            cursor: pointer;
            user-select: none;
            gap: 14px;
        }}
        summary.ds-finding-summary::-webkit-details-marker {{
            display: none;
        }}
        .finding-summary-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-shrink: 0;
        }}
        .badge-fact {{
            background: {"rgba(15, 23, 42, 0.06)" if is_light else "rgba(148, 163, 184, 0.12)"};
            color: var(--ds-text-secondary);
            border: 1px solid {"rgba(15, 23, 42, 0.12)" if is_light else "rgba(148, 163, 184, 0.22)"};
            font-size: 10.5px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            letter-spacing: 0.05em;
        }}
        .badge-heuristic {{
            background: {"rgba(124, 58, 237, 0.09)" if is_light else "rgba(192, 132, 252, 0.12)"};
            color: var(--ds-accent-primary);
            border: 1px solid {"rgba(124, 58, 237, 0.25)" if is_light else "rgba(192, 132, 252, 0.25)"};
            font-size: 10.5px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            letter-spacing: 0.05em;
        }}
        .sev-indicator {{
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .sev-critical {{ color: #EF4444; }}
        .sev-warning {{ color: #F59E0B; }}
        .sev-info {{ color: #6366F1; }}
        .finding-title {{
            font-size: 14px;
            font-weight: 600;
            color: var(--ds-text-primary);
        }}
        .finding-summary-preview {{
            font-size: 13px;
            color: var(--ds-text-muted);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            text-align: right;
            flex-grow: 1;
        }}
        .finding-details-content {{
            padding: 14px 18px;
            border-top: 1px solid var(--ds-border-subtle);
            background: var(--ds-finding-details-bg);
            font-size: 13.5px;
            color: var(--ds-text-secondary);
            line-height: 1.6;
        }}

        /* Two-Column Panel Card Headers */
        .panel-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }}
        .panel-title {{
            font-family: 'Playfair Display', Georgia, serif !important;
            font-size: 19px;
            font-weight: 600;
            color: var(--ds-text-primary) !important;
            display: flex;
            align-items: center;
            gap: 9px;
        }}

        /* Benchmark Ranked Capsule Rows */
        .benchmark-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            margin-bottom: 6px;
            border-radius: 12px;
            background: {"rgba(15, 23, 42, 0.02)" if is_light else "rgba(255, 255, 255, 0.02)"};
            border: 1px solid var(--ds-border-subtle);
            transition: all 0.2s ease;
        }}
        .benchmark-row:hover {{
            background: {"rgba(124, 58, 237, 0.04)" if is_light else "rgba(168, 85, 247, 0.06)"};
            border-color: var(--ds-card-hover-border);
            transform: translateX(2px);
        }}
        .benchmark-left {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .benchmark-rank-badge {{
            width: 22px;
            height: 22px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            background: {"rgba(124, 58, 237, 0.1)" if is_light else "rgba(168, 85, 247, 0.14)"};
            color: var(--ds-accent-primary);
        }}
        .benchmark-model-name {{
            font-size: 14px;
            font-weight: 600;
            color: var(--ds-text-primary);
        }}
        .benchmark-score-pill {{
            background: {"rgba(124, 58, 237, 0.08)" if is_light else "rgba(168, 85, 247, 0.12)"};
            border: 1px solid {"rgba(124, 58, 237, 0.22)" if is_light else "rgba(168, 85, 247, 0.24)"};
            color: var(--ds-accent-primary);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 13px;
            padding: 3px 10px;
            border-radius: 999px;
            letter-spacing: 0.01em;
        }}

        /* Streamlit Input & Selectbox Styling */
        div[data-baseweb="select"] > div {{
            background-color: var(--ds-select-bg) !important;
            border: 1px solid var(--ds-select-border) !important;
            border-radius: 10px !important;
            color: var(--ds-text-primary) !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }}
        div[data-baseweb="select"] * {{
            color: var(--ds-text-primary) !important;
        }}
        div[data-baseweb="popover"] > div {{
            background-color: var(--ds-select-bg) !important;
            border: 1px solid var(--ds-card-hover-border) !important;
            border-radius: 10px !important;
            box-shadow: var(--ds-card-shadow) !important;
        }}
        div[data-baseweb="popover"] li:hover {{
            background-color: {"rgba(124, 58, 237, 0.08)" if is_light else "rgba(168, 85, 247, 0.18)"} !important;
        }}

        /* Streamlit Dataframe Refinements */
        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--ds-border-subtle) !important;
            border-radius: var(--ds-card-radius) !important;
            overflow: hidden !important;
            box-shadow: var(--ds-card-shadow) !important;
        }}

        /* Feature chips */
        .feature-chip {{
            display: inline-block;
            background-color: {"rgba(124, 58, 237, 0.08)" if is_light else "rgba(168, 85, 247, 0.1)"};
            color: var(--ds-accent-primary);
            border: 1px solid {"rgba(124, 58, 237, 0.2)" if is_light else "rgba(168, 85, 247, 0.2)"};
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11.5px;
            font-family: 'JetBrains Mono', monospace;
            margin-right: 5px;
            margin-bottom: 5px;
        }}

        /* View Headings */
        .view-title {{
            font-family: 'Playfair Display', Georgia, serif !important;
            font-size: 32px;
            font-weight: 700;
            color: var(--ds-text-primary) !important;
            margin: 0;
            line-height: 1.2;
        }}
        .view-subtitle {{
            font-size: 14px;
            color: var(--ds-text-muted) !important;
            margin-top: 4px;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
