"""Exploratory Data Analysis (EDA): Interactive Plotly statistical visualizations in dark & light themes."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datasleuth.config import CONFIG


def get_current_theme() -> str:
    """Detect active Streamlit UI theme ('dark' or 'light')."""
    try:
        import streamlit as st
        return st.session_state.get("theme", "dark")
    except Exception:
        return "dark"


def get_plotly_layout(theme: str | None = None) -> dict[str, Any]:
    """Return responsive, editorial layout tokens adapting to the active theme."""
    t = theme or get_current_theme()
    is_light = t == "light"

    return {
        "font": {
            "family": "Plus Jakarta Sans, Inter, -apple-system, sans-serif",
            "color": "#475569" if is_light else "#94A3B8",
        },
        "title_font": {
            "size": 15,
            "color": "#0F172A" if is_light else "#F8FAFC",
            "family": "Playfair Display, Georgia, serif",
        },
        "margin": {"l": 45, "r": 25, "t": 40, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis": {
            "gridcolor": "rgba(0, 0, 0, 0.06)" if is_light else "rgba(255, 255, 255, 0.06)",
            "linecolor": "rgba(0, 0, 0, 0.12)" if is_light else "rgba(255, 255, 255, 0.15)",
            "tickcolor": "rgba(0, 0, 0, 0.12)" if is_light else "rgba(255, 255, 255, 0.15)",
        },
        "yaxis": {
            "gridcolor": "rgba(0, 0, 0, 0.06)" if is_light else "rgba(255, 255, 255, 0.06)",
            "linecolor": "rgba(0, 0, 0, 0.12)" if is_light else "rgba(255, 255, 255, 0.15)",
            "tickcolor": "rgba(0, 0, 0, 0.12)" if is_light else "rgba(255, 255, 255, 0.15)",
        },
    }


def get_bar_color(theme: str | None = None) -> str:
    """Return primary bar fill color matching current theme."""
    t = theme or get_current_theme()
    return CONFIG.COLOR_LAVENDER_BAR_LIGHT if t == "light" else CONFIG.COLOR_LAVENDER_BAR_DARK


def get_plotly_template(theme: str | None = None) -> str:
    """Return plotly template name."""
    t = theme or get_current_theme()
    return "plotly_white" if t == "light" else "plotly_dark"


def get_theme_palette(theme: str | None = None) -> tuple[str, ...]:
    """Return discrete color palette for categorical variables."""
    t = theme or get_current_theme()
    return CONFIG.COLOR_PALETTE_LIGHT if t == "light" else CONFIG.COLOR_PALETTE_DARK


def sample_for_plotting(
    df: pd.DataFrame, max_rows: int = CONFIG.MAX_SCATTER_POINTS
) -> tuple[pd.DataFrame, bool, int]:
    """Sample DataFrame safely if row count exceeds plotting limit.
    
    Returns:
        (sampled_df, is_sampled, original_row_count)
    """
    total_rows = len(df)
    if total_rows > max_rows:
        sampled_df = df.sample(n=max_rows, random_state=42)
        return sampled_df, True, total_rows
    return df, False, total_rows


def plot_overview_feature_distribution(
    df: pd.DataFrame, column: str, theme: str | None = None
) -> go.Figure:
    """Create the signature clean lavender/violet bar chart for the Overview page."""
    t = theme or get_current_theme()
    bar_fill = get_bar_color(t)
    layout_base = get_plotly_layout(t)

    series = df[column].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series) and series.nunique() > 7

    if is_numeric:
        n_bins = 7
        try:
            cats, bin_edges = pd.cut(series, bins=n_bins, retbins=True)
            counts = cats.value_counts(sort=False)
            
            labels = []
            for i in range(len(bin_edges) - 1):
                low = int(round(bin_edges[i]))
                high = int(round(bin_edges[i+1]))
                labels.append(f"{low}–{high}")
            
            x_vals = labels
            y_vals = counts.values
        except Exception:
            x_vals = series
            y_vals = None
    else:
        counts = series.astype(str).value_counts().head(8)
        x_vals = list(counts.index)
        y_vals = list(counts.values)

    fig = go.Figure()

    if y_vals is not None:
        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=bar_fill,
            marker_line_width=0,
            opacity=0.92,
            hovertemplate="<b>%{x}</b><br>Frequency: %{y:,}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Histogram(
            x=x_vals,
            nbinsx=15,
            marker_color=bar_fill,
            opacity=0.92,
            hovertemplate="Frequency: %{y:,}<extra></extra>",
        ))

    merged_layout = {**layout_base, "margin": {"l": 40, "r": 20, "t": 20, "b": 40}}
    fig.update_layout(
        **merged_layout,
        bargap=0.15,
        xaxis_title=column,
        yaxis_title="Frequency",
        height=280,
    )
    return fig


def plot_numerical_distribution(
    df: pd.DataFrame, column: str, show_box: bool = True, theme: str | None = None
) -> go.Figure:
    """Create an interactive histogram with KDE/box marginal for a numerical column."""
    t = theme or get_current_theme()
    series = df[column].dropna()
    fig = px.histogram(
        df,
        x=column,
        marginal="box" if show_box else None,
        nbins=min(35, max(10, int(np.sqrt(len(series))))),
        color_discrete_sequence=[get_bar_color(t)],
        template=get_plotly_template(t),
        title=f"Distribution of '{column}'",
    )
    fig.update_layout(
        **get_plotly_layout(t),
        bargap=0.10,
        xaxis_title=column,
        yaxis_title="Count",
    )
    return fig


def plot_categorical_distribution(
    df: pd.DataFrame, column: str, max_categories: int = 15, theme: str | None = None
) -> go.Figure:
    """Create a frequency bar chart for a categorical column."""
    t = theme or get_current_theme()
    counts = df[column].value_counts(dropna=False).head(max_categories)
    total = len(df)
    
    cat_df = pd.DataFrame({
        "Category": [str(c) if pd.notna(c) else "(Missing)" for c in counts.index],
        "Count": counts.values,
        "Percentage": (counts.values / total) * 100.0,
    }).sort_values(by="Count", ascending=True)
    
    fig = px.bar(
        cat_df,
        y="Category",
        x="Count",
        orientation="h",
        text=cat_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
        color_discrete_sequence=[get_bar_color(t)],
        template=get_plotly_template(t),
        title=f"Value Frequencies: '{column}' (Top {len(cat_df)})",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **get_plotly_layout(t),
        xaxis_title="Record Count",
        yaxis_title="Category",
    )
    return fig


def plot_missingness_summary(df: pd.DataFrame, theme: str | None = None) -> go.Figure:
    """Create a missing value bar chart across all columns."""
    t = theme or get_current_theme()
    missing = df.isna().sum()
    pct = (missing / len(df)) * 100.0
    
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing.values,
        "Missing_Pct": pct.values,
    }).sort_values(by="Missing_Pct", ascending=False)
    
    has_missing = missing_df["Missing_Count"].sum() > 0
    display_df = missing_df[missing_df["Missing_Count"] > 0] if has_missing else missing_df.head(15)
    
    primary_c = get_bar_color(t)
    colors = [
        CONFIG.COLOR_DANGER if p >= CONFIG.SEVERE_MISSING_WARN_PCT
        else CONFIG.COLOR_WARNING if p >= CONFIG.HIGH_MISSING_WARN_PCT
        else primary_c
        for p in display_df["Missing_Pct"]
    ]
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=display_df["Column"],
                y=display_df["Missing_Pct"],
                text=display_df["Missing_Pct"].apply(lambda p: f"{p:.1f}%"),
                textposition="outside",
                marker_color=colors,
                hovertext=[
                    f"{col}: {cnt:,} missing ({p:.2f}%)"
                    for col, cnt, p in zip(display_df["Column"], display_df["Missing_Count"], display_df["Missing_Pct"])
                ],
                hoverinfo="text",
            )
        ]
    )
    title_text = "Missing Values by Column (% of total records)" if has_missing else "Missing Values: All Columns 100% Complete"
    fig.update_layout(
        **get_plotly_layout(t),
        title=title_text,
        xaxis_title="Column",
        yaxis_title="Missing (%)",
        yaxis_range=[0, max(10, display_df["Missing_Pct"].max() * 1.15 if has_missing else 10)],
    )
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, method: str = "pearson", min_threshold: float = 0.0, theme: str | None = None
) -> go.Figure:
    """Generate an interactive correlation matrix for numerical features."""
    t = theme or get_current_theme()
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Fewer than 2 numerical columns found; correlation heatmap requires at least 2 numerical features.",
            showarrow=False,
            font={"size": 14, "color": "#94A3B8"},
        )
        fig.update_layout(**get_plotly_layout(t), title="Correlation Heatmap")
        return fig

    if num_df.shape[1] > CONFIG.CORRELATION_MAX_COLUMNS:
        variances = num_df.var()
        top_cols = variances.nlargest(CONFIG.CORRELATION_MAX_COLUMNS).index
        num_df = num_df[top_cols]

    corr = num_df.corr(method=method)

    if min_threshold > 0.0:
        mask = corr.abs() < min_threshold
        corr[mask] = np.nan

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Purples",
        zmin=-1.0,
        zmax=1.0,
        template=get_plotly_template(t),
        title=f"Feature Correlation Heatmap ({method.capitalize()} Method)",
    )
    fig.update_layout(
        **get_plotly_layout(t),
        coloraxis_colorbar={"title": "r"},
    )
    return fig


def plot_target_distribution(
    df: pd.DataFrame, target_col: str, task: str, theme: str | None = None
) -> go.Figure:
    """Plot the distribution of the target variable based on task type."""
    t = theme or get_current_theme()
    if task == "classification":
        counts = df[target_col].value_counts(dropna=False)
        total = len(df)
        target_df = pd.DataFrame({
            "Class": [str(c) if pd.notna(c) else "(Missing)" for c in counts.index],
            "Count": counts.values,
            "Percentage": (counts.values / total) * 100.0,
        })
        fig = px.bar(
            target_df,
            x="Class",
            y="Count",
            text=target_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
            color="Class",
            color_discrete_sequence=get_theme_palette(t),
            template=get_plotly_template(t),
            title=f"Target Distribution: '{target_col}' (Classification)",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            **get_plotly_layout(t),
            xaxis_title="Target Class",
            yaxis_title="Record Count",
            showlegend=False,
        )
        return fig
    else:
        fig = px.histogram(
            df,
            x=target_col,
            marginal="box",
            nbins=30,
            color_discrete_sequence=[get_bar_color(t)],
            template=get_plotly_template(t),
            title=f"Target Distribution: '{target_col}' (Regression)",
        )
        fig.update_layout(
            **get_plotly_layout(t),
            bargap=0.08,
            xaxis_title=target_col,
            yaxis_title="Frequency",
        )
        return fig


def plot_feature_vs_target(
    df: pd.DataFrame, feature_col: str, target_col: str, task: str, theme: str | None = None
) -> go.Figure:
    """Generate the optimal interactive comparison plot between a feature and target."""
    t = theme or get_current_theme()
    sampled_df, is_sampled, total_rows = sample_for_plotting(df)
    is_feat_num = pd.api.types.is_numeric_dtype(df[feature_col])
    sample_notice = f" [Sampled {len(sampled_df):,} of {total_rows:,} rows]" if is_sampled else ""

    if task == "classification":
        if is_feat_num:
            fig = px.box(
                sampled_df,
                x=target_col,
                y=feature_col,
                color=target_col,
                color_discrete_sequence=get_theme_palette(t),
                template=get_plotly_template(t),
                title=f"'{feature_col}' across Target Classes '{target_col}'{sample_notice}",
            )
            fig.update_layout(**get_plotly_layout(t), showlegend=False)
            return fig
        else:
            cross_tab = pd.crosstab(
                sampled_df[feature_col].astype(str),
                sampled_df[target_col].astype(str),
                normalize="index"
            ) * 100.0
            
            ct_df = cross_tab.reset_index().melt(
                id_vars=feature_col, var_name=target_col, value_name="Percentage"
            )
            
            fig = px.bar(
                ct_df,
                x=feature_col,
                y="Percentage",
                color=target_col,
                barmode="stack",
                color_discrete_sequence=get_theme_palette(t),
                template=get_plotly_template(t),
                title=f"Composition of '{target_col}' by '{feature_col}' (%){sample_notice}",
            )
            fig.update_layout(**get_plotly_layout(t), yaxis_title="Percentage (%)")
            return fig
    else:
        if is_feat_num:
            fig = px.scatter(
                sampled_df,
                x=feature_col,
                y=target_col,
                opacity=0.7,
                color_discrete_sequence=[get_bar_color(t)],
                trendline="ols",
                template=get_plotly_template(t),
                title=f"'{feature_col}' vs Target '{target_col}' (OLS Trendline){sample_notice}",
            )
            fig.update_layout(**get_plotly_layout(t))
            return fig
        else:
            fig = px.box(
                sampled_df,
                x=feature_col,
                y=target_col,
                color=feature_col,
                color_discrete_sequence=get_theme_palette(t),
                template=get_plotly_template(t),
                title=f"Target '{target_col}' distribution across '{feature_col}'{sample_notice}",
            )
            fig.update_layout(**get_plotly_layout(t), showlegend=False)
            return fig


def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    theme: str | None = None,
) -> tuple[go.Figure, bool, int]:
    """Bivariate scatter explorer with optional hue and sizing."""
    t = theme or get_current_theme()
    sampled_df, is_sampled, total_rows = sample_for_plotting(df)
    
    kwargs: dict[str, Any] = {
        "data_frame": sampled_df,
        "x": x_col,
        "y": y_col,
        "template": get_plotly_template(t),
        "opacity": 0.75,
    }
    if color_col and color_col in df.columns:
        kwargs["color"] = color_col
        if pd.api.types.is_numeric_dtype(df[color_col]):
            kwargs["color_continuous_scale"] = "Purples"
        else:
            kwargs["color_discrete_sequence"] = get_theme_palette(t)
            
    if size_col and size_col in df.columns and pd.api.types.is_numeric_dtype(df[size_col]):
        kwargs["size"] = size_col
        kwargs["size_max"] = 16

    fig = px.scatter(**kwargs)
    sample_notice = f" [Sampled: {len(sampled_df):,} of {total_rows:,} rows]" if is_sampled else ""
    fig.update_layout(
        **get_plotly_layout(t),
        title=f"Bivariate Explorer: '{x_col}' vs '{y_col}'{sample_notice}",
    )
    return fig, is_sampled, total_rows
