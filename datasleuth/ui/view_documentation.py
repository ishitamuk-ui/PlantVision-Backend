"""Documentation view: Technical guide, mathematical definitions, and methodology."""

import streamlit as st


def render_documentation_view() -> None:
    """Render the technical Documentation page in editorial dark lavender styling."""
    st.markdown(
        """
        <div style="margin-bottom: 28px;">
            <h1 style="font-family: 'Playfair Display', Georgia, serif; font-size: 34px; font-weight: 700; color: #FFFFFF; margin: 0;">
                Documentation & Methodology
            </h1>
            <div style="font-size: 14px; color: #94A3B8; margin-top: 6px;">
                Technical specifications, mathematical formulas, and algorithmic design principles behind DataSleuth.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    doc_tabs = st.tabs([
        "1. Ingestion & Preprocessing",
        "2. Quality Engine & IQR Outliers",
        "3. Leakage & Imbalance Heuristics",
        "4. ML Benchmarking Pipelines",
        "5. Reports & Exports",
    ])

    # Tab 1: Ingestion & Preprocessing
    with doc_tabs[0]:
        st.markdown(
            """
            <div class="ds-card">
                <h3 style="color: #FFFFFF; font-family: 'Playfair Display', serif; margin-top: 0;">Dataset Ingestion & Safety Safeguards</h3>
                <p style="color: #CBD5E1; font-size: 14px; line-height: 1.6;">
                    DataSleuth enforces strict parsing security and structural validation before loading tabular data into memory:
                </p>
                <ul style="color: #94A3B8; font-size: 14px; line-height: 1.8;">
                    <li><strong>Delimiter Sniffing:</strong> Automatically samples the file to detect comma (<code>,</code>), semicolon (<code>;</code>), tab (<code>\\t</code>), and pipe (<code>|</code>) delimiters.</li>
                    <li><strong>Multi-Encoding Resolution:</strong> Tries <code>utf-8</code>, falling back to <code>latin-1</code>, <code>cp1252</code>, and <code>iso-8859-1</code> if byte decode errors occur.</li>
                    <li><strong>Column Sanitization:</strong> Strips whitespace, auto-renames empty headers, and deduplicates identical column names (e.g. <code>col</code>, <code>col_1</code>).</li>
                    <li><strong>Zero Execution:</strong> Files are parsed purely as static CSV tables without evaluating arbitrary code or macros.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 2: Quality Engine & IQR Outliers
    with doc_tabs[1]:
        st.markdown(
            """
            <div class="ds-card">
                <h3 style="color: #FFFFFF; font-family: 'Playfair Display', serif; margin-top: 0;">Tukey's Interquartile Range (IQR) Methodology</h3>
                <p style="color: #CBD5E1; font-size: 14px; line-height: 1.6;">
                    Outliers are evaluated using Tukey's fences over the empirical distribution:
                </p>
                <div style="background-color: #1B182E; border-left: 3px solid #C084FC; padding: 14px 18px; border-radius: 6px; margin: 16px 0; font-family: monospace; font-size: 13px; color: #DDD6FE;">
                    IQR = Q3 (75th percentile) - Q1 (25th percentile)<br>
                    Lower Fence = Q1 - 1.5 &times; IQR<br>
                    Upper Fence = Q3 + 1.5 &times; IQR
                </div>
                <h4 style="color: #F8FAFC; font-size: 16px; margin-top: 18px;">Empirical Facts vs. Heuristic Warnings</h4>
                <p style="color: #94A3B8; font-size: 14px; line-height: 1.6;">
                    DataSleuth strictly demarcates <em>confirmed facts</em> (such as verified null cells and duplicate rows) from <em>heuristic warnings</em> (such as extreme values). In real-world data science, legitimate heavy-tailed observations (e.g., high net-worth individuals or rare medical indicators) should not be discarded blindly.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 3: Leakage & Imbalance Heuristics
    with doc_tabs[2]:
        st.markdown(
            """
            <div class="ds-card">
                <h3 style="color: #FFFFFF; font-family: 'Playfair Display', serif; margin-top: 0;">Automated Data Leakage Detection</h3>
                <p style="color: #CBD5E1; font-size: 14px; line-height: 1.6;">
                    Data leakage produces artificial over-optimism in evaluation metrics. DataSleuth runs two automated heuristic detectors:
                </p>
                <ul style="color: #94A3B8; font-size: 14px; line-height: 1.8;">
                    <li><strong>Identifier Leakage:</strong> Flags features with 100% uniqueness matching ID keywords (e.g. <code>id</code>, <code>uuid</code>, <code>patient_id</code>). Training models on row IDs induces severe memorization.</li>
                    <li><strong>Target Collinearity:</strong> Flags numerical features with absolute Pearson correlation <code>|r| &ge; 0.98</code> with the target, indicating potential post-event recording or duplicate target columns.</li>
                    <li><strong>Class Imbalance:</strong> Flags classification targets where the minority class accounts for <code>&lt; 15%</code> of observations, warning that basic accuracy scores can be deceptive.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 4: ML Benchmarking Pipelines
    with doc_tabs[3]:
        st.markdown(
            """
            <div class="ds-card">
                <h3 style="color: #FFFFFF; font-family: 'Playfair Display', serif; margin-top: 0;">Leak-Free Preprocessing & Estimators</h3>
                <p style="color: #CBD5E1; font-size: 14px; line-height: 1.6;">
                    To guarantee reproducible, unpolluted model benchmarks:
                </p>
                <ul style="color: #94A3B8; font-size: 14px; line-height: 1.8;">
                    <li><strong>Split Isolation:</strong> Datasets are partitioned before preprocessing. Imputers and scalers fit exclusively on the training split and transform the test split.</li>
                    <li><strong>Classification Baselines:</strong> <code>DummyClassifier (Stratified)</code>, <code>LogisticRegression (L2)</code>, <code>DecisionTreeClassifier</code>, and <code>RandomForestClassifier</code>.</li>
                    <li><strong>Regression Baselines:</strong> <code>DummyRegressor (Mean)</code>, <code>RidgeRegression</code>, and <code>RandomForestRegressor</code>.</li>
                    <li><strong>Metrics:</strong> Evaluated on hold-out test data with Weighted F1, Macro F1, Accuracy, Precision, Recall, Confusion Matrix, RMSE, MAE, and R&sup2;.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 5: Reports & Exports
    with doc_tabs[4]:
        st.markdown(
            """
            <div class="ds-card">
                <h3 style="color: #FFFFFF; font-family: 'Playfair Display', serif; margin-top: 0;">Exporting Reports & Metadata</h3>
                <p style="color: #CBD5E1; font-size: 14px; line-height: 1.6;">
                    Audit results and benchmark scores can be downloaded locally at any time:
                </p>
                <ul style="color: #94A3B8; font-size: 14px; line-height: 1.8;">
                    <li><strong>Standalone HTML Audit Report:</strong> A fully styled, responsive HTML file with inline CSS (no external network or CDN dependencies required) containing executive summary, health scores, quality tables, and benchmark leaderboards.</li>
                    <li><strong>Data Quality Table (CSV):</strong> Complete column-by-column inventory of missingness, outliers, and heuristic flags.</li>
                    <li><strong>Audit Metadata (JSON):</strong> Machine-readable JSON export for downstream CI/CD or data pipelines.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
