"""Reporting & Export: Standalone HTML and JSON audit report generation."""

import datetime
import json
from typing import Any

import pandas as pd

from datasleuth.modeling import BenchmarkSuiteResult
from datasleuth.quality import DatasetQualityReport


def generate_json_report(
    quality_report: DatasetQualityReport,
    benchmark_result: BenchmarkSuiteResult | None = None,
) -> str:
    """Generate a structured, complete JSON representation of the audit."""
    data: dict[str, Any] = {
        "generated_at": datetime.datetime.now().isoformat(),
        "platform": "DataSleuth",
        "quality_audit": quality_report.to_dict(),
    }
    if benchmark_result:
        data["ml_benchmarks"] = {
            "task": benchmark_result.task,
            "target": benchmark_result.target_column,
            "best_model": benchmark_result.best_model_name,
            "primary_metric": benchmark_result.primary_metric,
            "leaderboard": benchmark_result.get_leaderboard_df().to_dict(orient="records"),
        }
    return json.dumps(data, indent=2)


def generate_html_report(
    dataset_name: str,
    df: pd.DataFrame,
    quality_report: DatasetQualityReport,
    benchmark_result: BenchmarkSuiteResult | None = None,
) -> str:
    """Generate a self-contained, publication-ready HTML audit report with inline CSS."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Health score badge color
    hs = quality_report.health_score
    if hs >= 80:
        hs_color = "#059669"
        hs_label = "Good Health"
    elif hs >= 60:
        hs_color = "#D97706"
        hs_label = "Fair / Needs Attention"
    else:
        hs_color = "#DC2626"
        hs_label = "At Risk / Poor Quality"

    # Diagnostics HTML
    diags_html = ""
    for d in quality_report.diagnostics:
        badge_cls = "badge-info" if d.severity == "info" else "badge-warning" if d.severity == "warning" else "badge-critical"
        aff_cols = f"<div class='affected-cols'><strong>Affected Columns:</strong> {', '.join(d.affected_columns)}</div>" if d.affected_columns else ""
        diags_html += f"""
        <div class="card diagnostic-card">
            <div class="diagnostic-header">
                <span class="badge {badge_cls}">{d.kind.upper()} &bull; {d.severity.upper()}</span>
                <span class="diagnostic-title">{d.title}</span>
            </div>
            <div class="diagnostic-body">{d.message}</div>
            {aff_cols}
        </div>
        """

    # Column table HTML
    col_df = quality_report.to_dataframe()
    table_rows = ""
    for _, row in col_df.iterrows():
        table_rows += "<tr>"
        for val in row:
            table_rows += f"<td>{val}</td>"
        table_rows += "</tr>"

    table_headers = "".join([f"<th>{c}</th>" for c in col_df.columns])

    # ML Benchmark section HTML
    ml_html = ""
    if benchmark_result:
        lb_df = benchmark_result.get_leaderboard_df()
        lb_headers = "".join([f"<th>{c}</th>" for c in lb_df.columns])
        lb_rows = ""
        for _, r in lb_df.iterrows():
            is_best = (r["Model"] == benchmark_result.best_model_name)
            row_cls = "best-model-row" if is_best else ""
            best_tag = " <span class='badge badge-success'>Best Baseline</span>" if is_best else ""
            lb_rows += f"<tr class='{row_cls}'>"
            for col_idx, (cname, val) in enumerate(r.items()):
                cell_content = f"{val}{best_tag}" if (col_idx == 0 and is_best) else f"{val}"
                lb_rows += f"<td>{cell_content}</td>"
            lb_rows += "</tr>"

        ml_html = f"""
        <section class="section">
            <div class="section-title">Machine Learning Benchmark Leaderboard</div>
            <p class="section-desc">
                Task: <strong>{benchmark_result.task.capitalize()}</strong> &bull; 
                Target: <code>{benchmark_result.target_column}</code> &bull; 
                Features Evaluated: <strong>{len(benchmark_result.feature_columns)}</strong> &bull; 
                Train / Test Split: <strong>{benchmark_result.train_rows:,} / {benchmark_result.test_rows:,} ({benchmark_result.test_split_pct:.0f}% Test)</strong>
            </p>
            <div class="table-container">
                <table class="report-table">
                    <thead><tr>{lb_headers}</tr></thead>
                    <tbody>{lb_rows}</tbody>
                </table>
            </div>
            <div class="model-note">
                <strong>Best Baseline:</strong> {benchmark_result.best_model_name} achieved the top performance on {benchmark_result.primary_metric}.
                Remember to verify class distributions and potential data leakage before proceeding to production training.
            </div>
        </section>
        """
    else:
        ml_html = """
        <section class="section">
            <div class="section-title">Machine Learning Benchmarks</div>
            <p class="section-desc">No baseline models were executed for this audit session.</p>
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataSleuth Audit Report: {dataset_name}</title>
    <style>
        :root {{
            --primary: #2563EB;
            --dark: #0F172A;
            --slate: #334155;
            --light-slate: #64748B;
            --bg: #F8FAFC;
            --card-bg: #FFFFFF;
            --border: #E2E8F0;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--slate);
            line-height: 1.5;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .header-title {{
            font-size: 26px;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 6px;
        }}
        .header-meta {{
            color: var(--light-slate);
            font-size: 14px;
        }}
        .health-card {{
            text-align: right;
        }}
        .health-score {{
            font-size: 42px;
            font-weight: 800;
            color: {hs_color};
            line-height: 1;
        }}
        .health-label {{
            font-size: 13px;
            font-weight: 600;
            color: {hs_color};
            margin-top: 4px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 600;
            color: var(--light-slate);
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: 700;
            color: var(--dark);
        }}
        .section {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 6px;
        }}
        .section-desc {{
            font-size: 14px;
            color: var(--light-slate);
            margin-bottom: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .badge-info {{ background: #EFF6FF; color: #1D4ED8; }}
        .badge-warning {{ background: #FEF3C7; color: #B45309; }}
        .badge-critical {{ background: #FEE2E2; color: #B91C1C; }}
        .badge-success {{ background: #DCFCE7; color: #15803D; }}
        .card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .diagnostic-card {{
            background: #FAFAFA;
        }}
        .diagnostic-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
        }}
        .diagnostic-title {{
            font-weight: 600;
            color: var(--dark);
            font-size: 15px;
        }}
        .diagnostic-body {{
            font-size: 14px;
            color: var(--slate);
        }}
        .affected-cols {{
            margin-top: 8px;
            font-size: 13px;
            color: var(--light-slate);
        }}
        .table-container {{
            overflow-x: auto;
            margin-top: 12px;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        .report-table th {{
            background: #F1F5F9;
            color: var(--dark);
            font-weight: 600;
            padding: 10px 14px;
            border-bottom: 2px solid var(--border);
            white-space: nowrap;
        }}
        .report-table td {{
            padding: 9px 14px;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        .report-table tr:hover {{
            background: #F8FAFC;
        }}
        .best-model-row {{
            background: #F0FDF4;
            font-weight: 600;
        }}
        .model-note {{
            margin-top: 16px;
            padding: 12px 16px;
            background: #F8FAFC;
            border-left: 4px solid var(--primary);
            border-radius: 4px;
            font-size: 13px;
        }}
        .footer {{
            text-align: center;
            color: var(--light-slate);
            font-size: 13px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div>
                <h1 class="header-title">DataSleuth Audit Report</h1>
                <div class="header-meta">
                    Dataset: <strong>{dataset_name}</strong> &bull; Generated: {now_str} &bull; DataSleuth v0.1.0
                </div>
            </div>
            <div class="health-card">
                <div class="health-score">{hs}/100</div>
                <div class="health-label">{hs_label}</div>
            </div>
        </header>

        <div class="grid">
            <div class="stat-card">
                <div class="stat-label">Total Rows</div>
                <div class="stat-value">{quality_report.row_count:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Columns</div>
                <div class="stat-value">{quality_report.column_count:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Missing Cells</div>
                <div class="stat-value">{quality_report.total_missing_cells:,} ({quality_report.overall_missing_pct:.2f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Duplicate Rows</div>
                <div class="stat-value">{quality_report.duplicate_rows_count:,} ({quality_report.duplicate_rows_pct:.2f}%)</div>
            </div>
        </div>

        <section class="section">
            <div class="section-title">Quality Findings & Diagnostic Alerts</div>
            <p class="section-desc">Empirical findings and heuristic warnings discovered during comprehensive data auditing.</p>
            {diags_html}
        </section>

        {ml_html}

        <section class="section">
            <div class="section-title">Column-by-Column Data Quality Metrics</div>
            <p class="section-desc">Exhaustive statistical inventory, missingness rates, and IQR outlier boundaries per feature.</p>
            <div class="table-container">
                <table class="report-table">
                    <thead><tr>{table_headers}</tr></thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </section>

        <footer class="footer">
            Generated autonomously by DataSleuth &bull; Automated Dataset Auditing & ML Benchmarking Platform
        </footer>
    </div>
</body>
</html>
"""
    return html
