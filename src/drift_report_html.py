"""
drift_report_html.py — Generates a self-contained HTML drift report.
=====================================================================

Reads drift_reports/history.json (or history.csv) and produces a single
HTML file with:
- Batch summary table with severity color coding
- Trend charts for alerts, feature drift, score drift, null spikes
- Highlight of worst batch
- Model/schema version visibility

Dependencies: stdlib only (json, os, csv, html).
The output HTML uses Chart.js from CDN for lightweight charts.
"""

import csv
import html as _html
import json
import logging
import os
import re as _re
from datetime import UTC, datetime

log = logging.getLogger(__name__)

_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "drift_reports")


def _js_json(obj) -> str:
    """json.dumps safe for inline <script> blocks — escapes </ to prevent </script> injection."""
    return _re.sub(r"</", r"<\\/", json.dumps(obj))


def _load_history(reports_dir=None):
    """Load history from JSON or CSV, preferring JSON."""
    directory = reports_dir or _REPORTS_DIR

    json_path = os.path.join(directory, "history.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)

    csv_path = os.path.join(directory, "history.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Convert numeric fields
                for k in (
                    "observations",
                    "alerts_ok",
                    "alerts_warning",
                    "alerts_critical",
                    "feature_drift_count",
                    "score_drift_count",
                    "null_spike_count",
                ):
                    v = row.get(k, "0")
                    # CSV values are always strings; int("") and int("nan")
                    # both raise ValueError — catch and default to 0.
                    try:
                        row[k] = int(float(v)) if v not in (None, "", "nan", "NaN") else 0
                    except (ValueError, TypeError):
                        row[k] = 0
                rows.append(row)
            return rows

    return []


def _severity_color(severity):
    """Return CSS color for severity level."""
    s = str(severity).lower()
    if s == "critical":
        return "#dc3545"
    elif s == "warning":
        return "#ffc107"
    return "#28a745"


def _severity_badge(severity):
    """Return HTML badge for severity."""
    color = _severity_color(severity)
    text_color = "#fff" if severity.lower() != "warning" else "#000"
    return (
        f'<span style="background:{color};color:{text_color};'
        f'padding:2px 8px;border-radius:4px;font-size:12px;">'
        f"{severity.upper()}</span>"
    )


def generate_html(reports_dir=None):
    """
    Generate a self-contained HTML drift report string.

    Args:
        reports_dir: directory containing history.json or history.csv

    Returns:
        str: complete HTML document, or minimal empty-state HTML if no data
    """
    rows = _load_history(reports_dir)

    if not rows:
        return _empty_report_html()

    # Find worst batch
    severity_rank = {"critical": 3, "warning": 2, "ok": 1, "": 0}
    worst_idx = max(range(len(rows)), key=lambda i: severity_rank.get(rows[i].get("highest_severity", "").lower(), 0))

    # Prepare chart data
    labels = [r.get("batch_id", r.get("timestamp", f"batch-{i}"))[:20] for i, r in enumerate(rows)]
    warnings = [int(r.get("alerts_warning", 0)) for r in rows]
    criticals = [int(r.get("alerts_critical", 0)) for r in rows]
    feat_drift = [int(r.get("feature_drift_count", 0)) for r in rows]
    score_drift = [int(r.get("score_drift_count", 0)) for r in rows]
    null_spikes = [int(r.get("null_spike_count", 0)) for r in rows]
    observations = [int(r.get("observations", 0)) for r in rows]

    # Build table rows
    table_rows = ""
    for i, r in enumerate(rows):
        sev = r.get("highest_severity", "ok")
        highlight = ' style="background:#fff3cd;"' if i == worst_idx else ""
        # All string fields from history files are external data — escape to
        # prevent XSS if a batch_id / summary contains HTML/JS.
        esc = _html.escape
        table_rows += f"""<tr{highlight}>
            <td>{esc(str(r.get('timestamp', '')))[:19]}</td>
            <td>{esc(str(r.get('batch_id', '')))}</td>
            <td>{esc(str(r.get('model_version', '')))}</td>
            <td>{esc(str(r.get('schema_version', '')))}</td>
            <td>{r.get('observations', 0)}</td>
            <td>{r.get('alerts_ok', 0)}</td>
            <td>{r.get('alerts_warning', 0)}</td>
            <td>{r.get('alerts_critical', 0)}</td>
            <td>{r.get('feature_drift_count', 0)}</td>
            <td>{r.get('score_drift_count', 0)}</td>
            <td>{r.get('null_spike_count', 0)}</td>
            <td>{_severity_badge(esc(sev))}</td>
            <td style="font-size:11px;">{esc(str(r.get('summary', '')))}</td>
        </tr>\n"""

    # Summary stats
    total_batches = len(rows)
    total_obs = sum(observations)
    total_warn = sum(warnings)
    total_crit = sum(criticals)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drift Report — reef_imagery_pipeline</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js" integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ" crossorigin="anonymous"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 20px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #1a5276; }}
  .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; padding: 16px 24px; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .stat-card .value {{ font-size: 28px; font-weight: bold; }}
  .stat-card .label {{ font-size: 12px; color: #666; margin-top: 4px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .chart-box {{ background: #fff; padding: 16px; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px;
           overflow: hidden; font-size: 13px; }}
  th {{ background: #1a5276; color: #fff; padding: 10px 8px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f1f3f5; }}
  .generated {{ font-size: 11px; color: #999; margin-top: 20px; }}
  @media (max-width: 900px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Drift Report — reef_imagery_pipeline</h1>

<div class="stats">
  <div class="stat-card"><div class="value">{total_batches}</div><div class="label">Batches</div></div>
  <div class="stat-card"><div class="value">{total_obs}</div><div class="label">Observations</div></div>
  <div class="stat-card"><div class="value" style="color:#ffc107">{total_warn}</div><div class="label">Warnings</div></div>
  <div class="stat-card"><div class="value" style="color:#dc3545">{total_crit}</div><div class="label">Critical</div></div>
</div>

<div class="charts">
  <div class="chart-box"><canvas id="alertChart"></canvas></div>
  <div class="chart-box"><canvas id="driftChart"></canvas></div>
</div>

<h2>Batch History</h2>
<div style="overflow-x:auto;">
<table>
<thead><tr>
  <th>Timestamp</th><th>Batch ID</th><th>Model</th><th>Schema</th>
  <th>Obs</th><th>OK</th><th>Warn</th><th>Crit</th>
  <th>Feat Drift</th><th>Score Drift</th><th>Nulls</th>
  <th>Severity</th><th>Summary</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>

<p class="generated">Generated: {datetime.now(UTC).isoformat()[:19]}Z</p>

<script>
const labels = {_js_json(labels)};

new Chart(document.getElementById('alertChart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'Warnings', data: {_js_json(warnings)}, backgroundColor: '#ffc107' }},
      {{ label: 'Critical', data: {_js_json(criticals)}, backgroundColor: '#dc3545' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Alerts per Batch' }} }},
    scales: {{ x: {{ display: labels.length <= 30 }}, y: {{ beginAtZero: true }} }}
  }}
}});

new Chart(document.getElementById('driftChart'), {{
  type: 'line',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'Feature Drift', data: {_js_json(feat_drift)}, borderColor: '#6f42c1', fill: false, tension: 0.3 }},
      {{ label: 'Score Drift', data: {_js_json(score_drift)}, borderColor: '#fd7e14', fill: false, tension: 0.3 }},
      {{ label: 'Null Spikes', data: {_js_json(null_spikes)}, borderColor: '#20c997', fill: false, tension: 0.3 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Drift Trends' }} }},
    scales: {{ y: {{ beginAtZero: true }} }}
  }}
}});
</script>
</body>
</html>"""


def _empty_report_html():
    """Minimal HTML for empty state."""
    return """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Drift Report</title>
<style>body{font-family:sans-serif;margin:40px;color:#666;}</style>
</head><body>
<h1>Drift Report — reef_imagery_pipeline</h1>
<p>No drift history data available. Run batches with drift export enabled.</p>
</body></html>"""


def export_html(output_path=None, reports_dir=None):
    """
    Generate and write the HTML drift report to disk.

    Args:
        output_path: file path for HTML output (default: drift_reports/report.html)
        reports_dir: source directory for history artifacts

    Returns:
        str: path to written HTML file, or None on failure
    """
    directory = reports_dir or _REPORTS_DIR
    path = output_path or os.path.join(directory, "report.html")

    try:
        html = generate_html(reports_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(html)
        log.info(f"Drift HTML report exported: {path}")
        return path
    except Exception as e:
        log.error(f"Failed to generate drift HTML report: {e}")
        return None
