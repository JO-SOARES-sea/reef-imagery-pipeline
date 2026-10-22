"""Tests for src/drift_report_html.py — HTML drift report generation."""

import json
import os

from src.drift_report_html import _load_history, export_html, generate_html


def _make_history(tmp_path, batches):
    """Write a history.json file with given batch rows."""
    path = os.path.join(str(tmp_path), "history.json")
    with open(path, "w") as f:
        json.dump(batches, f)
    return str(tmp_path)


def _sample_batch(batch_id="b1", severity="ok", warn=0, crit=0):
    return {
        "timestamp": "2026-05-25T01:00:00+00:00",
        "pipeline": "reef_imagery_pipeline",
        "model_version": "1.2",
        "schema_version": "2.0",
        "batch_id": batch_id,
        "observations": 10,
        "alerts_ok": 10 - warn - crit,
        "alerts_warning": warn,
        "alerts_critical": crit,
        "feature_drift_count": 1 if warn or crit else 0,
        "score_drift_count": 1 if crit else 0,
        "null_spike_count": 0,
        "highest_severity": severity,
        "worst_alert_feature": "kd_b02" if severity != "ok" else "",
        "worst_alert_level": severity.upper() if severity != "ok" else "",
        "worst_alert_reason": "z=3.0" if severity != "ok" else "",
        "summary": f"{crit} critical, {warn} warnings in batch",
    }


class TestGenerateHTML:
    def test_empty_state(self, tmp_path):
        """Empty directory produces valid empty-state HTML."""
        html = generate_html(str(tmp_path))
        assert "<!DOCTYPE html>" in html
        assert "No drift history data available" in html

    def test_generates_valid_html(self, tmp_path):
        """Produces complete HTML with data."""
        reports_dir = _make_history(tmp_path, [
            _sample_batch("b1", "ok"),
            _sample_batch("b2", "warning", warn=2),
            _sample_batch("b3", "critical", warn=1, crit=1),
        ])
        html = generate_html(reports_dir)

        assert "<!DOCTYPE html>" in html
        assert "reef_imagery_pipeline" in html
        assert "chart.js" in html.lower() or "Chart" in html

    def test_contains_batch_ids(self, tmp_path):
        """Batch IDs appear in the HTML table."""
        reports_dir = _make_history(tmp_path, [
            _sample_batch("my-batch-001"),
            _sample_batch("my-batch-002"),
        ])
        html = generate_html(reports_dir)

        assert "my-batch-001" in html
        assert "my-batch-002" in html

    def test_severity_badges(self, tmp_path):
        """Severity badges with colors appear in HTML."""
        reports_dir = _make_history(tmp_path, [
            _sample_batch("b1", "critical", crit=1),
        ])
        html = generate_html(reports_dir)

        assert "#dc3545" in html  # Critical red
        assert "CRITICAL" in html

    def test_model_version_visible(self, tmp_path):
        """Model and schema versions appear in table."""
        reports_dir = _make_history(tmp_path, [_sample_batch("b1")])
        html = generate_html(reports_dir)

        assert "1.2" in html
        assert "2.0" in html

    def test_chart_data_embedded(self, tmp_path):
        """Chart.js data arrays are embedded in HTML."""
        reports_dir = _make_history(tmp_path, [
            _sample_batch("b1", "warning", warn=3),
        ])
        html = generate_html(reports_dir)

        assert "alertChart" in html
        assert "driftChart" in html

    def test_worst_batch_highlighted(self, tmp_path):
        """Worst batch row has highlight styling."""
        reports_dir = _make_history(tmp_path, [
            _sample_batch("b1", "ok"),
            _sample_batch("b2", "critical", crit=2),
            _sample_batch("b3", "ok"),
        ])
        html = generate_html(reports_dir)

        # The highlight background should appear
        assert "#fff3cd" in html


class TestExportHTML:
    def test_writes_file(self, tmp_path):
        """export_html writes file to disk."""
        reports_dir = _make_history(tmp_path, [_sample_batch("b1")])
        out = os.path.join(str(tmp_path), "report.html")

        result = export_html(output_path=out, reports_dir=reports_dir)

        assert result == out
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content

    def test_returns_none_on_failure(self):
        """Returns None if write fails."""
        result = export_html(output_path="/nonexistent/x/y/z/report.html",
                            reports_dir="/nonexistent")
        assert result is None


class TestLoadHistory:
    def test_prefers_json(self, tmp_path):
        """Loads from history.json when present."""
        reports_dir = _make_history(tmp_path, [_sample_batch("from-json")])
        rows = _load_history(str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["batch_id"] == "from-json"

    def test_empty_dir_returns_empty(self, tmp_path):
        """Returns empty list for directory with no history file."""
        rows = _load_history(str(tmp_path))
        assert rows == []
