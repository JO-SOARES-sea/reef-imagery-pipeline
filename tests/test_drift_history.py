"""Tests for src/drift_history.py — historical drift report aggregation."""

import csv
import json
import os

from src.drift_history import (
    HISTORY_FIELDS,
    _flatten_report,
    aggregate,
    export_history_csv,
    export_history_json,
    load_reports,
)


def _make_report(batch_id="batch-001", obs=10, ok=8, warn=1, crit=1,
                 feat_drift=1, score_drift=2, nulls=0,
                 severity="critical", worst_feat="score_value",
                 timestamp="2026-05-25T01:00:00+00:00"):
    """Helper to create a sample drift report dict."""
    return {
        "timestamp": timestamp,
        "pipeline": "reef_imagery_pipeline",
        "model_version": "1.2",
        "schema_version": "2.0",
        "batch_id": batch_id,
        "observations": obs,
        "alerts": {"ok": ok, "warning": warn, "critical": crit},
        "feature_drift_count": feat_drift,
        "score_drift_count": score_drift,
        "null_spike_count": nulls,
        "highest_severity": severity,
        "worst_alert": {
            "feature": worst_feat,
            "level": "CRITICAL",
            "reason": "z=6.0 (score=1.5000)",
        },
        "summary": f"{crit} critical drift event, {warn} warning in batch",
    }


def _write_reports(tmpdir, reports):
    """Write report dicts as JSON files in tmpdir."""
    for i, report in enumerate(reports):
        bid = report.get("batch_id", f"batch-{i:03d}")
        path = os.path.join(tmpdir, f"drift_{bid}.json")
        with open(path, "w") as f:
            json.dump(report, f)


class TestLoadReports:
    def test_empty_directory(self, tmp_path):
        """Returns empty list for directory with no drift files."""
        result = load_reports(str(tmp_path))
        assert result == []

    def test_nonexistent_directory(self, tmp_path):
        """Returns empty list for non-existent directory."""
        result = load_reports(str(tmp_path / "nonexistent"))
        assert result == []

    def test_loads_valid_reports(self, tmp_path):
        """Loads and parses valid JSON reports."""
        reports = [
            _make_report("batch-001", timestamp="2026-05-25T01:00:00+00:00"),
            _make_report("batch-002", timestamp="2026-05-25T02:00:00+00:00"),
        ]
        _write_reports(str(tmp_path), reports)

        result = load_reports(str(tmp_path))
        assert len(result) == 2
        assert result[0]["batch_id"] == "batch-001"
        assert result[1]["batch_id"] == "batch-002"

    def test_sorted_by_timestamp(self, tmp_path):
        """Reports are returned sorted by timestamp."""
        reports = [
            _make_report("batch-late", timestamp="2026-05-25T03:00:00+00:00"),
            _make_report("batch-early", timestamp="2026-05-25T01:00:00+00:00"),
        ]
        _write_reports(str(tmp_path), reports)

        result = load_reports(str(tmp_path))
        assert result[0]["batch_id"] == "batch-early"
        assert result[1]["batch_id"] == "batch-late"

    def test_skips_malformed_json(self, tmp_path):
        """Malformed JSON files are skipped without crashing."""
        # Write one valid report
        _write_reports(str(tmp_path), [_make_report("batch-good")])
        # Write one malformed file
        bad_path = os.path.join(str(tmp_path), "drift_batch-bad.json")
        with open(bad_path, "w") as f:
            f.write("{invalid json content")

        result = load_reports(str(tmp_path))
        assert len(result) == 1
        assert result[0]["batch_id"] == "batch-good"

    def test_skips_non_dict_json(self, tmp_path):
        """JSON files that aren't dicts are skipped."""
        _write_reports(str(tmp_path), [_make_report("batch-good")])
        bad_path = os.path.join(str(tmp_path), "drift_batch-array.json")
        with open(bad_path, "w") as f:
            json.dump([1, 2, 3], f)

        result = load_reports(str(tmp_path))
        assert len(result) == 1


class TestFlattenReport:
    def test_all_fields_present(self):
        """Flattened report contains all HISTORY_FIELDS."""
        report = _make_report()
        row = _flatten_report(report)
        for field in HISTORY_FIELDS:
            assert field in row, f"Missing field: {field}"

    def test_handles_missing_fields(self):
        """Missing fields default to empty/zero."""
        row = _flatten_report({})
        assert row["timestamp"] == ""
        assert row["observations"] == 0
        assert row["alerts_ok"] == 0
        assert row["worst_alert_feature"] == ""

    def test_handles_null_worst_alert(self):
        """worst_alert=None produces empty strings."""
        report = _make_report()
        report["worst_alert"] = None
        row = _flatten_report(report)
        assert row["worst_alert_feature"] == ""
        assert row["worst_alert_level"] == ""


class TestAggregate:
    def test_aggregate_returns_flat_rows(self, tmp_path):
        """aggregate() returns list of flat dicts."""
        _write_reports(str(tmp_path), [_make_report("b1"), _make_report("b2")])
        rows = aggregate(str(tmp_path))
        assert len(rows) == 2
        assert rows[0]["batch_id"] == "b1"
        assert "alerts_ok" in rows[0]

    def test_aggregate_empty_dir(self, tmp_path):
        """Returns empty list for empty directory."""
        rows = aggregate(str(tmp_path))
        assert rows == []


class TestExportCSV:
    def test_creates_csv(self, tmp_path):
        """export_history_csv writes a valid CSV file."""
        _write_reports(str(tmp_path), [
            _make_report("b1", timestamp="2026-05-25T01:00:00+00:00"),
            _make_report("b2", timestamp="2026-05-25T02:00:00+00:00"),
        ])
        out = os.path.join(str(tmp_path), "history.csv")
        result = export_history_csv(output_path=out, reports_dir=str(tmp_path))

        assert result == out
        assert os.path.exists(out)

        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["batch_id"] == "b1"
        assert set(reader.fieldnames) == set(HISTORY_FIELDS)

    def test_returns_none_if_empty(self, tmp_path):
        """Returns None if no reports to aggregate."""
        result = export_history_csv(reports_dir=str(tmp_path))
        assert result is None


class TestExportJSON:
    def test_creates_json(self, tmp_path):
        """export_history_json writes a valid JSON array."""
        _write_reports(str(tmp_path), [
            _make_report("b1", timestamp="2026-05-25T01:00:00+00:00"),
        ])
        out = os.path.join(str(tmp_path), "history.json")
        result = export_history_json(output_path=out, reports_dir=str(tmp_path))

        assert result == out
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["batch_id"] == "b1"

    def test_returns_none_if_empty(self, tmp_path):
        """Returns None if no reports to aggregate."""
        result = export_history_json(reports_dir=str(tmp_path))
        assert result is None
