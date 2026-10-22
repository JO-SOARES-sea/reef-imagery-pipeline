"""Tests for src/drift_export.py — drift summary JSON export."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.drift_export import export_payload, export_to_file, export_to_webhook
from src.drift_monitor import observe, reset


@pytest.fixture(autouse=True)
def reset_state():
    """Reset drift state between tests."""
    reset()


class TestPayload:
    def test_payload_structure(self):
        """Payload contains all required dashboard fields."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        payload = export_payload(batch_id="test-batch-001")

        required_fields = [
            "timestamp", "pipeline", "model_version", "schema_version",
            "batch_id", "observations", "alerts", "feature_drift_count",
            "score_drift_count", "null_spike_count", "highest_severity",
            "worst_alert", "summary",
        ]
        for field in required_fields:
            assert field in payload, f"Missing field: {field}"

    def test_payload_alert_counts(self):
        """Alert counts match observations."""
        normal = {'benthic_contrast': 0.2, 'snr': 120,
                  'fft_clean': 10000, 'edge_entropy': 6.5,
                  'dyn_range': 0.008, 'signal': 0.125}
        for _ in range(3):
            observe(normal, 0.60)
        for _ in range(2):
            observe(normal, 0.22)  # WARNING (z=2.5)

        payload = export_payload()

        assert payload["observations"] == 5
        assert payload["alerts"]["ok"] == 3
        assert payload["alerts"]["warning"] == 2
        assert payload["alerts"]["critical"] == 0

    def test_payload_highest_severity(self):
        """highest_severity reflects worst event."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 1.5)  # CRITICAL

        payload = export_payload()
        assert payload["highest_severity"] == "critical"

    def test_payload_json_serializable(self):
        """Payload can be serialized to JSON without errors."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        payload = export_payload(batch_id="ser-test")
        json_str = json.dumps(payload)
        parsed = json.loads(json_str)
        assert parsed["batch_id"] == "ser-test"

    def test_payload_model_version_from_metadata(self):
        """Model version is read from metadata file."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        payload = export_payload()
        assert payload["model_version"] == "2.0"
        assert payload["schema_version"] == "3.0"

    def test_payload_summary_text(self):
        """Summary text is human-readable."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        payload = export_payload()
        assert "in batch" in payload["summary"]

    def test_payload_batch_id_override(self):
        """Custom batch_id is used when provided."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        payload = export_payload(batch_id="my-custom-batch")
        assert payload["batch_id"] == "my-custom-batch"


class TestFileExport:
    def test_export_creates_file(self):
        """export_to_file writes a valid JSON file."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_to_file(batch_id="file-test", output_dir=tmpdir)

            assert path is not None
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)
            assert data["batch_id"] == "file-test"
            assert data["observations"] == 1

    def test_export_failure_non_blocking(self):
        """Export failure returns None without raising."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        # Write to an invalid path
        result = export_to_file(batch_id="fail-test", output_dir="/nonexistent/path/x/y/z")
        assert result is None  # Non-blocking, no exception


class TestWebhookExport:
    def test_webhook_success(self):
        """Successful webhook POST returns True."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_resp):
            result = export_to_webhook("http://localhost:9999/hook", batch_id="wh-test")
            assert result is True

    def test_webhook_failure_non_blocking(self):
        """Webhook failure returns False without raising."""
        observe({'benthic_contrast': 0.2, 'snr': 120,
                 'fft_clean': 10000, 'edge_entropy': 6.5,
                 'dyn_range': 0.008, 'signal': 0.125}, 0.60)

        with patch('urllib.request.urlopen', side_effect=Exception("Connection refused")):
            result = export_to_webhook("http://localhost:9999/hook", batch_id="wh-fail")
            assert result is False
