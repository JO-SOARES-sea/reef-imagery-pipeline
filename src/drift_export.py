"""
drift_export.py — Serializes drift summary to JSON for dashboard/ops consumption.
==================================================================================

Provides:
- export_payload(): builds a stable JSON-serializable dict
- export_to_file(): writes payload to a JSON file on disk
- export_to_webhook(): POSTs payload to a URL (non-blocking, failure-safe)

All exports are non-blocking and failure-safe — they never break inference or batch.
"""

import json
import logging
import os
from datetime import UTC, datetime

log = logging.getLogger(__name__)

# Default export directory
_EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "drift_reports")

# Pipeline identity
_PIPELINE_NAME = "reef_imagery_pipeline"


def export_payload(batch_id=None, model_version=None, schema_version=None):
    """
    Build a dashboard-friendly JSON payload from the current drift summary.

    Args:
        batch_id: optional batch identifier string
        model_version: optional model version (auto-read from metadata if None)
        schema_version: optional schema version (auto-read from metadata if None)

    Returns:
        dict: stable JSON-serializable payload
    """
    from src.drift_monitor import CRITICAL, OK, WARNING, summary

    s = summary()

    # Auto-read model metadata if versions not provided
    if model_version is None or schema_version is None:
        model_version, schema_version = _read_model_versions(model_version, schema_version)

    # Build human-readable summary line
    crit = s["counts"].get(CRITICAL, 0)
    warn = s["counts"].get(WARNING, 0)
    parts = []
    if crit > 0:
        parts.append(f"{crit} critical drift event{'s' if crit != 1 else ''}")
    if warn > 0:
        parts.append(f"{warn} warning{'s' if warn != 1 else ''}")
    if not parts:
        parts.append("no drift detected")
    summary_text = ", ".join(parts) + " in batch"

    # Build worst alert detail
    worst_detail = None
    if s["worst_alert"]:
        worst_detail = {
            "feature": s["worst_alert"][0],
            "level": s["worst_alert"][1],
            "reason": s["worst_alert"][2],
        }

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "pipeline": _PIPELINE_NAME,
        "model_version": model_version or "unknown",
        "schema_version": schema_version or "unknown",
        "batch_id": batch_id or _generate_batch_id(),
        "observations": s["total_observations"],
        "alerts": {
            "ok": s["counts"].get(OK, 0),
            "warning": s["counts"].get(WARNING, 0),
            "critical": s["counts"].get(CRITICAL, 0),
        },
        "feature_drift_count": s["feature_drift_count"],
        "score_drift_count": s["score_drift_count"],
        "null_spike_count": s["null_spike_count"],
        "highest_severity": s["worst_level"].lower(),
        "worst_alert": worst_detail,
        "summary": summary_text,
    }


def export_to_file(batch_id=None, model_version=None, schema_version=None, output_dir=None):
    """
    Write drift payload as JSON to disk. Non-blocking on failure.

    Args:
        batch_id: optional batch identifier
        model_version: optional model version
        schema_version: optional schema version
        output_dir: directory to write to (default: drift_reports/)

    Returns:
        str: path to written file, or None on failure
    """
    try:
        payload = export_payload(batch_id, model_version, schema_version)
        out_dir = output_dir or _EXPORT_DIR
        os.makedirs(out_dir, exist_ok=True)

        filename = f"drift_{payload['batch_id']}.json"
        filepath = os.path.join(out_dir, filename)

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        log.info(f"Drift report exported: {filepath}")
        return filepath
    except Exception as e:
        log.error(f"Drift export to file failed (non-blocking): {e}")
        return None


def export_to_webhook(url, batch_id=None, model_version=None, schema_version=None, timeout=5):
    """
    POST drift payload to a webhook URL. Non-blocking on failure.

    Args:
        url: webhook endpoint URL
        batch_id: optional batch identifier
        model_version: optional model version
        schema_version: optional schema version
        timeout: request timeout in seconds

    Returns:
        bool: True if successful, False on failure
    """
    try:
        import urllib.request

        payload = export_payload(batch_id, model_version, schema_version)
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status < 300:
                log.info(f"Drift report posted to webhook ({resp.status})")
                return True
            else:
                log.warning(f"Webhook returned status {resp.status}")
                return False
    except Exception as e:
        log.error(f"Drift export to webhook failed (non-blocking): {e}")
        return False


def _read_model_versions(model_version=None, schema_version=None):
    """Read model/schema versions from metadata file."""
    try:
        meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "feature_ranker_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            return (
                model_version or meta.get("model_version", "unknown"),
                schema_version or meta.get("schema_version", "unknown"),
            )
    except Exception:
        pass
    return model_version or "unknown", schema_version or "unknown"


def _generate_batch_id():
    """Generate a default batch ID from timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")
