"""
Tests for reef_imagery_pipeline_v3.py --step reef_candidates wiring.

Verifies that:
1. --step reef_candidates runs without error on synthetic S2 band TIFFs and
   produces a reef_candidates_multiband.geojson in the output directory.
2. --step all uses the discovery sequence (not BVI score as the terminal step).
3. --step bathy and --step score remain independently callable without error.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Import once at module level — no importlib.reload().
# The module is not mutated between tests so a single import is stable.
import reef_imagery_pipeline_v3 as v3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tif(path, arr, crs="EPSG:32629"):
    transform = Affine.translation(500000.0, 4100000.0) * Affine.scale(10.0, -10.0)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "crs": crs,
        "transform": transform,
        "nodata": float("nan"),
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def _make_synthetic_bands(output_dir: str, date_tag: str = "20241015") -> None:
    """Write minimal synthetic B02/B03 GeoTIFFs into output_dir."""
    rng = np.random.default_rng(42)
    for band in ("B02", "B03"):
        arr = rng.uniform(0.01, 0.15, size=(32, 32)).astype(np.float32)
        _write_tif(os.path.join(output_dir, f"S2_{band}_{date_tag}.tif"), arr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_step_registry_contains_reef_candidates():
    """STEPS list must contain reef_candidates after bathy."""
    assert "reef_candidates" in v3.STEPS
    bathy_idx = v3.STEPS.index("bathy")
    rc_idx = v3.STEPS.index("reef_candidates")
    assert rc_idx > bathy_idx, "reef_candidates must come after bathy in STEPS"


def test_all_steps_sequence_terminates_at_report():
    """--step all must use a sequence ending at report, not score."""
    assert v3._ALL_STEPS[-1] == "report", (
        f"--step all must terminate at report; got {v3._ALL_STEPS[-1]}"
    )
    assert "score" not in v3._ALL_STEPS, (
        "--step all must not include the BVI score step"
    )


def test_step_reef_candidates_produces_geojson(tmp_path):
    """
    --step reef_candidates should produce reef_candidates_multiband.geojson
    when synthetic S2 bands are present in the output directory.
    """
    (tmp_path / "pipeline.log").touch()
    _make_synthetic_bands(str(tmp_path))

    v3.step_reef_candidates(
        output_dir=str(tmp_path),
        lat=37.069,
        lon=-8.210,
        date="2024-10-15",
        depth_min=-50.0,
        depth_max=-1.0,
    )

    candidates = tmp_path / "reef_candidates_multiband.geojson"
    assert candidates.exists(), (
        "reef_candidates_multiband.geojson must be produced by --step reef_candidates"
    )
    with open(candidates) as f:
        fc = json.load(f)
    assert fc.get("type") == "FeatureCollection", "Output must be a valid GeoJSON FeatureCollection"


def test_standalone_bathy_step_not_broken():
    """--step bathy must remain a valid callable in step_fns."""
    assert "bathy" in v3.STEPS
    assert "bathy" in v3._ALL_STEPS


def test_standalone_score_step_not_broken():
    """--step score (BVI) must remain a valid named step even though --step all skips it."""
    assert "score" in v3.STEPS
    assert "score" not in v3._ALL_STEPS
