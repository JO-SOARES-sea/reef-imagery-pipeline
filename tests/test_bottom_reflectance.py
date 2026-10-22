"""Physics tests for the water-column reflectance inversion formula.

Imports the production function directly so any change to the formula
in reef_ml_predictor_acolite.py is caught immediately.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reef_ml_predictor_acolite import invert_water_column


def test_bottom_reflectance_greater_than_surface():
    """Water column attenuates upwelling light, so bottom R must exceed surface R."""
    rrs = np.array([0.05, 0.10, 0.02], dtype=np.float32)
    kd = 0.045  # typical Algarve oligotrophic
    bottom = invert_water_column(rrs, kd, depth_m=10.0)
    assert np.all(bottom > rrs), "bottom reflectance must be > surface after water-column correction"


def test_deeper_water_amplifies_more():
    """Greater depth → stronger water-column correction → higher bottom R estimate."""
    rrs = np.array([0.05], dtype=np.float32)
    kd = 0.045
    shallow = invert_water_column(rrs, kd, depth_m=5.0)
    deep = invert_water_column(rrs, kd, depth_m=20.0)
    assert deep > shallow, "deeper correction should amplify more"


def test_clearer_water_smaller_correction():
    """Lower Kd → less attenuation → smaller correction factor."""
    rrs = np.array([0.05], dtype=np.float32)
    depth = 10.0
    turbid = invert_water_column(rrs, kd=0.30, depth_m=depth)
    clear = invert_water_column(rrs, kd=0.03, depth_m=depth)
    assert clear < turbid, "turbid water should produce larger bottom R correction"


def test_zero_depth_returns_surface():
    """At depth=0 the correction factor is exp(0)=1 → bottom R equals surface R."""
    rrs = np.array([0.10, 0.20], dtype=np.float32)
    result = invert_water_column(rrs, kd=0.045, depth_m=0.0)
    np.testing.assert_allclose(result, rrs, rtol=1e-5)


def test_negatives_clipped():
    """Negative surface values (noisy pixels) must be clipped to 0 in output."""
    rrs = np.array([-0.01, 0.0], dtype=np.float32)
    result = invert_water_column(rrs, kd=0.045, depth_m=10.0)
    assert np.all(result >= 0.0)
