import cv2
import numpy as np
import pytest

from src.enhancer import fetch_vsi_patch

pytestmark = pytest.mark.network

lat, lon = 37.05811, -8.20978


def analyze_spatial_cleanliness(date_str):
    try:
        b02_ref = fetch_vsi_patch(lat, lon, date_str, buffer_m=500.0)
    except Exception as e:
        return {"error": str(e)}
    valid = b02_ref[b02_ref > 0]
    if len(valid) == 0:
        return {"error": "Empty data"}
    macro = cv2.GaussianBlur(b02_ref, (9, 9), 0)
    micro = b02_ref - macro
    macro_var = float(np.var(macro))
    micro_var = float(np.var(micro))
    spatial_ratio = macro_var / (micro_var + 1e-12)
    sobelx = cv2.Sobel(macro, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(macro, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    if grad_mag.max() > grad_mag.min():
        grad_norm = ((grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min()) * 255).astype(np.uint8)
        hist = cv2.calcHist([grad_norm], [0], None, [256], [0, 256]).ravel()
        p = hist / np.sum(hist)
        p = p[p > 0]
        edge_entropy = float(-np.sum(p * np.log2(p)))
    else:
        edge_entropy = 0.0
    return {
        "macro_var": macro_var,
        "micro_var": micro_var,
        "spatial_ratio": spatial_ratio,
        "edge_entropy": edge_entropy,
    }


def test_spatial_cleanliness():
    dates = ["2025-09-25", "2025-09-02", "2023-09-01", "2024-09-30", "2022-09-26", "2019-09-07", "2020-09-06"]
    results = {}
    for d in dates:
        results[d] = analyze_spatial_cleanliness(d)
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    assert len(valid_results) > 0, "At least one date should return valid spatial cleanliness data"
    for res in valid_results.values():
        assert res["macro_var"] >= 0
        assert res["micro_var"] >= 0
        assert res["spatial_ratio"] >= 0
        assert res["edge_entropy"] >= 0
