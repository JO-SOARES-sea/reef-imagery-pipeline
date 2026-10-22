"""
tests/test_sentinel_ingestion_features.py — Testes unitários para Ingestão e Features Morfo-Espectrais.
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.ingestion.sentinel_reader import (
    SentinelScene,
    read_sentinel_bands,
    calculate_ndwi,
    build_glint_mask,
    build_clean_water_mask,
    apply_water_mask,
    save_geotiff,
)
from src.analysis.features import (
    compute_psdb,
    compute_slope,
    compute_bpi,
    compute_vrm,
    compute_curv,
    compute_focal_std,
    extract_morpho_spectral_features,
)


@pytest.fixture
def synthetic_scene(tmp_path: Path):
    """Cria uma cena sintética de 4 bandas com terra ao norte e mar com recife ao sul."""
    width, height = 64, 64
    transform = Affine(10.0, 0.0, 560000.0, 0.0, -10.0, 4104000.0)
    crs = CRS.from_epsg(32629)

    y, x = np.mgrid[0:height, 0:width]
    is_land = y < (height // 2)

    b02 = np.where(is_land, 0.04, 0.18).astype(np.float32)
    b03 = np.where(is_land, 0.07, 0.15).astype(np.float32)
    b04 = np.where(is_land, 0.09, 0.03).astype(np.float32)
    b08 = np.where(is_land, 0.35, 0.01).astype(np.float32)

    # Anomalia de recife submerso
    reef_spot = (~is_land) & ((x - 32) ** 2 + (y - 48) ** 2 < 6**2)
    b03[reef_spot] += 0.04
    b02[reef_spot] += 0.02

    # Espuma de rebentação costeira
    foam_spot = (~is_land) & (y == (height // 2))
    b08[foam_spot] = 0.15

    stack = np.stack([b02, b03, b04, b08], axis=0)
    tif_path = tmp_path / "synthetic_s2.tif"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(stack)

    return tif_path


def test_read_sentinel_bands_and_ndwi(synthetic_scene: Path):
    scene = read_sentinel_bands(synthetic_scene, normalize_reflectance=False)
    assert scene.shape == (64, 64)
    assert scene.nir is not None

    ndwi, water_mask = calculate_ndwi(scene.green, scene.nir, threshold=0.0)
    assert ndwi.shape == (64, 64)
    # Metade sul deve ser maioritariamente classificada como água
    assert np.all(~water_mask[:32, :])
    assert np.any(water_mask[32:, :])


def test_glint_and_clean_water_mask(synthetic_scene: Path):
    scene = read_sentinel_bands(synthetic_scene, normalize_reflectance=False)
    glint_mask = build_glint_mask(scene.green, scene.nir, nir_thresh=0.08)
    assert glint_mask.shape == (64, 64)

    clean_mask = build_clean_water_mask(scene, ndwi_threshold=0.0, apply_glint_filter=True)
    assert clean_mask.shape == (64, 64)
    # A linha de espuma simulada deve ser filtrada
    assert not clean_mask[32, 0]


def test_morpho_spectral_features(synthetic_scene: Path):
    scene = read_sentinel_bands(synthetic_scene, normalize_reflectance=False)
    clean_mask = build_clean_water_mask(scene, ndwi_threshold=0.0, apply_glint_filter=False)

    features = extract_morpho_spectral_features(
        scene.blue, scene.green, scene.red, scene.nir, water_mask=clean_mask
    )

    expected_keys = [
        "B02", "B03", "B04", "B08", "psdb", "psdb_slope", "psdb_bpi100",
        "psdb_bpi200", "psdb_vrm50", "psdb_curv100", "B02_std30", "B03_std30",
        "z_b02_b03", "z_b03_b04"
    ]
    for key in expected_keys:
        assert key in features, f"Feature obrigatória {key} ausente"
        assert features[key].shape == (64, 64)


def test_save_geotiff(tmp_path: Path):
    data = np.ones((32, 32), dtype=np.float32)
    profile = {
        "crs": CRS.from_epsg(32629),
        "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0),
    }
    out_file = tmp_path / "test_out.tif"
    save_geotiff(out_file, data, profile)
    assert out_file.exists()
    with rasterio.open(out_file) as src:
        assert src.shape == (32, 32)
        assert src.count == 1
