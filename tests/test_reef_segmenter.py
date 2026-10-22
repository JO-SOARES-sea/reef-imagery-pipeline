"""Tests for src/reef_segmenter.py — U-Net reef-mask inference."""
import numpy as np
import pytest

from src.reef_segmenter import _DEFAULT_WEIGHTS, HAS_TORCH, normalize_depth

# ── normalize_depth: pure, no torch needed ────────────────────────────────────

class TestNormalizeDepth:
    def test_sea_level_maps_to_one(self):
        assert normalize_depth(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_optical_limit_maps_to_zero(self):
        # -40 m (default -SDB_OPTICAL_LIMIT_M) is the deep end → 0.0
        assert normalize_depth(np.array([-40.0]))[0] == pytest.approx(0.0)

    def test_midpoint(self):
        assert normalize_depth(np.array([-20.0]))[0] == pytest.approx(0.5)

    def test_nan_becomes_zero_depth_then_one(self):
        # NaN → 0 m depth → normalized 1.0
        assert normalize_depth(np.array([np.nan]))[0] == pytest.approx(1.0)

    def test_below_limit_clipped(self):
        assert normalize_depth(np.array([-100.0]))[0] == pytest.approx(0.0)

    def test_matches_dataset_normalization(self):
        """Inference and training normalization must be identical."""
        ds_mod = pytest.importorskip("src.ml_unet_dataset")
        ds = ds_mod.ReefPatchDataset.__new__(ds_mod.ReefPatchDataset)
        ds.min_depth_m = -40.0
        ds.max_depth_m = 0.0
        arr = np.array([0.0, -10.0, -20.0, -40.0, -60.0], dtype=np.float32)
        np.testing.assert_allclose(normalize_depth(arr), ds._normalize_depth(arr))


# ── model load + inference: needs torch + the trained weights ─────────────────

_NEEDS_MODEL = pytest.mark.skipif(
    not HAS_TORCH or not _DEFAULT_WEIGHTS.exists(),
    reason="torch and models/unet_reef_best.pth required",
)


@_NEEDS_MODEL
class TestSegmentReef:
    def test_array_inference_shapes_and_ranges(self):
        from src.reef_segmenter import segment_reef_array
        depth = np.random.uniform(-30.0, 0.0, size=(128, 128)).astype(np.float32)
        out = segment_reef_array(depth)
        assert out["prob"].shape == (128, 128)
        assert out["mask"].shape == (128, 128)
        assert out["prob"].min() >= 0.0 and out["prob"].max() <= 1.0
        assert set(np.unique(out["mask"])).issubset({0, 1})
        assert 0.0 <= out["reef_fraction"] <= 1.0

    def test_non_square_raster_tiles_and_crops_back(self):
        from src.reef_segmenter import segment_reef_array
        depth = np.full((300, 200), -15.0, dtype=np.float32)
        out = segment_reef_array(depth)
        assert out["prob"].shape == (300, 200)  # padded for tiling, cropped back

    def test_nan_depth_handled(self):
        from src.reef_segmenter import segment_reef_array
        depth = np.full((64, 64), np.nan, dtype=np.float32)
        out = segment_reef_array(depth)  # must not raise
        assert out["prob"].shape == (64, 64)

    def test_tif_roundtrip_writes_georeferenced_mask(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds

        from src.reef_segmenter import segment_reef_tif

        src = tmp_path / "bathy_dgt_lidar_20240905.tif"
        depth = np.random.uniform(-25.0, 0.0, size=(128, 128)).astype(np.float32)
        transform = from_bounds(-8.23, 37.07, -8.22, 37.08, 128, 128)
        with rasterio.open(
            str(src), "w", driver="GTiff", height=128, width=128, count=1,
            dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(depth, 1)

        out = segment_reef_tif(str(src))
        # date suffix preserved, bathy prefix stripped
        assert out["out_tif"].endswith("reef_mask_20240905.tif")

        with rasterio.open(out["out_tif"]) as r:
            assert r.count == 1
            assert r.crs.to_string() == "EPSG:4326"
            assert r.width == 128 and r.height == 128
            assert set(np.unique(r.read(1))).issubset({0, 1})


@pytest.mark.skipif(HAS_TORCH, reason="exercises the torch-less degradation path")
def test_segment_raises_clearly_without_torch():
    from src.reef_segmenter import segment_reef_array
    with pytest.raises(RuntimeError, match="PyTorch not installed"):
        segment_reef_array(np.zeros((64, 64), dtype=np.float32))
