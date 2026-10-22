"""Tests for src/sensor_config.py — multi-sensor registry."""
import pytest

from src.sensor_config import (
    SENSOR_REGISTRY,
    get_blue_band,
    get_reef_bands,
    get_sensor,
    list_sensors,
    map_band_name,
)

# ── Registry integrity ────────────────────────────────────────────────────────

class TestRegistry:
    def test_sentinel2_present(self):
        assert "sentinel-2" in SENSOR_REGISTRY

    def test_all_sensors_have_b02(self):
        for name, cfg in SENSOR_REGISTRY.items():
            assert "B02" in cfg.bands, f"{name} missing B02"

    def test_all_sensors_have_positive_dn_scale(self):
        for name, cfg in SENSOR_REGISTRY.items():
            assert cfg.dn_scale > 0, f"{name} dn_scale must be positive"

    def test_all_bands_have_valid_wavelength(self):
        for sensor_name, cfg in SENSOR_REGISTRY.items():
            for band_key, band in cfg.bands.items():
                assert 300 <= band.wavelength_nm <= 2400, (
                    f"{sensor_name}.{band_key} wavelength {band.wavelength_nm} out of range"
                )

    def test_all_bands_have_positive_resolution(self):
        for sensor_name, cfg in SENSOR_REGISTRY.items():
            for band_key, band in cfg.bands.items():
                assert band.resolution_m > 0


# ── get_sensor ────────────────────────────────────────────────────────────────

class TestGetSensor:
    def test_returns_sentinel2(self):
        cfg = get_sensor("sentinel-2")
        assert cfg.name == "sentinel-2"

    def test_case_insensitive(self):
        cfg = get_sensor("Sentinel-2")
        assert cfg.name == "sentinel-2"

    def test_raises_for_unknown_sensor(self):
        with pytest.raises(ValueError, match="Unknown sensor"):
            get_sensor("not-a-sensor")

    def test_error_lists_available_sensors(self):
        with pytest.raises(ValueError) as exc_info:
            get_sensor("bogus")
        assert "sentinel-2" in str(exc_info.value)


# ── list_sensors ──────────────────────────────────────────────────────────────

class TestListSensors:
    def test_returns_list(self):
        result = list_sensors()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_sentinel2_in_list(self):
        assert "sentinel-2" in list_sensors()


# ── get_reef_bands ────────────────────────────────────────────────────────────

class TestGetReefBands:
    def test_returns_dict(self):
        result = get_reef_bands("sentinel-2")
        assert isinstance(result, dict)

    def test_priority_filter_works(self):
        # max_priority=1 should return fewer bands than max_priority=3
        p1 = get_reef_bands("sentinel-2", max_priority=1)
        p3 = get_reef_bands("sentinel-2", max_priority=3)
        assert len(p1) <= len(p3)

    def test_b02_always_reef_critical(self):
        bands = get_reef_bands("sentinel-2", max_priority=1)
        assert "B02" in bands


# ── get_blue_band ─────────────────────────────────────────────────────────────

class TestGetBlueBand:
    def test_returns_b02_config(self):
        b = get_blue_band("sentinel-2")
        assert b.name is not None

    def test_blue_wavelength_approx_490nm(self):
        # S2 B02 is centred near 490 nm
        b = get_blue_band("sentinel-2")
        assert 440 <= b.wavelength_nm <= 530

    def test_raises_for_unknown_sensor(self):
        with pytest.raises(ValueError):
            get_blue_band("not-a-sensor")


# ── map_band_name ─────────────────────────────────────────────────────────────

class TestMapBandName:
    def test_b02_maps_to_itself_on_same_sensor(self):
        result = map_band_name("sentinel-2", "sentinel-2", "B02")
        assert result == "B02"

    def test_unknown_band_returns_none(self):
        result = map_band_name("sentinel-2", "sentinel-2", "NONEXISTENT")
        assert result is None

    def test_cross_sensor_b02_finds_closest_blue(self):
        sensors = list_sensors()
        if len(sensors) < 2:
            pytest.skip("Only one sensor registered")
        other = [s for s in sensors if s != "sentinel-2"][0]
        result = map_band_name("sentinel-2", other, "B02")
        # Should find the closest blue band (or None if >50nm away)
        if result is not None:
            target_cfg = get_sensor(other)
            assert result in target_cfg.bands
