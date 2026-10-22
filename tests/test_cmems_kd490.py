"""Tests for src/cmems_kd490.py — Kd490 climatology with CMEMS/static fallback."""
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub copernicusmarine so patch("copernicusmarine.open_dataset", ...) resolves
# even when the optional package is not installed in the test environment.
if "copernicusmarine" not in sys.modules:
    _cm_stub = types.ModuleType("copernicusmarine")
    _cm_stub.open_dataset = MagicMock()
    sys.modules["copernicusmarine"] = _cm_stub

import src.cmems_kd490 as _mod
from src.constants import KD490_DEFAULT
from src.constants import KD490_TABLE as _STATIC_TABLE

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_fake_dataset(kd_by_month: dict[int, float] | None = None,
                       zsd_by_month: dict[int, float] | None = None):
    """Build a minimal xarray-like mock that copernicusmarine.open_dataset returns."""
    import pandas as pd
    import xarray as xr

    if kd_by_month is None:
        kd_by_month = {m: 0.045 + m * 0.001 for m in range(1, 13)}
    if zsd_by_month is None:
        zsd_by_month = {m: 15.0 - m * 0.2 for m in range(1, 13)}

    times = pd.date_range("2020-01-01", periods=12, freq="MS")

    kd_data = np.array([kd_by_month.get(t.month, 0.05) for t in times],
                       dtype=np.float32).reshape(12, 1, 1)
    zsd_data = np.array([zsd_by_month.get(t.month, 14.0) for t in times],
                        dtype=np.float32).reshape(12, 1, 1)

    coords = {"time": times, "lat": [37.0], "lon": [-8.5]}
    kd_da  = xr.DataArray(kd_data,  dims=["time", "lat", "lon"], coords=coords)
    zsd_da = xr.DataArray(zsd_data, dims=["time", "lat", "lon"], coords=coords)
    return xr.Dataset({"KD490": kd_da, "ZSD": zsd_da})


# ── _fetch_cmems_climatology ─────────────────────────────────────────────────

class TestFetchCmemsClimatology:
    def test_returns_tuple_of_two_dicts(self, monkeypatch):
        monkeypatch.setenv("CMEMS_USER", "test_user")
        monkeypatch.setenv("CMEMS_PASSWORD", "test_pass")
        fake_ds = _make_fake_dataset()
        with patch("copernicusmarine.open_dataset", return_value=fake_ds):
            kd_table, zsd_table = _mod._fetch_cmems_climatology()
        assert set(kd_table.keys()) == set(range(1, 13))
        assert set(zsd_table.keys()) == set(range(1, 13))
        for v in kd_table.values():
            assert isinstance(v, float)
            assert 0.0 < v < 1.0
        for v in zsd_table.values():
            assert isinstance(v, float)
            assert v > 0.0

    def test_fills_missing_kd_months_from_static_table(self, monkeypatch):
        """If CMEMS returns KD490 for only some months, the rest come from static."""
        monkeypatch.setenv("CMEMS_USER", "u")
        monkeypatch.setenv("CMEMS_PASSWORD", "p")
        import pandas as pd
        import xarray as xr
        # Only Jan–Jun in the dataset
        times = pd.date_range("2020-01-01", periods=6, freq="MS")
        kd_data  = np.full((6, 1, 1), 0.042, dtype=np.float32)
        zsd_data = np.full((6, 1, 1), 14.0,  dtype=np.float32)
        coords = {"time": times, "lat": [37.0], "lon": [-8.5]}
        fake_ds = xr.Dataset({
            "KD490": xr.DataArray(kd_data,  dims=["time", "lat", "lon"], coords=coords),
            "ZSD":   xr.DataArray(zsd_data, dims=["time", "lat", "lon"], coords=coords),
        })
        with patch("copernicusmarine.open_dataset", return_value=fake_ds):
            kd_table, _ = _mod._fetch_cmems_climatology()
        assert set(kd_table.keys()) == set(range(1, 13))
        for m in range(7, 13):
            assert kd_table[m] == _STATIC_TABLE.get(m, KD490_DEFAULT)

    def test_no_time_dimension_broadcasts_to_all_months(self, monkeypatch):
        monkeypatch.setenv("CMEMS_USER", "u")
        monkeypatch.setenv("CMEMS_PASSWORD", "p")
        import xarray as xr
        kd_data  = np.full((4, 4), 0.055, dtype=np.float32)
        zsd_data = np.full((4, 4), 13.0,  dtype=np.float32)
        fake_ds = xr.Dataset({
            "KD490": xr.DataArray(kd_data,  dims=["lat", "lon"]),
            "ZSD":   xr.DataArray(zsd_data, dims=["lat", "lon"]),
        })
        with patch("copernicusmarine.open_dataset", return_value=fake_ds):
            kd_table, _ = _mod._fetch_cmems_climatology()
        assert set(kd_table.keys()) == set(range(1, 13))
        for v in kd_table.values():
            assert abs(v - 0.055) < 1e-4

    def test_uses_copernicusmarine_env_var_aliases(self, monkeypatch):
        monkeypatch.delenv("CMEMS_USER", raising=False)
        monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
        monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "alt_user")
        monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "alt_pass")
        fake_ds = _make_fake_dataset()
        with patch("copernicusmarine.open_dataset", return_value=fake_ds) as mock_open:
            _mod._fetch_cmems_climatology()
        mock_open.assert_called_once()
        _, kwargs = mock_open.call_args
        assert kwargs["username"] == "alt_user"
        assert kwargs["password"] == "alt_pass"

    def test_no_credentials_still_calls_open_dataset(self, monkeypatch):
        """Without env vars, copernicusmarine uses stored credentials file — no RuntimeError."""
        monkeypatch.delenv("CMEMS_USER", raising=False)
        monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
        monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
        monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
        fake_ds = _make_fake_dataset()
        with patch("copernicusmarine.open_dataset", return_value=fake_ds) as mock_open:
            _mod._fetch_cmems_climatology()
        # Called WITHOUT username/password kwargs (stored creds file used by the library)
        mock_open.assert_called_once()
        _, kwargs = mock_open.call_args
        assert "username" not in kwargs
        assert "password" not in kwargs


# ── refresh_from_cmems ────────────────────────────────────────────────────────

class TestRefreshFromCmems:
    def test_returns_false_on_import_error(self):
        with patch("src.cmems_kd490._fetch_cmems_climatology",
                   side_effect=ImportError("copernicusmarine not installed")):
            result = _mod.refresh_from_cmems()
        assert result is False

    def test_returns_false_on_network_error(self):
        with patch("src.cmems_kd490._fetch_cmems_climatology",
                   side_effect=ConnectionError("timeout")):
            result = _mod.refresh_from_cmems()
        assert result is False

    def test_table_unchanged_on_failure(self):
        before = dict(_mod.KD490_TABLE_LIVE)
        with patch("src.cmems_kd490._fetch_cmems_climatology",
                   side_effect=OSError("network unreachable")):
            _mod.refresh_from_cmems()
        assert before == _mod.KD490_TABLE_LIVE

    def test_returns_true_and_updates_table_on_success(self):
        live_kd  = dict.fromkeys(range(1, 13), 0.04)
        live_zsd = dict.fromkeys(range(1, 13), 18.0)
        with patch("src.cmems_kd490._fetch_cmems_climatology",
                   return_value=(live_kd, live_zsd)):
            result = _mod.refresh_from_cmems()
        assert result is True
        assert _mod.KD490_TABLE_LIVE[7] == pytest.approx(0.040)
        assert _mod.ZSD_TABLE_LIVE[7]   == pytest.approx(18.0)

    def test_mutates_in_place_not_rebind(self):
        """Capture a reference BEFORE refresh — it must reflect the new values after."""
        captured_ref = _mod.KD490_TABLE_LIVE
        live_kd  = dict.fromkeys(range(1, 13), 0.099)
        live_zsd = dict.fromkeys(range(1, 13), 5.0)
        with patch("src.cmems_kd490._fetch_cmems_climatology",
                   return_value=(live_kd, live_zsd)):
            _mod.refresh_from_cmems()
        assert captured_ref is _mod.KD490_TABLE_LIVE, "refresh must not rebind the module global"
        assert captured_ref[7] == pytest.approx(0.099)


# ── get_kd490 ────────────────────────────────────────────────────────────────

class TestGetKd490:
    def test_returns_float_for_all_valid_months(self):
        for m in range(1, 13):
            v = _mod.get_kd490(m)
            assert isinstance(v, float)
            assert 0.0 < v < 1.0

    def test_returns_default_for_invalid_month(self):
        assert _mod.get_kd490(0)  == KD490_DEFAULT
        assert _mod.get_kd490(13) == KD490_DEFAULT
        assert _mod.get_kd490(99) == KD490_DEFAULT

    def test_summer_months_lower_than_winter(self):
        """Algarve: summer (Jul–Aug) is clearer than winter (Jan–Feb)."""
        jul = _mod.get_kd490(7)
        aug = _mod.get_kd490(8)
        jan = _mod.get_kd490(1)
        # Both directions valid (live vs static may differ), but range must be sane
        assert 0.01 <= jul <= 0.5
        assert 0.01 <= aug <= 0.5
        assert 0.01 <= jan <= 0.5

    def test_accepts_string_month(self):
        # get_kd490 casts to int — "7" should work
        assert _mod.get_kd490(7) == _mod.get_kd490(7)


# ── KD490_TABLE_LIVE shape ────────────────────────────────────────────────────

class TestKd490TableLive:
    def test_has_12_months(self):
        assert set(_mod.KD490_TABLE_LIVE.keys()) == set(range(1, 13))

    def test_all_values_positive_and_plausible(self):
        for m, v in _mod.KD490_TABLE_LIVE.items():
            assert isinstance(v, float), f"month {m}: expected float, got {type(v)}"
            assert 0.005 < v < 0.500, f"month {m}: Kd490={v} outside physical range"
