"""Tests for src/ipma_sea_state.py — IPMA sea-state filter."""
from unittest.mock import MagicMock, patch

import pytest

import src.ipma_sea_state as _mod

# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _clear_caches():
    """Clear lru_cache on fetch functions so mocks take effect."""
    _mod._fetch_wave_forecast.cache_clear()
    _mod._fetch_wind_obs.cache_clear()


@pytest.fixture(autouse=True)
def clear_cache():
    _clear_caches()
    yield
    _clear_caches()


def _wave_station(lat="37.19", lon="-8.00", wave_max="0.8", wave_min="0.5",
                  total_sea=1.0, period_max="5.0", dir_="W", sst="18.0",
                  station_id=1080526):
    return {
        "globalIdLocal": station_id,
        "latitude": lat, "longitude": lon,
        "waveHighMax": wave_max, "waveHighMin": wave_min,
        "totalSeaMax": total_sea, "totalSeaMin": total_sea - 0.2,
        "wavePeriodMax": period_max, "wavePeriodMin": "4.0",
        "predWaveDir": dir_,
        "sstMax": sst, "sstMin": str(float(sst) - 0.5),
    }


def _wind_obs(speed_ms=5.0, station_id="1080526"):
    """Return a minimal IPMA wind observation dict using an Algarve station ID."""
    return {
        "2026-06-06T12:00": {
            str(station_id): {"intensidadeVento": speed_ms, "temperatura": 20.0,
                              "humidade": 60.0, "pressao": 1013.0},
        }
    }


def _mock_wave_response(stations: list[dict]):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": stations, "forecastDate": "2026-06-06"}
    return resp


def _mock_wind_response(obs: dict):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = obs
    return resp


# ── _fetch_wave_forecast ──────────────────────────────────────────────────────

class TestFetchWaveForecast:
    def test_returns_station_list_on_success(self):
        stations = [_wave_station()]
        with patch("requests.get", return_value=_mock_wave_response(stations)):
            result = _mod._fetch_wave_forecast()
        assert len(result) == 1
        assert result[0]["globalIdLocal"] == 1080526

    def test_returns_empty_list_on_http_error(self):
        import requests as _req
        with patch("requests.get", side_effect=_req.RequestException("timeout")):
            result = _mod._fetch_wave_forecast()
        assert result == []

    def test_result_is_cached(self):
        stations = [_wave_station()]
        with patch("requests.get", return_value=_mock_wave_response(stations)) as mock_get:
            _mod._fetch_wave_forecast()
            _mod._fetch_wave_forecast()
        # Only one real HTTP call despite two invocations
        assert mock_get.call_count == 1


# ── _fetch_wind_obs ───────────────────────────────────────────────────────────

class TestFetchWindObs:
    def test_returns_dict_on_success(self):
        obs = _wind_obs(8.0)
        with patch("requests.get", return_value=_mock_wind_response(obs)):
            result = _mod._fetch_wind_obs()
        assert "2026-06-06T12:00" in result

    def test_returns_empty_dict_on_network_error(self):
        import requests as _req
        with patch("requests.get", side_effect=_req.RequestException("connection refused")):
            result = _mod._fetch_wind_obs()
        assert result == {}


# ── _algarve_wave_conditions ──────────────────────────────────────────────────

class TestAlgarveWaveConditions:
    def test_matches_by_station_id(self):
        algarve = _wave_station(station_id=1080526, wave_max="1.2")
        other   = _wave_station(station_id=9999999, lat="40.0", lon="-5.0", wave_max="3.0")
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[algarve, other]):
            result = _mod._algarve_wave_conditions()
        assert result["station_id"] == 1080526
        assert result["wave_high_max_m"] == 1.2

    def test_matches_by_lat_lon_when_id_not_in_known_set(self):
        # Station with unknown ID but Algarve coordinates
        unknown = _wave_station(station_id=5555555, lat="37.05", lon="-8.50", wave_max="0.9")
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[unknown]):
            result = _mod._algarve_wave_conditions()
        assert result["wave_high_max_m"] == 0.9

    def test_returns_worst_case_when_multiple_algarve_stations(self):
        s1 = _wave_station(station_id=1080526, wave_max="0.7")
        s2 = _wave_station(station_id=1081526, wave_max="2.1", lon="-8.94")
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[s1, s2]):
            result = _mod._algarve_wave_conditions()
        assert result["wave_high_max_m"] == 2.1

    def test_returns_empty_dict_when_no_algarve_stations(self):
        non_alg = _wave_station(station_id=9999, lat="41.0", lon="-5.0")
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[non_alg]):
            result = _mod._algarve_wave_conditions()
        assert result == {}

    def test_returns_empty_dict_when_forecast_empty(self):
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[]):
            result = _mod._algarve_wave_conditions()
        assert result == {}

    def test_nan_wave_height_coerced_to_zero(self):
        """API returning 'nan' or float NaN for waveHighMax must not propagate NaN."""
        s = _wave_station(wave_max="nan")  # IPMA sometimes returns "nan" as string
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[s]):
            result = _mod._algarve_wave_conditions()
        assert result["wave_high_max_m"] == 0.0
        # And the usable flag in get_conditions must not be False due to NaN
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[s]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value={}):
            cond = _mod.get_conditions()
        assert cond["usable"] is True  # 0.0 <= 1.5 → usable

    def test_all_expected_keys_present(self):
        s = _wave_station()
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[s]):
            result = _mod._algarve_wave_conditions()
        for key in ("wave_high_max_m", "wave_high_min_m", "total_sea_max_m",
                    "pred_wave_dir", "sst_max_c", "station_id"):
            assert key in result


# ── _algarve_wind_ms ─────────────────────────────────────────────────────────

class TestAlgarveWindMs:
    def test_returns_max_speed_from_algarve_stations(self):
        """Only Algarve station IDs contribute to the max wind reading."""
        obs = {
            "2026-06-06T12:00": {
                "1080526": {"intensidadeVento": 5.0},    # Faro — Algarve ✓
                "1081526": {"intensidadeVento": 12.3},   # Sagres — Algarve ✓
                "9999999": {"intensidadeVento": 25.0},   # mountain — NOT Algarve ✗
            }
        }
        with patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod._algarve_wind_ms()
        assert result == 12.3  # mountain station excluded

    def test_ignores_non_algarve_stations(self):
        """A mountain station with extreme wind must not reject an Algarve scene."""
        obs = {
            "2026-06-06T12:00": {
                "9999999": {"intensidadeVento": 25.0},  # non-Algarve station only
            }
        }
        with patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod._algarve_wind_ms()
        assert result is None  # no Algarve station data → treat as unavailable

    def test_returns_none_when_obs_empty(self):
        with patch("src.ipma_sea_state._fetch_wind_obs", return_value={}):
            result = _mod._algarve_wind_ms()
        assert result is None

    def test_uses_latest_timestamp(self):
        obs = {
            "2026-06-06T06:00": {"1080526": {"intensidadeVento": 3.0}},
            "2026-06-06T18:00": {"1080526": {"intensidadeVento": 9.0}},
        }
        with patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod._algarve_wind_ms()
        assert result == 9.0

    def test_skips_stations_with_none_speed(self):
        obs = {
            "2026-06-06T12:00": {
                "1080526": {"intensidadeVento": None},
                "1081526": {"intensidadeVento": 4.5},
            }
        }
        with patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod._algarve_wind_ms()
        assert result == 4.5


# ── get_conditions ────────────────────────────────────────────────────────────

class TestGetConditions:
    def test_returns_all_required_keys(self):
        wave = _wave_station(wave_max="0.8")
        obs  = _wind_obs(5.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod.get_conditions("2026-06-06")
        for key in ("wave_high_max_m", "total_sea_max_m", "pred_wave_dir",
                    "wind_speed_ms", "sst_max_c", "source", "fetch_date", "usable"):
            assert key in result

    def test_usable_true_when_calm(self):
        wave = _wave_station(wave_max="0.5")
        obs  = _wind_obs(3.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod.get_conditions()
        assert result["usable"] is True

    def test_usable_false_when_high_waves(self):
        wave = _wave_station(wave_max="2.5")
        obs  = _wind_obs(3.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod.get_conditions()
        assert result["usable"] is False

    def test_usable_false_when_high_wind(self):
        wave = _wave_station(wave_max="0.5")
        obs  = _wind_obs(15.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            result = _mod.get_conditions()
        assert result["usable"] is False

    def test_usable_true_when_wind_unavailable(self):
        """No wind data should not penalise the scene."""
        wave = _wave_station(wave_max="0.8")
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value={}):
            result = _mod.get_conditions()
        assert result["usable"] is True
        assert result["wind_speed_ms"] is None

    def test_source_field_is_ipma(self):
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value={}):
            result = _mod.get_conditions()
        assert result["source"] == "IPMA day-0 forecast"


# ── is_scene_usable ───────────────────────────────────────────────────────────

class TestIsSceneUsable:
    def test_returns_true_for_calm_conditions(self):
        wave = _wave_station(wave_max="0.6")
        obs  = _wind_obs(4.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            assert _mod.is_scene_usable("2026-06-06") is True

    def test_returns_false_when_wave_exceeds_threshold(self):
        wave = _wave_station(wave_max="2.0")
        obs  = _wind_obs(3.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            assert _mod.is_scene_usable("2026-06-06", max_wave_m=1.5) is False

    def test_returns_false_when_wind_exceeds_threshold(self):
        wave = _wave_station(wave_max="0.5")
        obs  = _wind_obs(12.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            assert _mod.is_scene_usable("2026-06-06", max_wind_ms=10.0) is False

    def test_fail_open_when_no_wave_data(self):
        """If IPMA is unreachable the scene must be accepted (fail-open)."""
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value={}):
            assert _mod.is_scene_usable("2026-06-06") is True

    def test_custom_thresholds_respected(self):
        wave = _wave_station(wave_max="1.0")
        obs  = _wind_obs(6.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            # Tight threshold → reject
            assert _mod.is_scene_usable(max_wave_m=0.5, max_wind_ms=5.0) is False
            _clear_caches()
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            # Relaxed threshold → accept
            assert _mod.is_scene_usable(max_wave_m=2.0, max_wind_ms=15.0) is True

    def test_exactly_at_threshold_is_accepted(self):
        """Boundary: wave_high_max == max_wave_m should be accepted (not strictly >)."""
        wave = _wave_station(wave_max="1.5")
        obs  = _wind_obs(3.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            assert _mod.is_scene_usable(max_wave_m=1.5) is True

    def test_date_parameter_does_not_affect_result(self):
        """date is for logging only — same conditions produce same result regardless."""
        wave = _wave_station(wave_max="0.7")
        obs  = _wind_obs(3.0)
        with patch("src.ipma_sea_state._fetch_wave_forecast", return_value=[wave]), \
             patch("src.ipma_sea_state._fetch_wind_obs", return_value=obs):
            r1 = _mod.is_scene_usable("2024-01-01")
            r2 = _mod.is_scene_usable("2025-12-31")
        assert r1 == r2 == True
