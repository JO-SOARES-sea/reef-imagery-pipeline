"""Offline tests for scripts/forecast_visibility.py physics and day aggregation."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import forecast_visibility as fv  # noqa: E402


def _hourly(day: str, height: float, period: float, direction: float) -> dict:
    times = [f"{day}T{h:02d}:00" for h in range(24)]
    return {
        "time": times,
        "wave_height": [height] * 24,
        "wave_period": [period] * 24,
        "wave_direction": [direction] * 24,
    }


def _merge(*parts: dict) -> dict:
    return {k: sum((p[k] for p in parts), []) for k in parts[0]}


def test_wavenumber_satisfies_dispersion_relation():
    k = fv.wavenumber(8.0, 12.0)
    omega = 2 * math.pi / 8.0
    assert omega**2 == pytest.approx(9.81 * k * math.tanh(k * 12.0), rel=1e-6)


def test_orbital_velocity_decreases_with_depth_and_vanishes_for_flat_sea():
    assert fv.near_bed_orbital_velocity(1.0, 8.0, 6.0) > fv.near_bed_orbital_velocity(1.0, 8.0, 20.0)
    assert fv.near_bed_orbital_velocity(0.0, 8.0, 12.0) == 0.0


def test_long_period_swell_reaches_the_bed_more_than_short_chop():
    assert fv.near_bed_orbital_velocity(0.7, 10.0, 12.0) > fv.near_bed_orbital_velocity(0.7, 4.0, 12.0)


def test_exposure_factor_head_on_vs_oblique():
    assert fv.exposure_factor(180, 180) == pytest.approx(1.0)
    assert fv.exposure_factor(270, 180) == pytest.approx(math.sqrt(0.3))
    assert fv.exposure_factor(350, 10) == pytest.approx(fv.exposure_factor(30, 10))


@pytest.mark.parametrize("vis,label", [(15, "Excelente"), (9, "Boa"), (6, "Razoável"), (3, "Fraca")])
def test_visibility_label(vis, label):
    assert fv.visibility_label(vis) == label


def test_calm_sea_gives_calm_visibility():
    out = fv.forecast_site(_hourly("2026-10-17", 0.1, 4.0, 180), 180, {}, near_ria=False)
    assert out["2026-10-17"]["visibility_m"] == pytest.approx(round(fv.calm_visibility(10), 1))


def test_storm_turbidity_carries_over_to_the_next_calm_day():
    hourly = _merge(_hourly("2026-10-16", 1.5, 7.0, 180), _hourly("2026-10-17", 0.1, 4.0, 180))
    out = fv.forecast_site(hourly, 180, {}, near_ria=False)
    assert out["2026-10-16"]["visibility_m"] < out["2026-10-17"]["visibility_m"] < round(fv.calm_visibility(10), 1)


def test_rain_penalty_is_stronger_near_the_ria_formosa():
    hourly = _hourly("2026-10-18", 0.1, 4.0, 180)
    rain = {"2026-10-18": 90.0}
    dry = fv.forecast_site(hourly, 180, {}, near_ria=True)["2026-10-18"]["visibility_m"]
    coast = fv.forecast_site(hourly, 180, rain, near_ria=False)["2026-10-18"]["visibility_m"]
    ria = fv.forecast_site(hourly, 180, rain, near_ria=True)["2026-10-18"]["visibility_m"]
    assert ria < coast < dry


def test_rain_penalty_persists_into_the_next_day_only():
    hourly = _merge(*(_hourly(d, 0.1, 4.0, 180) for d in ("2026-10-18", "2026-10-19", "2026-10-20")))
    out = fv.forecast_site(hourly, 180, {"2026-10-18": 90.0}, near_ria=False)
    calm = round(fv.calm_visibility(10), 1)
    assert out["2026-10-18"]["visibility_m"] < calm
    assert out["2026-10-19"]["visibility_m"] < calm
    assert out["2026-10-20"]["visibility_m"] == pytest.approx(calm)


def test_label_matches_the_rounded_visibility():
    # 7.96 m must not print as "8.0 m Razoável"
    hourly = _hourly("2026-10-17", 0.1, 4.0, 180)
    for site_day in fv.forecast_site(hourly, 180, {"2026-10-17": 90.0}, near_ria=True).values():
        assert site_day["label"] == fv.visibility_label(site_day["visibility_m"])


def test_load_sites_skips_duplicate_locations_and_moves_coastal_cells_offshore():
    sites = {s["name"]: s for s in fv.load_sites()}
    assert "albufeira_reef" not in sites  # same coordinates as pedra_sta_eulalia
    assert (sites["salgados"]["lat"], sites["salgados"]["lon"]) == fv.OFFSHORE_POINTS["salgados"]
