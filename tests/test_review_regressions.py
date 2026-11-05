"""Regressões da revisão completa: bootstrap por cume, extração por janela, holdout fail-closed, imports lazy."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from src.analysis.reef_segmenter import _bootstrap_gain_ci, extract_candidate_reefs

ROOT = Path(__file__).resolve().parent.parent


def test_cluster_bootstrap_is_wider_than_pixel_bootstrap():
    """Pixels de um mesmo cume são correlacionados: reamostrar cumes dá um IC mais largo."""
    rng = np.random.default_rng(0)
    n_groups, per = 8, 40
    groups = np.repeat(np.arange(n_groups), per)
    y = np.tile(np.r_[np.ones(per // 2, int), np.zeros(per // 2, int)], n_groups)
    shift = rng.normal(0, 1.5, n_groups)[groups]  # efeito por cume partilhado pelos seus pixels
    model = y * 1.0 + shift + rng.normal(0, 0.3, len(y))
    base = rng.normal(0, 1, len(y))
    lo_p, hi_p = _bootstrap_gain_ci(y, model, base, n_boot=300)
    lo_c, hi_c = _bootstrap_gain_ci(y, model, base, n_boot=300, groups=groups)
    assert (hi_c - lo_c) > (hi_p - lo_p)


def test_bootstrap_without_valid_replicates_returns_nan():
    y = np.ones(6, int)
    lo, hi = _bootstrap_gain_ci(y, np.arange(6.0), np.arange(6.0), n_boot=20, groups=np.arange(6))
    assert np.isnan(lo) and np.isnan(hi)


def test_extract_candidates_windowed_mask_matches_stats():
    proba = np.zeros((60, 80), dtype=np.float32)
    proba[10:20, 5:15] = 0.9
    proba[10, 5] = 1.0
    proba[40:50, 60:79] = 0.6  # junto à borda direita
    profile = {"transform": Affine.translation(500000.0, 4100000.0) @ Affine.scale(10.0, -10.0), "crs": "EPSG:32629"}
    gdf = extract_candidate_reefs(proba, profile, threshold=0.5, min_area_m2=500.0)
    assert len(gdf) == 2
    by_max = sorted(gdf["prob_max"].tolist())
    assert by_max == [0.6, 1.0]
    first = gdf.sort_values("prob_max").iloc[1]
    assert first["prob_mean"] == pytest.approx((0.9 * 99 + 1.0) / 100, abs=1e-3)


def test_reef_detector_import_does_not_pull_heavy_packages():
    code = (
        "import sys; sys.path.insert(0, %r); import src.analysis.features; "
        "print(any(m in sys.modules for m in ('geopandas', 'sklearn', 'PIL', 'src.analysis.reef_segmenter')))" % str(ROOT)
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_analysis_lazy_exports_still_resolve():
    pkg = importlib.import_module("src.analysis")
    assert callable(pkg.extract_candidate_reefs)
    assert callable(pkg.compute_vrm)
    with pytest.raises(AttributeError):
        pkg.does_not_exist  # noqa: B018


def test_quarteira_report_fstring_has_no_undefined_names():
    """Regressão: ``\\text{ m}`` dentro de f-string avaliava a variável ``m`` (NameError)."""
    path = ROOT / "scripts" / "generate_quarteira_garrao_probability_surface.py"
    if not path.exists():
        pytest.skip("script ainda não rastreado neste checkout")
    src = path.read_text(encoding="utf-8")
    assert r"\text{ m}" not in src.replace(r"\text{{ m}}", "")
    ast.parse(src)


def test_olex_without_holdout_column_fails_closed(tmp_path, monkeypatch):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        mod = importlib.import_module("persistent_structures_ml")
    except Exception as exc:  # dependências/dados opcionais em falta
        pytest.skip(f"persistent_structures_ml não importável offline: {exc}")
    gj = tmp_path / "olex.geojson"
    gpd.GeoDataFrame(
        {"ground_truth_class": ["CONFIRMED_HARD_POSITIVE"]}, geometry=[Point(-8.0, 37.0)], crs="EPSG:4326"
    ).to_file(gj, driver="GeoJSON")
    monkeypatch.setattr(mod, "OLEX_GT", gj)
    with pytest.raises(RuntimeError, match="is_holdout"):
        mod.reference_geometries()


def test_calculate_physics_score_is_public_and_matches_ranker_composition():
    """Regressão: hybrid_stac_physical_orchestrator e evaluate_all_shortlist importam este nome."""
    from datetime import datetime

    from src.ranking_model import predict_score
    from src.reef_ml_predictor import calculate_physics_score, extract_features_from_stac

    clear = {"cloud_cover": 2.0, "sun_elevation": 55.0, "datetime": datetime(2025, 9, 25, 11, 0)}
    cloudy = {**clear, "cloud_cover": 95.0}
    expected = predict_score(extract_features_from_stac(clear, 16.0))["score"]
    assert calculate_physics_score(clear, 16.0) == pytest.approx(expected)
    assert calculate_physics_score(clear, 16.0) > calculate_physics_score(cloudy, 16.0)
