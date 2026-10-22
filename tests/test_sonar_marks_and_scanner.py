"""
Testes unitários para o parser de coordenadas náuticas (DMM/Decimal),
gestão do catálogo de marcas de sonda e integração do scanner de coordenadas.
"""

import pytest
from pathlib import Path
from src.navigation.sonar_marks import parse_coordinates, to_dmm, calculate_distance_and_bearing, SonarMarkCatalog


def test_parse_dmm_various_formats():
    # Formato fornecido pelo utilizador
    lat1, lon1 = parse_coordinates("n36,57,898 007,57,579w")
    assert lat1 == pytest.approx(36.964967, abs=1e-5)
    assert lon1 == pytest.approx(-7.959650, abs=1e-5)

    # Formato tradicional Furuno / DMM com graus e plicas
    lat2, lon2 = parse_coordinates("36°57.898'N 007°57.579'W")
    assert lat2 == pytest.approx(lat1, abs=1e-5)
    assert lon2 == pytest.approx(lon1, abs=1e-5)

    # Formato decimal direto
    lat3, lon3 = parse_coordinates("36.964967, -7.959650")
    assert lat3 == pytest.approx(lat1, abs=1e-5)
    assert lon3 == pytest.approx(lon1, abs=1e-5)


def test_to_dmm_formatting():
    dmm_str = to_dmm(36.964967, -7.959650)
    assert "36°57,898' N" in dmm_str
    assert "007°57,579' W" in dmm_str


def test_distance_and_bearing():
    # Mesma posição -> distância 0
    d, b = calculate_distance_and_bearing(36.964967, -7.959650, 36.964967, -7.959650)
    assert d == pytest.approx(0.0)

    # Movimento para Norte (~111 km por grau)
    d_north, b_north = calculate_distance_and_bearing(36.0, -8.0, 37.0, -8.0)
    assert d_north == pytest.approx(111195.0, abs=1000.0)
    assert b_north == pytest.approx(0.0, abs=0.5)


def test_sonar_mark_catalog_tmp(tmp_path: Path):
    csv_file = tmp_path / "test_catalog.csv"
    gpx_file = tmp_path / "test_catalog.gpx"

    cat = SonarMarkCatalog(csv_path=csv_file, gpx_path=gpx_file)
    rec = cat.add_mark("Lages", 36.964967, -7.959650, depth_m=13.6, source_type="sonda")

    assert rec["nome"] == "Lages"
    assert "36°57,898' N" in rec["plotter_dmm"]
    assert csv_file.exists()
    assert gpx_file.exists()

    gpx_content = gpx_file.read_text(encoding="utf-8")
    assert "<name>Lages</name>" in gpx_content
    assert 'lat="36.964967"' in gpx_content
