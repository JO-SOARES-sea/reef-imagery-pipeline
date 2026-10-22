"""
Módulo de Navegação e Registo de Marcas de Sonda / Waypoints GPS.
Suporta conversão universal entre Graus e Minutos Decimais (DMM - Furuno, Garmin, Raymarine)
e Graus Decimais (WGS84), cálculo de rumos e distâncias geodésicas, e persistência em catálogo GPX e CSV.
"""

from __future__ import annotations

import csv
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG_CSV = PROJECT_ROOT / "data" / "vector" / "sonar_marks_catalog.csv"
DEFAULT_CATALOG_GPX = PROJECT_ROOT / "data" / "vector" / "sonar_marks_catalog.gpx"


def parse_coordinates(coord_str: str) -> tuple[float, float]:
    """
    Interpreta coordenadas em múltiplos formatos comuns na navegação e plotters:
      - DMM com vírgulas/pontos: "n36,57,898 007,57,579w" ou "36°57.898'N 007°57.579'W"
      - Separado por espaço/hífen: "N 36 57.898 W 07 57.579"
      - Decimal direto: "36.964967, -7.959650" ou "36.964967 -7.959650"
    Retorna:
      (lat, lon) em graus decimais (float, float).
    """
    s = coord_str.strip()

    # 1. Tenta formato decimal simples: "36.964967, -7.959650"
    dec_match = re.match(r"^([+-]?\d+\.\d+)[,\s]+([+-]?\d+\.\d+)$", s)
    if dec_match:
        lat, lon = float(dec_match.group(1)), float(dec_match.group(2))
        return lat, lon

    # Normaliza separadores de símbolos para parsing de DMM
    clean = s.replace("°", " ").replace("'", " ").replace('"', " ").replace("’", " ")

    # Padrão flexível para DMM (Graus e Minutos Decimais)
    # Suporta: [N/S] graus [,.] minutos [N/S] e [W/E/O] graus [,.] minutos [W/E/O]
    pattern = re.compile(
        r"([NSns]?)\s*(\d{1,2})[,\s]+(\d{1,2}(?:[.,]\d+)?)\s*([NSns]?)[,\s/]+"
        r"([WEOweo]?)\s*(\d{1,3})[,\s]+(\d{1,2}(?:[.,]\d+)?)\s*([WEOweo]?)",
        re.IGNORECASE,
    )

    m = pattern.search(clean)
    if not m:
        # Tenta split mais solto se vier apenas números com hemisférios
        # Ex: "n36,57,898 007,57,579w" -> extrai partes norte e oeste
        tokens = re.findall(r"[NSWEOnsweo]?\d+(?:[.,]\d+)?", s)
        if len(tokens) >= 2:
            return _parse_loose_tokens(tokens)
        raise ValueError(
            f"Não foi possível interpretar a coordenada: '{coord_str}'. "
            f"Exemplos válidos: 'n36,57,898 007,57,579w' ou '36.964967, -7.959650'"
        )

    ns_pre, lat_d, lat_m, ns_post, we_pre, lon_d, lon_m, we_post = m.groups()

    lat_min_float = float(lat_m.replace(",", "."))
    lon_min_float = float(lon_m.replace(",", "."))

    lat_deg = float(lat_d) + (lat_min_float / 60.0)
    lon_deg = float(lon_d) + (lon_min_float / 60.0)

    ns = (ns_pre or ns_post or "N").upper()
    we = (we_pre or we_post or "W").upper()

    if ns == "S":
        lat_deg = -lat_deg
    if we in ("W", "O"):
        lon_deg = -lon_deg

    return lat_deg, lon_deg


def _parse_loose_tokens(tokens: list[str]) -> tuple[float, float]:
    """Auxiliar para tokens do tipo ['n36,57,898', '007,57,579w']."""
    lat_tok = tokens[0].lower().replace("n", "").replace("s", "")
    lon_tok = tokens[1].lower().replace("w", "").replace("e", "").replace("o", "")

    # Divide por vírgulas ou pontos
    lat_parts = re.split(r"[,.]", lat_tok)
    lon_parts = re.split(r"[,.]", lon_tok)

    if len(lat_parts) >= 3:
        lat_deg = float(lat_parts[0])
        lat_min = float(f"{lat_parts[1]}.{lat_parts[2]}")
        lat = lat_deg + (lat_min / 60.0)
    else:
        lat = float(".".join(lat_parts))

    if len(lon_parts) >= 3:
        lon_deg = float(lon_parts[0])
        lon_min = float(f"{lon_parts[1]}.{lon_parts[2]}")
        lon = lon_deg + (lon_min / 60.0)
    else:
        lon = float(".".join(lon_parts))

    # Verifica hemisférios originais
    if "s" in tokens[0].lower():
        lat = -lat
    if "w" in tokens[1].lower() or "o" in tokens[1].lower():
        lon = -lon
    elif not tokens[1].startswith("-") and lon > 0 and lon < 20:
        # No Algarve / Portugal continental longitude é sempre Oeste (negativa)
        lon = -lon

    return lat, lon


def to_dmm(lat: float, lon: float) -> str:
    """
    Converte coordenadas decimais para formato padrão de Plotter Náutico (Furuno/Garmin):
    Exemplo: 36°57,898' N 007°57,579' W
    """
    def _fmt(val: float, deg_width: int, pos_hemi: str, neg_hemi: str) -> str:
        hemi = pos_hemi if val >= 0 else neg_hemi
        abs_v = abs(val)
        d = int(abs_v)
        m = (abs_v - d) * 60.0
        return f"{d:0{deg_width}d}°{m:06.3f}'".replace(".", ",") + f" {hemi}"

    return f"{_fmt(lat, 2, 'N', 'S')} {_fmt(lon, 3, 'E', 'W')}"


def calculate_distance_and_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """
    Calcula a distância geodésica em metros e o rumo (bearing) em graus (0° a 360°).
    """
    r = 6371000.0  # Raio médio da Terra em metros
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Fórmula de Haversine
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    dist_m = r * c

    # Rumo inicial (bearing)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - (math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))
    bearing_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    return dist_m, bearing_deg


class SonarMarkCatalog:
    """Catálogo persistente de marcas de sonda e waypoints de recifes."""

    def __init__(self, csv_path: Path = DEFAULT_CATALOG_CSV, gpx_path: Path = DEFAULT_CATALOG_GPX):
        self.csv_path = csv_path
        self.gpx_path = gpx_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.gpx_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "nome",
                    "lat",
                    "lon",
                    "plotter_dmm",
                    "profundidade_m",
                    "tipo_origem",
                    "data_registo",
                    "notas",
                ])

    def add_mark(
        self,
        name: str,
        lat: float,
        lon: float,
        depth_m: float | None = None,
        source_type: str = "sonda_barco",
        notes: str = "",
    ) -> dict[str, Any]:
        """Adiciona ou atualiza uma marca no catálogo CSV e GPX."""
        plotter_str = to_dmm(lat, lon)
        depth_val = f"{depth_m:.1f}" if depth_m is not None else ""
        today = datetime.now().strftime("%Y-%m-%d %H:%M")

        record = {
            "nome": name,
            "lat": f"{lat:.6f}",
            "lon": f"{lon:.6f}",
            "plotter_dmm": plotter_str,
            "profundidade_m": depth_val,
            "tipo_origem": source_type,
            "data_registo": today,
            "notas": notes,
        }

        # Lê marcas existentes para evitar duplicações exatas
        existing = []
        if self.csv_path.exists():
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing = list(reader)

        # Atualiza se já existir com o mesmo nome, ou adiciona
        updated = False
        for i, row in enumerate(existing):
            if row["nome"].strip().lower() == name.strip().lower():
                existing[i] = record
                updated = True
                break

        if not updated:
            existing.append(record)

        # Grava CSV
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nome",
                    "lat",
                    "lon",
                    "plotter_dmm",
                    "profundidade_m",
                    "tipo_origem",
                    "data_registo",
                    "notas",
                ],
            )
            writer.writeheader()
            writer.writerows(existing)

        # Atualiza GPX
        self._export_gpx(existing)
        return record

    def _export_gpx(self, records: list[dict[str, str]]) -> None:
        """Gera ficheiro GPX 1.1 compatível com plotters Furuno / Garmin / Raymarine."""
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="ReefImageryPipeline" xmlns="http://www.topografix.com/GPX/1/1">',
            '  <metadata>',
            '    <name>Marcas de Sonda e Recifes Naturais do Algarve</name>',
            f'    <time>{datetime.utcnow().isoformat()}Z</time>',
            '  </metadata>',
        ]

        for r in records:
            lat = r["lat"]
            lon = r["lon"]
            name = r["nome"]
            cmt = f"Sonda: {r['profundidade_m']}m | {r['plotter_dmm']} | {r['notas']}"
            xml_lines.append(f'  <wpt lat="{lat}" lon="{lon}">')
            xml_lines.append(f'    <name>{name}</name>')
            xml_lines.append(f'    <cmt>{cmt}</cmt>')
            xml_lines.append(f'    <sym>Rock</sym>')
            xml_lines.append('  </wpt>')

        xml_lines.append('</gpx>')
        self.gpx_path.write_text("\n".join(xml_lines), encoding="utf-8")
