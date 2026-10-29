#!/usr/bin/env python3
"""
scripts/download_s2_zone.py — Download gratuito e recorte Sentinel-2 L2A (Planetary Computer)
para qualquer zona costeira de recife do Algarve.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_s2_zone")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ZONES = {
    "valado": {
        "name": "Pedra do Valado (Armação de Pêra)",
        "bbox": (-8.44, 37.03, -8.28, 37.12),
        "output_file": RAW_DIR / "pedra_do_valado_sample.tif",
    },
    "greta": {
        "name": "Pedra da Greta (Ilha de Faro)",
        "bbox": (-8.02, 36.96, -7.96, 37.02),
        "output_file": RAW_DIR / "pedra_da_greta_sample.tif",
    },
    "santa_eulalia": {
        "name": "Pedra de Santa Eulália (Albufeira)",
        "bbox": (-8.26, 37.04, -8.17, 37.10),
        "output_file": RAW_DIR / "santa_eulalia_sample.tif",
    },
    "quarteira": {
        "name": "Recifes de Quarteira / Vilamoura / Garrão",
        "bbox": (-8.15, 37.02, -8.05, 37.08),
        "output_file": RAW_DIR / "quarteira_sample.tif",
    },
}


def native_window(bounds: tuple[float, float, float, float], transform) -> Window:
    """Janela de píxeis inteiros que cobre ``bounds`` (left, bottom, right, top na CRS da cena).

    Uma janela fracionária (from_bounds direto) devolve os píxeis nativos mas rotula-os com uma origem
    deslocada até 1 píxel da grelha Sentinel-2: erro de georreferenciação de alguns metros.
    """
    w = from_bounds(*bounds, transform=transform)
    col0, row0 = math.floor(w.col_off + 1e-6), math.floor(w.row_off + 1e-6)
    col1 = math.ceil(w.col_off + w.width - 1e-6)
    row1 = math.ceil(w.row_off + w.height - 1e-6)
    return Window(col0, row0, col1 - col0, row1 - row0)


def download_zone_sample(
    zone_key: str,
    date_range: str = "2025-07-01/2025-09-30",
    max_cloud: float = 3.0,
    force: bool = False,
) -> Path:
    if zone_key not in ZONES:
        raise ValueError(f"Zona desconhecida: {zone_key}. Zonas disponíveis: {list(ZONES.keys())}")

    cfg = ZONES[zone_key]
    out_file = cfg["output_file"]
    bbox = cfg["bbox"]
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        logger.info(f"Ficheiro existente encontrado: {out_file} (utilize --force para substituir).")
        return out_file

    logger.info(f"Iniciando download para {cfg['name']}...")
    logger.info("Conectando ao catálogo STAC (Planetary Computer)...")
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=date_range,
        query={"s2:mgrs_tile": {"eq": "29SNB"}, "eo:cloud_cover": {"lt": max_cloud}},
        limit=5,
    )
    items = list(search.items())
    if not items:
        # Tenta intervalo mais amplo se não encontrar no verão 2025
        logger.warning("Nenhuma cena no intervalo padrão; procurando entre 2023 e 2026...")
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime="2023-05-01/2026-05-30",
            query={"s2:mgrs_tile": {"eq": "29SNB"}, "eo:cloud_cover": {"lt": max_cloud}},
            limit=5,
        )
        items = list(search.items())

    if not items:
        raise RuntimeError(f"Nenhuma cena Sentinel-2 encontrada para {zone_key}.")

    selected_item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    scene_id = selected_item.id
    cloud_cover = selected_item.properties.get("eo:cloud_cover", 0)
    logger.info(f"Cena selecionada: {scene_id} | Cobertura de nuvens: {cloud_cover:.1f}%")

    bands = ["B02", "B03", "B04", "B08"]
    arrays = []
    out_profile = None

    for band in bands:
        if band not in selected_item.assets:
            raise KeyError(f"Banda {band} ausente no item {scene_id}")
        asset_url = selected_item.assets[band].href
        logger.info(f"Baixando janela COG da banda {band}...")

        with rasterio.open(asset_url) as src:
            if out_profile is None:
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326", src.crs, bbox[0], bbox[1], bbox[2], bbox[3]
                )
                window = native_window((left, bottom, right, top), src.transform)

                out_profile = src.profile.copy()
                out_transform = rasterio.windows.transform(window, src.transform)
                h_win, w_win = int(window.height), int(window.width)

                out_profile.update(
                    driver="GTiff",
                    height=h_win,
                    width=w_win,
                    count=len(bands),
                    dtype=rasterio.float32,
                    transform=out_transform,
                    nodata=0,
                )

            arr = src.read(1, window=window).astype(np.float32)
            arr = np.where(arr > 0, np.maximum((arr - 1000.0) * 0.0001, 0.0001), 0.0)
            arrays.append(arr)

    stack = np.stack(arrays, axis=0)
    with rasterio.open(out_file, "w", **out_profile) as dst:
        dst.write(stack)
        dst.set_band_description(1, "B02_Blue")
        dst.set_band_description(2, "B03_Green")
        dst.set_band_description(3, "B04_Red")
        dst.set_band_description(4, "B08_NIR")

    logger.info(f"GeoTIFF guardado com sucesso em: {out_file} ({stack.shape[1]}x{stack.shape[2]} px)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download de cena Sentinel-2 L2A para zona do Algarve.")
    parser.add_argument("--zone", choices=list(ZONES.keys()), default="santa_eulalia", help="Zona costeira")
    parser.add_argument("--force", action="store_true", help="Força novo download")
    args = parser.parse_args()

    download_zone_sample(args.zone, force=args.force)


if __name__ == "__main__":
    main()
