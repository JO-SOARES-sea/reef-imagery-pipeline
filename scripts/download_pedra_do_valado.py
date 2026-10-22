#!/usr/bin/env python3
"""
scripts/download_pedra_do_valado.py — Download gratuito de amostra Sentinel-2 L2A (Pedra do Valado, Algarve).

Utiliza o catálogo STAC aberto da Planetary Computer para extrair um recorte 4-bandas
(B02 Azul, B03 Verde, B04 Vermelho, B08 NIR) centrado no Parque Marinho da Pedra do Valado / Armação de Pêra.
Grava diretamente em: data/raw/pedra_do_valado_sample.tif
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_sample")

# Coordenadas WGS84 completas do Parque Marinho da Pedra do Valado (Benagil a Albufeira)
# Cobre os 36 cumes de recife mapeados por multifeixe do CCMAR
PEDRA_DO_VALADO_BBOX_WGS84 = (-8.44, 37.03, -8.28, 37.12)  # (min_lon, min_lat, max_lon, max_lat)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "pedra_do_valado_sample.tif"


def fetch_sample_scene(
    bbox: tuple[float, float, float, float] = PEDRA_DO_VALADO_BBOX_WGS84,
    date_range: str = "2025-07-01/2025-09-30",
    max_cloud: float = 3.0,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Conectando ao catálogo STAC (Planetary Computer) com assinatura aberta...")
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

    logger.info(f"Procurando cenas Sentinel-2 L2A no Algarve (Tile 29SNB, Nuvens < {max_cloud}%)...")
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=date_range,
        query={"s2:mgrs_tile": {"eq": "29SNB"}, "eo:cloud_cover": {"lt": max_cloud}},
        limit=5,
    )
    items = list(search.items())

    if not items:
        raise RuntimeError("Nenhuma cena encontrada para os critérios especificados.")

    # Escolhe a cena com menor cobertura de nuvens
    selected_item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    scene_id = selected_item.id
    cloud_cover = selected_item.properties.get("eo:cloud_cover", 0)
    acq_date = selected_item.datetime.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Cena selecionada: {scene_id} ({acq_date}) | Cobertura de nuvens: {cloud_cover:.1f}%")

    bands = ["B02", "B03", "B04", "B08"]
    arrays = []
    out_profile = None

    for i, band in enumerate(bands, start=1):
        if band not in selected_item.assets:
            raise KeyError(f"Banda {band} ausente nos ativos da cena {scene_id}.")

        asset_url = selected_item.assets[band].href
        logger.info(f"Lendo banda {band} via Cloud-Optimized GeoTIFF (COG)...")

        with rasterio.open(asset_url) as src:
            if out_profile is None:
                # Transforma a BBox de EPSG:4326 para o CRS nativo da cena (ex: UTM 29N / EPSG:32629)
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326", src.crs, bbox[0], bbox[1], bbox[2], bbox[3]
                )
                window = from_bounds(left, bottom, right, top, transform=src.transform)

                out_profile = src.profile.copy()
                out_transform = rasterio.windows.transform(window, src.transform)
                h_win, w_win = int(round(window.height)), int(round(window.width))

                out_profile.update(
                    driver="GTiff",
                    height=h_win,
                    width=w_win,
                    count=len(bands),
                    dtype=rasterio.float32,
                    transform=out_transform,
                    nodata=0,
                )

            # Leitura por janela (baixa apenas a área do recife, ~alguns MBs)
            arr = src.read(1, window=window).astype(np.float32)
            # Sentinel-2 L2A baseline >= 04.00: DN 0 é nodata; BOA_ADD_OFFSET = -1000 DN (-0.1 reflectância)
            arr = np.where(arr > 0, np.maximum((arr - 1000.0) * 0.0001, 0.0001), 0.0)
            arrays.append(arr)

    # Empilha as 4 bandas (B02, B03, B04, B08)
    stack = np.stack(arrays, axis=0)

    logger.info(f"Salvando GeoTIFF recortado em: {OUTPUT_FILE} ({stack.shape[1]}x{stack.shape[2]} px)...")
    with rasterio.open(OUTPUT_FILE, "w", **out_profile) as dst:
        dst.write(stack)
        dst.set_band_description(1, "B02_Blue")
        dst.set_band_description(2, "B03_Green")
        dst.set_band_description(3, "B04_Red")
        dst.set_band_description(4, "B08_NIR")

    logger.info("Download e recorte da Pedra do Valado concluídos com sucesso!")
    return OUTPUT_FILE


if __name__ == "__main__":
    try:
        fetch_sample_scene()
    except Exception as e:
        logger.error(f"Erro ao descarregar cena: {e}", exc_info=True)
        sys.exit(1)
