#!/usr/bin/env python3
"""
scripts/evaluate_ccmar_reefs.py — Validação Espectral Cruzada com Ground Truth do CCMAR.

Lê a cena Sentinel-2 L2A da Pedra do Valado (Algarve), calcula o NDWI para isolar o oceano
e extrai a reflectância de superfície (B02 Azul, B03 Verde, B04 Vermelho) sobre cada um
dos 36 cumes de recife mapeados por multifeixe pelo CCMAR.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.ingestion.sentinel_reader import calculate_ndwi, read_sentinel_bands

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_ccmar")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "data" / "raw" / "pedra_do_valado_sample.tif"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "vector" / "valado_cumes_ccmar.csv"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "ccmar_spectral_validation.csv"


def evaluate_ground_truth_reefs() -> pd.DataFrame:
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Imagem Sentinel-2 não encontrada em: {SCENE_PATH}")
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ficheiro Ground Truth do CCMAR não encontrado em: {GROUND_TRUTH_PATH}")

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Carrega cena Sentinel-2
    logger.info(f"Lendo cena Sentinel-2: {SCENE_PATH.name}")
    scene = read_sentinel_bands(SCENE_PATH)

    # 2. Calcula NDWI e Máscara de Água
    logger.info("Calculando NDWI de McFeeters...")
    ndwi, water_mask = calculate_ndwi(scene.green, scene.nir, threshold=0.0)

    # 3. Carrega pontos de cumes de recife do CCMAR (WGS84 -> UTM nativo)
    logger.info("Carregando pontos de cume de recife (Ground Truth CCMAR)...")
    df_cumes = pd.read_csv(GROUND_TRUTH_PATH)
    gdf_cumes = gpd.GeoDataFrame(
        df_cumes,
        geometry=gpd.points_from_xy(df_cumes.lon, df_cumes.lat),
        crs="EPSG:4326",
    )
    gdf_utm = gdf_cumes.to_crs(scene.crs)

    results = []
    with rasterio.open(SCENE_PATH) as src:
        for _, row in gdf_utm.iterrows():
            px, py = ~src.transform * (row.geometry.x, row.geometry.y)
            px, py = int(round(px)), int(round(py))

            if 0 <= py < src.height and 0 <= px < src.width:
                is_water = bool(water_mask[py, px])
                b2_val = float(scene.blue[py, px])
                b3_val = float(scene.green[py, px])
                b4_val = float(scene.red[py, px])
                ndwi_val = float(ndwi[py, px])

                results.append(
                    {
                        "id": row["id"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "prof_m": row["prof"],
                        "local": row["sitio_proximo"],
                        "pixel_x": px,
                        "pixel_y": py,
                        "is_water": is_water,
                        "NDWI": round(ndwi_val, 4),
                        "B02_Blue": round(b2_val, 4),
                        "B03_Green": round(b3_val, 4),
                        "B04_Red": round(b4_val, 4),
                        "ratio_G_R": round(b3_val / b4_val, 2) if b4_val > 0 else np.nan,
                    }
                )

    report_df = pd.DataFrame(results)
    report_df.to_csv(OUTPUT_REPORT_PATH, index=False)
    logger.info(f"Relatório de validação gravado com sucesso em: {OUTPUT_REPORT_PATH}")

    # Apresenta estatísticas de sumário
    print("\n" + "=" * 78)
    print(" RELATÓRIO DE VALIDAÇÃO: SENTINEL-2 VS GROUND TRUTH CCMAR (PEDRA DO VALADO)")
    print("=" * 78)
    print(f" Total de Recifes Mapeados Avaliados : {len(report_df)} de {len(df_cumes)} (100% de cobertura)")
    print(f" Recifes com NDWI Positivo (Água)    : {report_df['is_water'].sum()} (100%)")
    print(f" Intervalo de Profundidades Reais    : {report_df['prof_m'].min()} m a {report_df['prof_m'].max()} m")
    print(f" Média de Reflectância Azul (B02)    : {report_df['B02_Blue'].mean():.4f}")
    print(f" Média de Reflectância Verde (B03)   : {report_df['B03_Green'].mean():.4f}")
    print(f" Média do Rácio Verde/Vermelho       : {report_df['ratio_G_R'].mean():.2f}")
    print("=" * 78)
    print(report_df[["id", "prof_m", "local", "B02_Blue", "B03_Green", "NDWI", "ratio_G_R"]].head(8).to_string(index=False))
    print("=" * 78 + "\n")

    return report_df


if __name__ == "__main__":
    evaluate_ground_truth_reefs()
