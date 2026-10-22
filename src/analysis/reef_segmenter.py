"""
src/analysis/reef_segmenter.py — Segmentação Semântica e Detetor de Recifes Marinhos.

Integra:
1. Máscara de água NDWI (Sentinel-2) + Filtro de profundidade (2 m a 25 m EMODnet).
2. Índice de probabilidade de recife combinando VRM (rugosidade 3D), BPI (cume), e textura B03.
3. Extração vetorial de polígonos/aglomerados de candidatos a recife natural.
4. Validação em tempo real contra o Ground Truth do CCMAR (36 cumes multifeixe).
5. Roteamento e decisão ultrarrápida com o TypeSafe Jev (Nouls).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from src.analysis.features import extract_morpho_spectral_features, resample_emodnet_screen
from src.ingestion.sentinel_reader import calculate_ndwi, read_sentinel_bands, save_geotiff

logger = logging.getLogger("reef_pipeline.analysis.segmenter")


def normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalização Min-Max percentílica (p2 - p98) robusta a outliers."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p2, p98 = np.percentile(valid, (2, 98))
    if p98 == p2:
        return np.zeros_like(arr, dtype=np.float32)
    norm = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
    return np.where(np.isfinite(arr), norm, np.nan).astype(np.float32)


def compute_reef_probability_map(
    features: dict[str, np.ndarray],
    water_mask: np.ndarray,
    depth_mask: np.ndarray,
) -> np.ndarray:
    """
    Combina rugosidade, relevo local e contraste espectral (bio-cobertura) em um mapa de probabilidade contínua.
    """
    logger.info("Calculando mapa ponderado de probabilidade de recife...")

    analysis_mask = water_mask & depth_mask

    # Rácio Verde / Vermelho (indicador de bio-cobertura vegetal/calcária e contraste rocha/areia)
    with np.errstate(invalid="ignore", divide="ignore"):
        gr_ratio = np.where(features["B04"] > 0.005, features["B03"] / features["B04"], 1.0)

    def _norm(arr: np.ndarray) -> np.ndarray:
        valid = arr[analysis_mask & np.isfinite(arr)]
        if len(valid) == 0:
            return np.zeros_like(arr, dtype=np.float32)
        p5, p95 = np.percentile(valid, (5, 95))
        if p95 == p5:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - p5) / (p95 - p5), 0.0, 1.0).astype(np.float32)

    gr_norm = _norm(gr_ratio)
    tex_norm = _norm(features.get("B03_std30", np.zeros_like(gr_norm)))
    psdb_norm = _norm(features["psdb"])
    slope_norm = _norm(features.get("psdb_slope", np.zeros_like(gr_norm)))
    vrm_norm = _norm(features.get("psdb_vrm50", np.zeros_like(gr_norm)))
    bpi_norm = _norm(features.get("psdb_bpi100", np.zeros_like(gr_norm)))
    z_norm = _norm(features.get("z_b02_b03", np.zeros_like(gr_norm)))

    # Combinação ponderada de evidências morfológicas e espectrais de última geração
    # Dá peso relevante à rugosidade 3D (VRM) e cumes de relevo (BPI) mapeados nos recifes do Algarve
    score = (
        0.25 * gr_norm +      # Contraste espectral e bio-cobertura
        0.20 * vrm_norm +     # Rugosidade 3D do relevo submarino (VRM50)
        0.20 * bpi_norm +     # Cume topográfico sobrelevado (BPI100)
        0.15 * tex_norm +     # Textura focal de alta frequência
        0.10 * slope_norm +   # Declive acentuado
        0.10 * z_norm         # Invariância z-score na coluna de água
    )

    # Anula pixels fora da área de interesse (terra firme ou água fora de 2-25m)
    reef_proba = np.where(analysis_mask, score, 0.0).astype(np.float32)
    return reef_proba


def extract_candidate_reefs(
    reef_proba: np.ndarray,
    profile: dict[str, Any],
    threshold: float = 0.55,
    min_area_m2: float = 500.0,
) -> gpd.GeoDataFrame:
    """Extrai aglomerados e polígonos contíguos de recifes prováveis."""
    logger.info(f"Vetorizando candidatos com probabilidade >= {threshold}...")
    binary_mask = (reef_proba >= threshold).astype(np.uint8)

    transform = profile["transform"]
    crs = profile["crs"]

    records = []
    for geom_dict, val in shapes(binary_mask, mask=(binary_mask == 1), transform=transform):
        geom = shape(geom_dict)
        area_m2 = geom.area
        if area_m2 >= min_area_m2:
            records.append({
                "geometry": geom,
                "area_m2": round(area_m2, 1),
            })

    if not records:
        logger.warning("Nenhum candidato a recife encontrado com o limiar especificado.")
        return gpd.GeoDataFrame(columns=["geometry", "area_m2"], crs=crs)

    gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info(f"Detetados {len(gdf)} aglomerados de recife natural (área mínima: {min_area_m2} m²).")
    return gdf
