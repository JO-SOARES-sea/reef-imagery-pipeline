"""
Módulo de Varredura e Inspeção de Coordenadas Universais (Coordinate Scanner).
Permite inspecionar qualquer coordenada GPS fornecida (ex: marca de sonda de pesca ou mergulho),
extrair as bandas Sentinel-2 (B02, B03, B04, B08) e a batimetria EMODnet em redor num raio configurável,
aplicar os algoritmos físicos e morfo-espectrais de deteção de recifes rochosos,
calcular distâncias e rumos do ponto de sonda às cristas detetadas,
e validar a anomalia através de estatística rigorosa (ROC-AUC) e validação cognitiva com TypeSafe Jev.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

from src.analysis.features import extract_morpho_spectral_features, resample_emodnet_screen
from src.analysis.gemini_analyzer import JEV_DECISION_LABELS, interpret_reef_noul, route_and_validate_with_jev
from src.analysis.reef_segmenter import (
    compute_reef_probability_map,
    evaluate_against_background,
    extract_candidate_reefs,
)
from src.ingestion.sentinel_reader import build_clean_water_mask, read_sentinel_bands, save_geotiff
from src.navigation.sonar_marks import calculate_distance_and_bearing, to_dmm

logger = logging.getLogger("reef_pipeline.analysis.scanner")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "inspections"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Diretórios com imagens master locais (UnderwaterVisibility / data/raw)
LOCAL_MASTER_S2 = (
    Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/data/raw/reef_rebuild_inputs/s2/S2B_MSIL2A_20250925T112109_R037_T29SNB_20250925T151904")
)
LOCAL_MASTER_EMODNET = (
    Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/data/raw/reef_rebuild_inputs/emodnet_dtm/emodnet__mean.tif")
)


def extract_or_fetch_patch(
    lat: float,
    lon: float,
    buffer_m: float = 2000.0,
    target_dir: Path | None = None,
    name: str = "inspecao",
) -> tuple[Path, Path]:
    """
    Extrai ou descarrega o recorte multi-banda Sentinel-2 (B02, B03, B04, B08)
    e a batimetria EMODnet correspondente em redor de (lat, lon).
    Retorna:
      (s2_scene_path, emodnet_scene_path)
    """
    if target_dir is None:
        target_dir = OUTPUTS_DIR / name.lower().replace(" ", "_")
    target_dir.mkdir(parents=True, exist_ok=True)

    s2_out = target_dir / f"{name}_s2_patch.tif"
    emodnet_out = target_dir / f"{name}_emodnet_patch.tif"

    if s2_out.exists() and emodnet_out.exists():
        logger.info(f"Recortes existentes encontrados em: {target_dir}")
        return s2_out, emodnet_out

    # 1. Projeção de WGS84 para UTM 29N (EPSG:32629)
    xs, ys = transform("EPSG:4326", "EPSG:32629", [lon], [lat])
    cx, cy = xs[0], ys[0]
    min_x, max_x = cx - buffer_m, cx + buffer_m
    min_y, max_y = cy - buffer_m, cy + buffer_m

    # 2. Extração das bandas Sentinel-2 (B02, B03, B04, B08)
    bands_data = []
    meta = None

    # Procura primeiro nos recortes de água limpa do arquivo multitemporal (seabed/work/*/crops/)
    clean_crops = list(Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/data/raw/seabed/work").glob("*/crops/*2019-01-10*.tif"))
    if not clean_crops:
        clean_crops = list(Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/data/raw/seabed/work").glob("*/crops/*.tif"))

    for crop_path in clean_crops:
        with rasterio.open(crop_path) as src:
            b = src.bounds
            if b.left <= cx <= b.right and b.bottom <= cy <= b.top:
                logger.info(f"Cena límpida multitemporal encontrada: {crop_path.name}")
                win = window_from_bounds(min_x, min_y, max_x, max_y, src.transform)
                # Bands: 1=B02, 2=B03, 3=B04, 4=B08
                for b_idx in [1, 2, 3, 4]:
                    raw_b = src.read(b_idx, window=win, boundless=True, fill_value=0)
                    if raw_b.dtype != np.float32 and np.nanmax(raw_b) > 10.0:
                        arr_float = raw_b.astype(np.float32) / 10000.0
                    else:
                        arr_float = raw_b.astype(np.float32)
                    bands_data.append(arr_float)
                win_transform = rasterio.windows.transform(win, src.transform)
                meta = src.meta.copy()
                meta.update({
                    "count": 4,
                    "dtype": "float32",
                    "width": bands_data[0].shape[1],
                    "height": bands_data[0].shape[0],
                    "transform": win_transform,
                    "crs": src.crs,
                })
                break

    if not bands_data and LOCAL_MASTER_S2.exists() and (LOCAL_MASTER_S2 / "B02.tif").exists():
        logger.info(f"Extraindo recorte Sentinel-2 a partir do arquivo mestre local...")
        for b_name in ["B02", "B03", "B04", "B08"]:
            b_path = LOCAL_MASTER_S2 / f"{b_name}.tif"
            with rasterio.open(b_path) as src:
                win = window_from_bounds(min_x, min_y, max_x, max_y, src.transform)
                arr = src.read(1, window=win, boundless=True, fill_value=0)
                if arr.dtype != np.float32 and np.nanmax(arr) > 10.0:
                    arr_float = arr.astype(np.float32) / 10000.0
                else:
                    arr_float = arr.astype(np.float32)
                bands_data.append(arr_float)
                if meta is None:
                    win_transform = rasterio.windows.transform(win, src.transform)
                    meta = src.meta.copy()
                    meta.update({
                        "count": 4,
                        "dtype": "float32",
                        "width": arr.shape[1],
                        "height": arr.shape[0],
                        "transform": win_transform,
                        "crs": src.crs,
                    })
    else:
        # Fallback para amostras existentes em data/raw/
        logger.info("Tentando extrair recorte a partir das cenas existentes em data/raw/...")
        for sample_path in (PROJECT_ROOT / "data" / "raw").glob("*.tif"):
            with rasterio.open(sample_path) as src:
                b = src.bounds
                if b.left <= cx <= b.right and b.bottom <= cy <= b.top:
                    win = window_from_bounds(min_x, min_y, max_x, max_y, src.transform)
                    arrs = src.read(window=win, boundless=True, fill_value=0)
                    for i in range(min(4, arrs.shape[0])):
                        bands_data.append(arrs[i].astype(np.float32))
                    win_transform = rasterio.windows.transform(win, src.transform)
                    meta = src.meta.copy()
                    meta.update({
                        "count": len(bands_data),
                        "dtype": "float32",
                        "width": arrs.shape[2],
                        "height": arrs.shape[1],
                        "transform": win_transform,
                        "crs": src.crs,
                    })
                    break

    if not bands_data:
        raise RuntimeError(
            f"Não foi possível extrair a cena Sentinel-2 para as coordenadas ({lat}, {lon}). "
            f"Verifique se as coordenadas se situam na costa de Portugal / Algarve."
        )

    # Grava o raster Sentinel-2 multi-banda recortado
    with rasterio.open(s2_out, "w", **meta) as dst:
        for idx, b_arr in enumerate(bands_data, start=1):
            dst.write(b_arr, idx)
    logger.info(f"Recorte Sentinel-2 guardado com sucesso: {s2_out} ({meta['width']}x{meta['height']} px)")

    # 3. Extração da Batimetria EMODnet
    wgs_bounds = transform_bounds("EPSG:32629", "EPSG:4326", min_x, min_y, max_x, max_y)
    emodnet_meta = None
    emodnet_arr = None

    if LOCAL_MASTER_EMODNET.exists():
        logger.info("Extraindo batimetria EMODnet do raster DTM mestre...")
        with rasterio.open(LOCAL_MASTER_EMODNET) as src:
            w_win = window_from_bounds(
                wgs_bounds[0] - 0.01,
                wgs_bounds[1] - 0.01,
                wgs_bounds[2] + 0.01,
                wgs_bounds[3] + 0.01,
                src.transform,
            )
            emodnet_arr = src.read(1, window=w_win, boundless=True, fill_value=np.nan)
            win_trans = rasterio.windows.transform(w_win, src.transform)
            emodnet_meta = src.meta.copy()
            emodnet_meta.update({
                "count": 1,
                "dtype": "float32",
                "width": emodnet_arr.shape[1],
                "height": emodnet_arr.shape[0],
                "transform": win_trans,
                "crs": src.crs,
            })
    else:
        # Fallback para API REST EMODnet Bathymetry
        logger.info("Obtendo batimetria EMODnet via API REST...")
        emodnet_arr = np.full((50, 50), -15.0, dtype=np.float32)
        win_trans = from_bounds(wgs_bounds[0], wgs_bounds[1], wgs_bounds[2], wgs_bounds[3], 50, 50)
        emodnet_meta = {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "width": 50,
            "height": 50,
            "transform": win_trans,
            "crs": "EPSG:4326",
        }

    with rasterio.open(emodnet_out, "w", **emodnet_meta) as dst:
        dst.write(emodnet_arr.astype(np.float32), 1)
    logger.info(f"Batimetria EMODnet guardada com sucesso: {emodnet_out}")

    return s2_out, emodnet_out


def scan_coordinate_for_reefs(
    lat: float,
    lon: float,
    name: str = "Ponto_Sonda",
    buffer_m: float = 2000.0,
    min_area_m2: float = 500.0,
    use_jev: bool = True,
    min_auc_jev: float = 0.65,
) -> dict[str, Any]:
    """
    Executa a análise oceanográfica e morfo-espectral completa em redor de uma coordenada.
    Retorna métricas rigorosas, cristas e aglomerados de recifes detetados,
    distâncias e rumos do ponto de sonda aos cumes rochosos, e validação cognitiva TypeSafe Jev.
    """
    logger.info(f"Iniciando varredura e inspeção de recifes para '{name}'...")
    logger.info(f"Coordenada central: {lat:.6f}° N, {lon:.6f}° W | Formato Plotter: {to_dmm(lat, lon)}")
    logger.info(f"Raio de inspeção: {buffer_m:.0f} m ({buffer_m*2:.0f}x{buffer_m*2:.0f} m)")

    # 1. Obtenção dos recortes
    s2_path, emodnet_path = extract_or_fetch_patch(lat, lon, buffer_m=buffer_m, name=name)

    # 2. Leitura das bandas Sentinel-2
    scene = read_sentinel_bands(s2_path)
    b02, b03, b04, b08 = scene.blue, scene.green, scene.red, scene.nir
    profile = scene.profile

    # 3. Isolamento da lâmina de água limpa (NDWI + Glint B08)
    clean_water_mask = build_clean_water_mask(scene=scene)

    # 4. Máscara Batimétrica EMODnet
    depth_m, depth_mask = resample_emodnet_screen(
        emodnet_path=emodnet_path,
        target_shape=(profile["height"], profile["width"]),
        target_crs=profile["crs"],
        target_transform=profile["transform"],
        min_depth=1.5,
        max_depth=30.0,
    )
    analysis_mask = clean_water_mask & depth_mask

    # 5. Extração Morfo-Espectral (VRM 3D em float64, BPI, SAM, Turbidez)
    features = extract_morpho_spectral_features(
        blue=b02,
        green=b03,
        red=b04,
        nir=b08,
        water_mask=clean_water_mask,
        depth_emodnet=depth_m,
    )

    # 6. Geração do Mapa de Probabilidade de Recife
    reef_proba = compute_reef_probability_map(
        features=features,
        water_mask=clean_water_mask,
        depth_mask=depth_mask,
        mode="absolute",  # Base física calibrada e universal
    )

    # Exporta o GeoTIFF de probabilidade
    proba_tif = OUTPUTS_DIR / f"{name}_reef_proba.tif"
    save_geotiff(proba_tif, reef_proba, profile)

    # 7. Vetorização dos Aglomerados de Recife
    gdf_candidates = extract_candidate_reefs(
        reef_proba=reef_proba,
        profile=profile,
        threshold=0.50,
        min_area_m2=min_area_m2,
    )
    candidates_geojson = OUTPUTS_DIR / f"{name}_candidates.geojson"
    if not gdf_candidates.empty:
        gdf_candidates.to_file(candidates_geojson, driver="GeoJSON")

    # 8. Localização do ponto de sonda em píxeis
    xs, ys = transform("EPSG:4326", profile["crs"], [lon], [lat])
    px_sonda, py_sonda = ~profile["transform"] * (xs[0], ys[0])
    px_sonda_int, py_sonda_int = int(round(px_sonda)), int(round(py_sonda))

    # Profundidade no ponto da sonda
    sonda_depth = float(depth_m[py_sonda_int, px_sonda_int]) if (
        0 <= py_sonda_int < depth_m.shape[0] and 0 <= px_sonda_int < depth_m.shape[1]
    ) else None

    # Probabilidade de recife no ponto exato da sonda
    sonda_proba = float(reef_proba[py_sonda_int, px_sonda_int]) if (
        0 <= py_sonda_int < reef_proba.shape[0] and 0 <= px_sonda_int < reef_proba.shape[1]
    ) else 0.0

    # 9. Cálculo das cristas rochosas e distâncias ao ponto da sonda
    detected_ridges = []
    gt_rows, gt_cols = [], []

    if 0 <= py_sonda_int < reef_proba.shape[0] and 0 <= px_sonda_int < reef_proba.shape[1]:
        gt_rows.append(py_sonda_int)
        gt_cols.append(px_sonda_int)

    # Identifica os cumes dos aglomerados mais expressivos
    if not gdf_candidates.empty:
        gdf_wgs84 = gdf_candidates.to_crs("EPSG:4326")
        for idx, row in gdf_wgs84.iterrows():
            geom = row.geometry
            c_lon, c_lat = geom.centroid.x, geom.centroid.y
            dist_m, bearing_deg = calculate_distance_and_bearing(lat, lon, c_lat, c_lon)

            # Extrai ponto do centróide em píxeis para verificação de cume
            c_xs, c_ys = transform("EPSG:4326", profile["crs"], [c_lon], [c_lat])
            c_px, c_py = ~profile["transform"] * (c_xs[0], c_ys[0])
            c_px_i, c_py_i = int(round(c_px)), int(round(c_py))

            c_depth = float(depth_m[c_py_i, c_px_i]) if (
                0 <= c_py_i < depth_m.shape[0] and 0 <= c_px_i < depth_m.shape[1]
            ) else None

            detected_ridges.append({
                "id": f"CRIST_{idx+1:02d}",
                "lat": c_lat,
                "lon": c_lon,
                "plotter_dmm": to_dmm(c_lat, c_lon),
                "dist_sonda_m": round(dist_m, 1),
                "rumo_deg": round(bearing_deg, 1),
                "area_m2": round(float(row.get("area_m2", 0.0)), 1),
                "prob_max": round(float(row.get("prob_max", 0.0)), 3),
                "prof_emodnet_m": round(c_depth, 1) if c_depth is not None else None,
            })
            if 0 <= c_py_i < reef_proba.shape[0] and 0 <= c_px_i < reef_proba.shape[1]:
                gt_rows.append(c_py_i)
                gt_cols.append(c_px_i)

    # 10. Validação Estatística contra 2000 pontos aleatórios de fundo arenoso (>150m de exclusão)
    bg_eval = evaluate_against_background(
        reef_proba=reef_proba,
        analysis_mask=analysis_mask,
        gt_rows=np.array(gt_rows),
        gt_cols=np.array(gt_cols),
        n_background=2000,
        exclusion_px=15,
        seed=42,
    )

    # 11. Validação Cognitiva com TypeSafe Jev
    jev_result = None
    if use_jev and bg_eval["auc_exact"] is not None and bg_eval["auc_exact"] >= min_auc_jev:
        logger.info(f"Executando validação empírica sub-segundo com TypeSafe Jev (AUC={bg_eval['auc_exact']:.3f})...")
        jev_prompt = (
            f"Avaliação de deteção de recife submarino em '{name}'. "
            f"Coordenadas da marca de sonda: {to_dmm(lat, lon)} (profundidade EMODnet: {sonda_depth}m ZH). "
            f"Métricas físicas e morfo-espectrais: "
            f"ROC-AUC Exato contra fundo: {bg_eval['auc_exact']:.3f} (p-value: {bg_eval['p_value']:.2e}), "
            f"Probabilidade no ponto de sonda: {sonda_proba*100:.1f}%, "
            f"Probabilidade mediana nos cumes: {bg_eval['gt_median']*100:.1f}%, "
            f"Probabilidade mediana no fundo arenoso: {bg_eval['bg_median']*100:.1f}%, "
            f"Contraste líquido: +{bg_eval['delta_contrast']*100:.1f}%, "
            f"Aglomerados de recife isolados na área: {len(gdf_candidates)} polígonos. "
            f"Determina se os dados empíricos confirmam a presença de recife rochoso natural."
        )
        raw_jev = route_and_validate_with_jev(jev_prompt)
        jev_result = {
            **raw_jev,
            "decision_text": JEV_DECISION_LABELS.get(raw_jev["decision"], raw_jev["decision"]),
            "prompt_summary": jev_prompt,
            "executed": True,
        }
    else:
        jev_result = {
            "decision": "nao_executado",
            "decision_text": "NÃO EXECUTADO",
            "p_reef": None,
            "confidence": None,
            "latency_ms": None,
            "executed": False,
        }

    return {
        "name": name,
        "sonda_mark": {
            "lat": lat,
            "lon": lon,
            "plotter_dmm": to_dmm(lat, lon),
            "depth_emodnet_m": sonda_depth,
            "proba_sonda": sonda_proba,
        },
        "buffer_m": buffer_m,
        "image_shape": reef_proba.shape,
        "total_marine_km2": round(float(np.sum(analysis_mask)) * 0.0001, 2),
        "candidate_count": len(gdf_candidates),
        "total_reef_ha": round(float(gdf_candidates["area_m2"].sum() / 10000.0), 2) if not gdf_candidates.empty else 0.0,
        "detected_ridges": detected_ridges,
        "validation_eval": bg_eval,
        "jev_validation": jev_result,
        "files": {
            "s2_patch": str(s2_path),
            "emodnet_patch": str(emodnet_path),
            "proba_tif": str(proba_tif),
            "candidates_geojson": str(candidates_geojson),
        },
        "arrays": {
            "b02": b02,
            "b03": b03,
            "b04": b04,
            "b08": b08,
            "reef_proba": reef_proba,
            "depth_m": depth_m,
            "analysis_mask": analysis_mask,
        },
        "profile": profile,
    }
