#!/usr/bin/env python3
"""
scripts/run_reef_pipeline.py — Pipeline Rigoroso de Deteção e Validação de Recifes Naturais do Algarve.

Fluxo Científico:
  1. Ingestão Sentinel-2 (B02, B03, B04, B08) + Máscara NDWI + Máscara adaptativa B08 de glint e esteiras.
  2. Filtro de profundidade batimétrica EMODnet (2 m a 25 m).
  3. Extração das features morfo-espectrais submarinas calibradas em metros (pSDB, VRM 3D em float64, BPI, Textura).
  4. Classificação supervisionada (treinada em Ground Truth multifeixe) ou limiares físicos absolutos.
  5. Vetorização de polígonos candidatos a recife rochoso natural (GeoJSON).
  6. Validação estrita: ROC-AUC EXATO entre pontos de Ground Truth e 2000 pontos aleatórios do fundo (>150 m de recifes).
  7. Decisão e confirmação lógica via TypeSafe Jev condicionada a AUC >= 0.65 e baseada estritamente em medições reais.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv

from src.analysis.features import extract_morpho_spectral_features, resample_emodnet_screen
from src.analysis.gemini_analyzer import JEV_DECISION_LABELS, interpret_reef_noul
from src.analysis.reef_segmenter import (
    ReefClassifier,
    compute_reef_probability_map,
    evaluate_against_background,
    extract_candidate_reefs,
    train_multi_zone_classifier,
)
from src.ingestion.sentinel_reader import (
    build_clean_water_mask,
    calculate_ndwi,
    read_sentinel_bands,
    save_geotiff,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reef_pipeline.orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ZONE_CONFIGS = {
    "valado": {
        "name": "Pedra do Valado (Armação de Pêra)",
        "scene_path": PROJECT_ROOT / "data" / "raw" / "pedra_do_valado_sample.tif",
        "emodnet_path": PROJECT_ROOT / "data" / "processed" / "EMODnet_batimetria_Pedra_do_Valado.tif",
        "ground_truth_path": PROJECT_ROOT / "data" / "vector" / "valado_cumes_ccmar.csv",
        "ground_truth_type": "ccmar",
        "out_proba_tif": PROJECT_ROOT / "outputs" / "classifications" / "pedra_do_valado_reef_proba.tif",
        "out_candidates_geojson": PROJECT_ROOT / "outputs" / "classifications" / "pedra_do_valado_candidates.geojson",
        "out_report_csv": PROJECT_ROOT / "data" / "reports" / "pedra_do_valado_validation.csv",
        "description": "Recife da Pedra do Valado / Armação de Pêra",
        "target_depth_str": "10 m a 20 m",
    },
    "greta": {
        "name": "Pedra da Greta (Ilha de Faro)",
        "scene_path": PROJECT_ROOT / "data" / "raw" / "pedra_da_greta_sample.tif",
        "emodnet_path": PROJECT_ROOT / "data" / "processed" / "EMODnet_batimetria_Pedra_da_Greta.tif",
        "ground_truth_path": PROJECT_ROOT / "data" / "vector" / "greta_coordenadas.csv",
        "ground_truth_type": "greta",
        "out_proba_tif": PROJECT_ROOT / "outputs" / "classifications" / "pedra_da_greta_reef_proba.tif",
        "out_candidates_geojson": PROJECT_ROOT / "outputs" / "classifications" / "pedra_da_greta_candidates.geojson",
        "out_report_csv": PROJECT_ROOT / "data" / "reports" / "pedra_da_greta_validation.csv",
        "description": "Recife da Pedra da Greta / Ilha de Faro",
        "target_depth_str": "10 m a 16.5 m",
    },
    "santa_eulalia": {
        "name": "Pedra de Santa Eulália (Albufeira)",
        "scene_path": PROJECT_ROOT / "data" / "raw" / "santa_eulalia_sample.tif",
        "emodnet_path": PROJECT_ROOT / "data" / "processed" / "EMODnet_batimetria_Santa_Eulalia.tif",
        "ground_truth_path": PROJECT_ROOT / "data" / "vector" / "santa_eulalia_coordenadas.csv",
        "ground_truth_type": "santa_eulalia",
        "out_proba_tif": PROJECT_ROOT / "outputs" / "classifications" / "santa_eulalia_reef_proba.tif",
        "out_candidates_geojson": PROJECT_ROOT / "outputs" / "classifications" / "santa_eulalia_candidates.geojson",
        "out_report_csv": PROJECT_ROOT / "data" / "reports" / "santa_eulalia_validation.csv",
        "description": "Complexo de Recifes de Santa Eulália e Albufeira",
        "target_depth_str": "4 m a 15 m",
    },
    "quarteira": {
        "name": "Recifes de Quarteira / Vilamoura / Garrão",
        "scene_path": PROJECT_ROOT / "data" / "raw" / "quarteira_sample.tif",
        "emodnet_path": PROJECT_ROOT / "data" / "processed" / "EMODnet_batimetria_Quarteira.tif",
        "ground_truth_path": PROJECT_ROOT / "data" / "vector" / "quarteira_coordenadas.csv",
        "ground_truth_type": "quarteira",
        "out_proba_tif": PROJECT_ROOT / "outputs" / "classifications" / "quarteira_reef_proba.tif",
        "out_candidates_geojson": PROJECT_ROOT / "outputs" / "classifications" / "quarteira_candidates.geojson",
        "out_report_csv": PROJECT_ROOT / "data" / "reports" / "quarteira_validation.csv",
        "description": "Faixa de Recifes e Afloramentos de Quarteira a Vilamoura e Garrão",
        "target_depth_str": "6 m a 15 m",
    },
}


def verify_with_typesafe(state_description: str, question_text: str | None = None) -> dict:
    """Valida um aglomerado de recife com o TypeSafe Jev."""
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        logger.warning("TYPESAFE_API_KEY ausente; validação Jev não executada.")
        return {**interpret_reef_noul(None), "latency_ms": None}

    from typesafe_sdk import Noul, TypeSafeClient

    q_text = question_text or (
        "Com base nos dados instrumentais e na separação estatística face ao fundo arenoso, "
        "este candidato corresponde a um recife marinho natural comprovado?"
    )

    questions = {"is_natural_reef": Noul(instructions=q_text)}

    t0 = time.perf_counter()
    with TypeSafeClient(api_key=api_key) as client:
        result = client.system_one(state=state_description, questions=questions, model="jev-latest")
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        **interpret_reef_noul(result.nouls["is_natural_reef"].noul),
        "latency_ms": round(latency_ms, 1),
    }


def run_zone_pipeline(
    zone_key: str,
    cfg: dict[str, Any],
    classifier: ReefClassifier | None = None,
    mode: str = "supervised",
    no_jev: bool = False,
    min_auc_jev: float = 0.65,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" PIPELINE DE DETEÇÃO DE RECIFES: {cfg['name'].upper()}")
    print("=" * 80)

    # 1. Ingestão da Cena Sentinel-2 e Cálculo do NDWI + Glint Mask adaptativa
    logger.info("Passo 1/6: Carregando cena Sentinel-2 L2A e isolando coluna de água limpa (NDWI + Glint B08)...")
    scene = read_sentinel_bands(cfg["scene_path"])
    water_mask = build_clean_water_mask(scene, ndwi_threshold=0.0, apply_glint_filter=True)

    # 2. Resamostragem e Depth Screen do EMODnet
    logger.info("Passo 2/6: Aplicando máscara batimétrica EMODnet (2 m a 25 m)...")
    depth_m, depth_mask = resample_emodnet_screen(
        cfg["emodnet_path"],
        scene.shape,
        scene.crs,
        scene.transform,
        min_depth=2.0,
        max_depth=25.0,
    )
    analysis_mask = water_mask & depth_mask

    # 3. Extração das Features Morfo-Espectrais
    logger.info("Passo 3/6: Extraindo pSDB, VRM (Rugosidade 3D em float64), BPI (Cumes) e Textura...")
    features = extract_morpho_spectral_features(
        scene.blue, scene.green, scene.red, scene.nir, water_mask=water_mask, depth_emodnet=depth_m
    )

    # 4. Geração do Mapa de Probabilidade
    logger.info(f"Passo 4/6: Gerando mapa de probabilidade (modo: {mode}) e exportando GeoTIFF...")
    reef_proba = compute_reef_probability_map(
        features, water_mask, depth_mask, model=classifier, mode=mode
    )
    cfg["out_proba_tif"].parent.mkdir(parents=True, exist_ok=True)
    save_geotiff(cfg["out_proba_tif"], reef_proba, scene.profile, nodata=0.0)

    # 5. Segmentação Vetorial de Candidatos
    logger.info("Passo 5/6: Segmentando aglomerados de recifes rochosos...")
    threshold_polygon = 0.50 if mode == "supervised" else 0.45
    gdf_candidates = extract_candidate_reefs(reef_proba, scene.profile, threshold=threshold_polygon, min_area_m2=500.0)

    if not gdf_candidates.empty:
        cfg["out_candidates_geojson"].parent.mkdir(parents=True, exist_ok=True)
        gdf_candidates.to_file(cfg["out_candidates_geojson"], driver="GeoJSON")
        logger.info(f"Candidatos vetorizados gravados em: {cfg['out_candidates_geojson']}")

    # 6. Validação Cruzada com Ground Truth e Fundo Arenoso
    logger.info("Passo 6/6: Validando deteções contra Ground Truth e 2000 pontos aleatórios de fundo...")
    df_gt = pd.read_csv(cfg["ground_truth_path"])
    gdf_gt = gpd.GeoDataFrame(
        df_gt,
        geometry=gpd.points_from_xy(df_gt.lon, df_gt.lat),
        crs="EPSG:4326",
    ).to_crs(scene.crs)

    gt_rows, gt_cols = [], []
    validation_records = []

    with rasterio.open(cfg["out_proba_tif"]) as src:
        for idx, row in gdf_gt.iterrows():
            px, py = ~src.transform * (row.geometry.x, row.geometry.y)
            px, py = int(round(px)), int(round(py))
            if 0 <= py < src.height and 0 <= px < src.width:
                gt_rows.append(py)
                gt_cols.append(px)
                val_exact = float(reef_proba[py, px])
                d_val = float(depth_m[py, px]) if np.isfinite(depth_m[py, px]) else np.nan

                # Janela 3x3 para registo auxiliar
                y_min, y_max = max(0, py - 1), min(src.height, py + 2)
                x_min, x_max = max(0, px - 1), min(src.width, px + 2)
                local_max = float(np.nanmax(reef_proba[y_min:y_max, x_min:x_max]))

                item = {
                    "lon": row.lon,
                    "lat": row.lat,
                    "emodnet_depth_m": round(d_val, 1) if np.isfinite(d_val) else None,
                    "proba_exact": round(val_exact, 3),
                    "proba_3x3_max": round(local_max, 3),
                }
                if "plotter_nome" in row:
                    item["nome"] = row.get("plotter_nome")
                    item["tipo"] = row.get("tipo", "ground_truth")
                elif "cume_id" in row:
                    item["nome"] = f"Cume_{int(row.get('cume_id', idx))}"
                    item["tipo"] = "ccmar_multibeam"
                else:
                    item["nome"] = f"Ponto_{idx}"
                    item["tipo"] = "ground_truth"

                validation_records.append(item)

    # Executa validação estrita exata contra o fundo
    bg_eval = evaluate_against_background(
        reef_proba,
        analysis_mask,
        np.array(gt_rows),
        np.array(gt_cols),
        n_background=2000,
        exclusion_px=15,
        seed=42,
    )

    # Marcação nos registos com base no limiar de operação a FPR 5%
    fpr5_cut = bg_eval["fpr5_thresh"] if bg_eval["fpr5_thresh"] is not None else 0.50
    for item in validation_records:
        item["is_detected_fpr5"] = bool(item["proba_exact"] >= fpr5_cut)

    df_val = pd.DataFrame(validation_records)
    cfg["out_report_csv"].parent.mkdir(parents=True, exist_ok=True)
    df_val.to_csv(cfg["out_report_csv"], index=False)
    logger.info(f"Relatório de validação exportado para: {cfg['out_report_csv']}")

    # 7. Validação Cognitiva via TypeSafe Jev (Empírica e Condicionada)
    auc_exact = bg_eval["auc_exact"] or 0.0
    p_val = bg_eval["p_value"] or 1.0
    delta_contrast = bg_eval["delta_contrast"] or 0.0

    # Extrai métricas físicas da zona para inclusão no resumo
    gt_vrms = [float(features["sdb_vrm50"][r, c]) for r, c in zip(gt_rows, gt_cols, strict=True) if np.isfinite(features["sdb_vrm50"][r, c])]
    med_vrm = float(np.median(gt_vrms)) if gt_vrms else 0.0
    gt_bpis = [float(features["sdb_bpi100"][r, c]) for r, c in zip(gt_rows, gt_cols, strict=True) if np.isfinite(features["sdb_bpi100"][r, c])]
    med_bpi = float(np.median(gt_bpis)) if gt_bpis else 0.0
    gt_depths = [float(depth_m[r, c]) for r, c in zip(gt_rows, gt_cols, strict=True) if np.isfinite(depth_m[r, c])]
    mean_depth = float(np.mean(gt_depths)) if gt_depths else 0.0

    total_sea_px = int(np.sum(analysis_mask))
    sea_area_km2 = (total_sea_px * 100.0) / 1e6
    cand_area_ha = float(gdf_candidates["area_m2"].sum() / 10000.0) if not gdf_candidates.empty else 0.0
    cand_pct = (cand_area_ha * 10000.0 / (total_sea_px * 100.0)) * 100.0 if total_sea_px > 0 else 0.0

    if no_jev:
        logger.info("TypeSafe Jev desativado pelo utilizador (--no-jev).")
        jev_result = {
            "decision": "SKIPPED",
            "p_reef": None,
            "confidence": None,
            "latency_ms": None,
            "reason": "Desativado por opção do utilizador (--no-jev)",
        }
    elif auc_exact < min_auc_jev:
        logger.warning(
            f"AUC exato ({auc_exact:.3f}) inferior ao limiar mínimo ({min_auc_jev}). "
            "TypeSafe Jev omitido para evitar falsos positivos cognitivos."
        )
        jev_result = {
            "decision": "INCONCLUSIVE_AUC_LOW",
            "p_reef": None,
            "confidence": None,
            "latency_ms": None,
            "reason": f"Detetor sem separabilidade estatística suficiente (AUC={auc_exact:.3f} < {min_auc_jev})",
        }
    else:
        logger.info("Executando validação empírica sub-segundo com TypeSafe Jev...")
        state_summary = (
            f"Relatório de Deteção Instrumental de Recifes Naturais — Zona: {cfg['name']}.\n"
            f"Métricas Quantitativas Medidas:\n"
            f"• Discriminação ROC-AUC (GT exato vs Fundo arenoso): {auc_exact:.3f} (Mann-Whitney p={p_val:.2e}).\n"
            f"• Contraste Mediano: GT={bg_eval['gt_median']:.3f} vs Fundo={bg_eval['bg_median']:.3f} (delta={delta_contrast:+.3f}).\n"
            f"• Taxa de Deteção no Ground Truth a FPR 5%: {bg_eval['recall_fpr5']*100:.1f}% ({int(round(bg_eval['recall_fpr5']*len(gdf_gt)))}/{len(gdf_gt)} cumes).\n"
            f"• Profundidade Média dos Cumes: {mean_depth:.1f} m (faixa batimétrica EMODnet: {cfg['target_depth_str']}).\n"
            f"• Rugosidade 3D (VRM mediano no GT): {med_vrm:.6f}.\n"
            f"• Relevo de Cume (BPI mediano no GT): {med_bpi:+.3f} m.\n"
            f"• Aglomerados Vetorizados: {len(gdf_candidates)} polígonos cobrindo {cand_area_ha:.1f} ha ({cand_pct:.2f}% da área marinha analisada).\n"
        )
        q_text = (
            f"Com base nas medições instrumentais (ROC-AUC={auc_exact:.3f}, separação={delta_contrast:+.3f}, VRM={med_vrm:.5f}), "
            f"os dados confirmam a presença de recife marinho natural com suporte estatístico?"
        )
        jev_result = verify_with_typesafe(state_summary, question_text=q_text)

    # 8. Relatório Final de Resultados
    print("\n" + "=" * 80)
    print(f" RELATÓRIO FINAL: {cfg['name'].upper()}")
    print("=" * 80)
    print(f" Imagem Sentinel-2 Processada  : {scene.shape[0]}x{scene.shape[1]} px (10 m / px, UTM 29N)")
    print(f" Área Marinha Analisada        : {sea_area_km2:.1f} km²")
    print(f" Modo do Detetor               : {mode.upper()}")
    print(f" Aglomerados de Recife Isolados: {len(gdf_candidates)} polígonos ({cand_area_ha:.1f} ha)")
    print(f" Ground Truth Conhecido        : {len(gdf_gt)} pontos de referência")
    print("-" * 80)
    print(" VALIDAÇÃO ESTATÍSTICA CONTRA PONTOS ALEATÓRIOS DO FUNDO (>150 m):")
    print(f" ROC-AUC EXATO (No-Inflation)  : {auc_exact:.3f} (0.5 = acaso, 1.0 = separação perfeita)")
    print(f" Significância Mann-Whitney U  : p = {p_val:.2e}")
    print(f" Probabilidade Mediana no GT   : {bg_eval['gt_median']*100:.1f}%")
    print(f" Probabilidade Mediana no Fundo: {bg_eval['bg_median']*100:.1f}%")
    print(f" Contraste Líquido (Delta)     : {delta_contrast*100:+.1f}%")
    print(f" Recall a FPR 5% (p >= {fpr5_cut:.3f}): {bg_eval['recall_fpr5']*100:.1f}% ({int(round(bg_eval['recall_fpr5']*len(gdf_gt)))}/{len(gdf_gt)} cumes)")
    print(f" Recall a FPR 10%              : {bg_eval['recall_fpr10']*100:.1f}%")
    print(f" [Registo Histórico] AUC 3x3 Max: {bg_eval['auc_3x3_max']:.3f} (inflação por máximo local)")
    print("-" * 80)
    print(" VALIDAÇÃO COGNITIVA (TYPESAFE JEV):")
    if jev_result["decision"] in JEV_DECISION_LABELS:
        print(f" Decisão do Jev                : {JEV_DECISION_LABELS[jev_result['decision']]}")
    else:
        print(f" Decisão do Jev                : {jev_result['decision']}")
    if jev_result.get("p_reef") is not None:
        print(f" P(recife natural)             : {jev_result['p_reef'] * 100:.1f}%")
        print(f" Confiança |2p−1|              : {jev_result['confidence'] * 100:.1f}%")
        print(f" Latência de Decisão do Jev    : {jev_result['latency_ms']} ms")
    elif "reason" in jev_result:
        print(f" Estado / Motivo               : {jev_result['reason']}")
    print("=" * 80)
    print(f" Ficheiro Ráster de Probabilidade: {cfg['out_proba_tif'].relative_to(PROJECT_ROOT)}")
    print(f" Ficheiro Vetorial de Polígonos  : {cfg['out_candidates_geojson'].relative_to(PROJECT_ROOT)}")
    print(f" Relatório CSV Detalhado         : {cfg['out_report_csv'].relative_to(PROJECT_ROOT)}")
    print("=" * 80 + "\n")

    return {
        "zone": zone_key,
        "auc_exact": auc_exact,
        "p_val": p_val,
        "delta": delta_contrast,
        "bg_eval": bg_eval,
        "jev_result": jev_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline Rigoroso de Deteção de Recifes Naturais do Algarve")
    parser.add_argument(
        "--zone",
        choices=["valado", "greta", "santa_eulalia", "quarteira", "all"],
        default="all",
        help="Zona a processar: 'valado', 'greta', 'santa_eulalia', 'quarteira' ou 'all' (todas).",
    )
    parser.add_argument(
        "--mode",
        choices=["supervised", "absolute"],
        default="supervised",
        help="Modo do detetor: 'supervised' (Random Forest treinado no Ground Truth) ou 'absolute' (limiares físicos).",
    )
    parser.add_argument(
        "--no-jev",
        action="store_true",
        help="Desativa chamadas ao TypeSafe Jev.",
    )
    parser.add_argument(
        "--min-auc-jev",
        type=float,
        default=0.65,
        help="ROC-AUC exato mínimo necessário para acionar validação no TypeSafe Jev (padrão: 0.65).",
    )
    args = parser.parse_args()

    classifier = None
    if args.mode == "supervised":
        logger.info("Inicializando treino do ReefClassifier supervisionado multizona...")
        classifier, train_metrics = train_multi_zone_classifier(ZONE_CONFIGS)
        logger.info(
            f"ReefClassifier pronto: {train_metrics['n_gt']} GT cumes + {train_metrics['n_bg']} fundo | "
            f"5-Fold CV AUC: {train_metrics['cv_auc_oof']:.3f}"
        )

    zones_to_run = list(ZONE_CONFIGS.keys()) if args.zone == "all" else [args.zone]

    for z_key in zones_to_run:
        run_zone_pipeline(
            z_key,
            ZONE_CONFIGS[z_key],
            classifier=classifier,
            mode=args.mode,
            no_jev=args.no_jev,
            min_auc_jev=args.min_auc_jev,
        )


if __name__ == "__main__":
    main()
