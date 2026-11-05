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
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv

from src.analysis.features import extract_morpho_spectral_features, resample_emodnet_screen
from src.analysis.gemini_analyzer import JEV_DECISION_LABELS, ask_reef_noul
from src.analysis.reef_segmenter import (
    ReefClassifier,
    compute_reef_probability_map,
    collect_zone_samples,
    evaluate_against_background,
    extract_candidate_reefs,
    evaluate_physical_rule,
    leave_one_zone_out,
    train_multi_zone_classifier,
)
from src.ingestion.sentinel_reader import (
    build_clean_water_mask,
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
        "ground_truth_path": PROJECT_ROOT / "data" / "vector" / "quarteira_cosmo_rocha.csv",  # COSMO LTHQUGA1 (build_cosmo_truth_points.py)
        "ground_truth_type": "cosmo",
        "out_proba_tif": PROJECT_ROOT / "outputs" / "classifications" / "quarteira_reef_proba.tif",
        "out_candidates_geojson": PROJECT_ROOT / "outputs" / "classifications" / "quarteira_candidates.geojson",
        "out_report_csv": PROJECT_ROOT / "data" / "reports" / "quarteira_validation.csv",
        "description": "Faixa de Recifes e Afloramentos de Quarteira a Vilamoura e Garrão",
        "target_depth_str": "6 m a 15 m",
    },
}


def verify_with_typesafe(state_description: str, question_text: str | None = None) -> dict:
    """Valida a zona com o TypeSafe Jev (timeout + retry; falhas dão "não executado", não "não")."""
    q_text = question_text or (
        "Com base nos dados instrumentais e na separação estatística face ao fundo arenoso, "
        "este candidato corresponde a um recife marinho natural?"
    )
    result = ask_reef_noul(state_description, q_text)
    if result["error"]:
        logger.warning(f"Validação Jev não executada: {result['error']}")
    return result


def run_zone_pipeline(
    zone_key: str,
    cfg: dict[str, Any],
    classifier: ReefClassifier | None = None,
    mode: str = "supervised",
    no_jev: bool = False,
    min_auc_jev: float = 0.65,
    lozo: dict[str, Any] | None = None,
    min_gain_over_depth: float = 0.05,
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
    threshold_polygon = 0.45 if mode == "absolute" else 0.50
    gdf_candidates = extract_candidate_reefs(reef_proba, scene.profile, threshold=threshold_polygon, min_area_m2=500.0)

    if not gdf_candidates.empty:
        cfg["out_candidates_geojson"].parent.mkdir(parents=True, exist_ok=True)
        gdf_candidates.to_file(cfg["out_candidates_geojson"], driver="GeoJSON")
        logger.info(f"Candidatos vetorizados gravados em: {cfg['out_candidates_geojson']}")

    # 6. Validação Cruzada com Ground Truth e Fundo Arenoso
    logger.info("Passo 6/6: Validando deteções contra Ground Truth e 2000 pontos aleatórios de fundo...")
    df_gt = pd.read_csv(cfg["ground_truth_path"])
    # Sítios de mergulho (algarve_dive_sites.csv) NÃO entram na verdade: só 1 de 14 cai na área de Teixeira e a
    # >1 km da rocha (ledger E027). Ficam como pistas para a sonda.
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
                    item["tipo"] = "cosmo_lidar" if cfg.get("ground_truth_type") == "cosmo" else "ccmar_multibeam"
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

    # Ganho do detetor sobre um modelo só de profundidade, ambos leave-one-zone-out
    # O gate usa o limite inferior do IC 95%: com poucos cumes um ganho pontual pode ser só ruído
    depth_gain = None
    if lozo and lozo.get("gain_ci95") is not None and np.isfinite(lozo["gain_ci95"][0]):
        depth_gain = float(lozo["gain_ci95"][0])

    if no_jev:
        logger.info("TypeSafe Jev desativado pelo utilizador (--no-jev).")
        jev_result = {
            "decision": "SKIPPED",
            "p_reef": None,
            "confidence": None,
            "latency_ms": None,
            "reason": "Desativado por opção do utilizador (--no-jev)",
        }
    elif mode != "absolute" and depth_gain is None:
        # Falha fechada: sem evidência de ganho sobre a profundidade (sem GT, uma só classe ou uma só zona)
        # o confundimento com a profundidade não foi excluído, logo o Jev não é chamado.
        logger.warning("Sem IC do ganho sobre um modelo só de profundidade: TypeSafe Jev omitido (gate falha fechado).")
        jev_result = {
            "decision": "INCONCLUSIVE_NO_DEPTH_EVIDENCE",
            "p_reef": None,
            "confidence": None,
            "latency_ms": None,
            "reason": "Sem validação leave-one-zone-out utilizável (GT ou zonas insuficientes): confundimento com a profundidade não excluído",
        }
    elif depth_gain is not None and depth_gain < min_gain_over_depth:
        logger.warning(
            f"Limite inferior do IC 95% do ganho sobre um modelo só de profundidade: {depth_gain:+.3f}. "
            "TypeSafe Jev omitido: o detetor reconhece a profundidade típica dos cumes, não o recife."
        )
        jev_result = {
            "decision": "INCONCLUSIVE_DEPTH_CONFOUND",
            "p_reef": None,
            "confidence": None,
            "latency_ms": None,
            "reason": (
                f"AUC do detetor ({lozo['auc']:.3f}) não supera a de um modelo só de profundidade "
                f"({lozo['auc_depth_only']:.3f}) com confiança: IC 95% do ganho "
                f"[{lozo['gain_ci95'][0]:+.3f}, {lozo['gain_ci95'][1]:+.3f}], mínimo exigido {min_gain_over_depth}"
            ),
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
            f"• Taxa de Deteção no Ground Truth a FPR 5%: {bg_eval['recall_fpr5']*100:.1f}% ({int(round(bg_eval['recall_fpr5'] * bg_eval['n_gt']))}/{bg_eval['n_gt']} cumes).\n"
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
    print(f" Recall a FPR 5% (p >= {fpr5_cut:.3f}): {bg_eval['recall_fpr5']*100:.1f}% ({int(round(bg_eval['recall_fpr5'] * bg_eval['n_gt']))}/{bg_eval['n_gt']} cumes)")
    print(f" Recall a FPR 10%              : {bg_eval['recall_fpr10']*100:.1f}%")
    print(f" [Registo Histórico] AUC 3x3 Max: {bg_eval['auc_3x3_max']:.3f} (inflação por máximo local)")
    if lozo and lozo.get("auc") is not None:
        print("-" * 80)
        if mode == "physical":
            print(" VALIDAÇÃO DA REGRA FÍSICA (sem treino; fundo à mesma profundidade ±1 m):")
        else:
            print(" VALIDAÇÃO LEAVE-ONE-ZONE-OUT (modelo nunca viu esta zona):")
        print(f" AUC do Detetor                : {lozo['auc']:.3f} ({lozo['n_gt']} GT vs {lozo['n_bg']} fundo)")
        print(f" AUC Só Profundidade (EMODnet) : {lozo['auc_depth_only']:.3f}")
        lo, hi = lozo["gain_ci95"]
        print(f" Ganho sobre a Profundidade    : {lozo['gain']:+.3f} (IC 95%: [{lo:+.3f}, {hi:+.3f}])")
        print(f" Gate do Jev (IC inferior >= {min_gain_over_depth}): {'PASSA' if lo >= min_gain_over_depth else 'FALHA'}")
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
        print(f" Modelo que Respondeu          : {jev_result.get('model')}")
    elif "reason" in jev_result:
        print(f" Estado / Motivo               : {jev_result['reason']}")
    elif jev_result.get("error"):
        print(f" Estado / Motivo               : {jev_result['error']}")
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
        "lozo": lozo,
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
        choices=["physical", "supervised", "absolute"],
        default="physical",
        help=(
            "Modo do detetor: 'physical' (anomalia de escurecimento à mesma profundidade, sem treino — padrão), "
            "'supervised' (Random Forest, avaliado leave-one-zone-out) ou 'absolute' (limiares fixos legados)."
        ),
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
    parser.add_argument(
        "--imagery",
        choices=["auto", "sample", "composite"],
        default="auto",
        help=(
            "Imagem de entrada: 'composite' (mediana de dias com água límpida, gerada por "
            "scripts/build_clear_water_composite.py), 'sample' (cena única) ou 'auto' (composição quando existe)."
        ),
    )
    args = parser.parse_args()

    zones_to_run = list(ZONE_CONFIGS.keys()) if args.zone == "all" else [args.zone]

    if args.imagery != "sample":
        for z_key in zones_to_run:
            cfg = ZONE_CONFIGS[z_key]
            scene = cfg["scene_path"]
            if not scene.name.endswith("_sample.tif"):
                parser.error(f"{z_key}: nome de cena inesperado ({scene.name}); esperado '*_sample.tif'")
            comp = scene.with_name(scene.name[: -len("_sample.tif")] + "_composite.tif")
            if not comp.exists():
                if args.imagery == "composite":
                    parser.error(f"Composição em falta para {z_key}: {comp} — corra scripts/build_clear_water_composite.py")
                logger.warning(
                    f"{z_key}: sem composição de dias límpidos; a usar a cena única (sujeita a esteiras e glint). "
                    "Gere-a com scripts/build_clear_water_composite.py"
                )
                continue
            cfg["scene_path"] = comp
            for key in ("out_proba_tif", "out_candidates_geojson", "out_report_csv"):
                cfg[key] = cfg[key].with_name(cfg[key].stem + "_composite" + cfg[key].suffix)

    samples, lozo_results = None, {}
    if args.mode == "supervised":
        logger.info("Recolhendo amostras multizona e validando leave-one-zone-out...")
        samples = collect_zone_samples(ZONE_CONFIGS)
        lozo_results = leave_one_zone_out(samples)
    elif args.mode == "physical":
        logger.info("Avaliando a regra física contra fundo à mesma profundidade (sem treino)...")
        zone_samples = collect_zone_samples({z: ZONE_CONFIGS[z] for z in zones_to_run})
        lozo_results = {z: evaluate_physical_rule(zone_samples[z]) for z in zones_to_run}

    for z_key in zones_to_run:
        classifier = None
        if samples is not None:
            # Modelo treinado SEM esta zona: as métricas da zona ficam out-of-sample
            classifier, train_metrics = train_multi_zone_classifier(ZONE_CONFIGS, exclude_zones=[z_key], samples=samples)
            logger.info(
                f"ReefClassifier para {z_key} (treino: {', '.join(train_metrics['train_zones'])}): "
                f"{train_metrics['n_gt']} GT + {train_metrics['n_bg']} fundo"
            )
        run_zone_pipeline(
            z_key,
            ZONE_CONFIGS[z_key],
            lozo=lozo_results.get(z_key),
            classifier=classifier,
            mode=args.mode,
            no_jev=args.no_jev,
            min_auc_jev=args.min_auc_jev,
        )


if __name__ == "__main__":
    main()
