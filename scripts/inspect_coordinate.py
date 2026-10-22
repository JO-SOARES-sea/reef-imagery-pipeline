#!/usr/bin/env python3
"""
scripts/inspect_coordinate.py — Varredura e Inspeção Universal de Recifes por Coordenada GPS.

Recebe qualquer coordenada (em graus e minutos náuticos DMM ou decimal),
regista a marca no catálogo oficial da embarcação, recorta o Sentinel-2 e batimetria EMODnet,
aplica o pipeline morfo-espectral completo, calcula distâncias e rumos às cristas rochosas,
valida com TypeSafe Jev e gera o pacote náutico completo (PDF 5 págs, GPX, CSV, GeoTIFF, GeoJSON).

Exemplo:
  python scripts/inspect_coordinate.py --name "Lages" --coord "n36,57,898 007,57,579w" --source "sonda" --generate-pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.coordinate_scanner import scan_coordinate_for_reefs
from src.navigation.sonar_marks import SonarMarkCatalog, parse_coordinates, to_dmm
from src.reports.nautical_pdf_generator import generate_nautical_report_package

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("inspect_coordinate")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Varredura e Inspeção Universal de Recifes a partir de Coordenada de Sonda / GPS"
    )
    parser.add_argument("--name", type=str, required=True, help="Nome do alvo ou marca (ex: Lages, Pedra da Greta)")
    parser.add_argument(
        "--coord",
        type=str,
        default=None,
        help="Coordenada em formato DMM ou decimal (ex: 'n36,57,898 007,57,579w' ou '36.964967, -7.959650')",
    )
    parser.add_argument("--lat", type=float, default=None, help="Latitude decimal (alternativa a --coord)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude decimal (alternativa a --coord)")
    parser.add_argument("--buffer", type=float, default=2000.0, help="Raio da janela de inspeção em metros (padrão: 2000 m)")
    parser.add_argument("--source", type=str, default="sonda_barco", help="Origem da marca (sonda_barco, mergulho, historico)")
    parser.add_argument("--generate-pdf", action="store_true", default=True, help="Gerar relatório náutico em PDF (5 páginas)")
    parser.add_argument("--no-pdf", action="store_false", dest="generate_pdf", help="Desativar geração de PDF")
    parser.add_argument("--no-jev", action="store_true", help="Desativar validação com TypeSafe Jev")
    parser.add_argument("--min-auc-jev", type=float, default=0.65, help="Limiar de AUC para acionar o Jev (padrão: 0.65)")

    args = parser.parse_args()

    # 1. Parse de coordenadas
    if args.coord:
        lat, lon = parse_coordinates(args.coord)
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        logger.error("É obrigatório fornecer --coord ou o par (--lat, --lon).")
        return 1

    dmm_str = to_dmm(lat, lon)
    logger.info("================================================================================")
    logger.info(f" INSPECIONANDO ALVO SUBMARINO: {args.name.upper()}")
    logger.info("================================================================================")
    logger.info(f" Coordenada GPS Decimal : {lat:.6f}° N, {lon:.6f}° W")
    logger.info(f" Formato Plotter Náutico: {dmm_str}")
    logger.info(f" Raio de Varredura      : {args.buffer:.0f} m ({args.buffer*2/1000:.1f}x{args.buffer*2/1000:.1f} km)")
    logger.info(f" Tipo de Origem         : {args.source}")
    logger.info("--------------------------------------------------------------------------------")

    # 2. Executa a varredura
    results = scan_coordinate_for_reefs(
        lat=lat,
        lon=lon,
        name=args.name,
        buffer_m=args.buffer,
        use_jev=not args.no_jev,
        min_auc_jev=args.min_auc_jev,
    )

    # 3. Guarda a marca no catálogo oficial
    catalog = SonarMarkCatalog()
    sonda_depth = results["sonda_mark"]["depth_emodnet_m"]
    catalog.add_mark(
        name=args.name,
        lat=lat,
        lon=lon,
        depth_m=sonda_depth,
        source_type=args.source,
        notes=f"Inspecionado via satélite: {results['candidate_count']} cristas detetadas",
    )
    logger.info(f"Marca '{args.name}' guardada com sucesso no Catálogo Náutico ({catalog.csv_path}) e GPX.")

    # 4. Gera o pacote náutico (PDF, PNG, GPX, CSV)
    report_files = None
    if args.generate_pdf:
        logger.info("Gerando pacote náutico completo (PDF 5 páginas, mapa PNG, GPX e CSV)...")
        report_files = generate_nautical_report_package(results)

    # 5. Apresentação do Resumo Executivo no Terminal
    eval_stat = results["validation_eval"]
    jev = results.get("jev_validation") or {}
    ridges = results["detected_ridges"]

    print("\n" + "=" * 80)
    print(f" RELATÓRIO DE INSPEÇÃO SUBMARINA: {args.name.upper()}")
    print("=" * 80)
    print(f" Marca de Sonda Fornecida     : {dmm_str}")
    print(f" Profundidade EMODnet (ZH)    : {sonda_depth:.1f} m" if sonda_depth is not None else " Profundidade EMODnet (ZH)    : N/A")
    print(f" Probabilidade de Rocha na Sonda: {results['sonda_mark']['proba_sonda']*100:.1f}%")
    print(f" Área Marinha Inspecionada    : {results['total_marine_km2']:.1f} km²")
    print(f" Cristas / Aglomerados Isolados: {results['candidate_count']} polígonos ({results['total_reef_ha']:.1f} ha)")
    print("-" * 80)
    print(" VALIDAÇÃO ESTATÍSTICA CONTRA O FUNDO ARENOSO (>150 m):")
    print(f" ROC-AUC Exato (No-Inflation) : {eval_stat.get('auc_exact', 0.0):.3f} (0.5 = acaso, 1.0 = separação perfeita)")
    print(f" Significância Mann-Whitney U : p = {eval_stat.get('p_value', 1.0):.2e}")
    print(f" Probabilidade Mediana Cumes  : {eval_stat.get('gt_median', 0.0)*100:.1f}%")
    print(f" Probabilidade Mediana Fundo  : {eval_stat.get('bg_median', 0.0)*100:.1f}%")
    print(f" Contraste Líquido (Delta)    : +{eval_stat.get('delta_contrast', 0.0)*100:.1f}%")
    print("-" * 80)

    if jev and jev.get("executed"):
        print(" VALIDAÇÃO COGNITIVA (TYPESAFE JEV):")
        print(f" Decisão do Jev               : {jev.get('decision_text')}")
        print(f" P(recife natural)            : {jev.get('p_reef', 0.0)*100:.1f}%")
        print(f" Grau de Confiança |2p-1|     : {jev.get('confidence', 0.0)*100:.1f}%")
        print(f" Latência de Resposta         : {jev.get('latency_ms', 0.0):.1f} ms")
        print("-" * 80)

    if ridges:
        print(" CRISTAS ROCHOSAS MAIS PRÓXIMAS DA MARCA DE SONDA:")
        print(f" {'ID':<10} {'Plotter (DMM)':<26} {'Prof(m)':<8} {'Dist(m)':<8} {'Rumo':<6} {'P(recife)':<8}")
        print(" " + "-" * 72)
        for r in ridges[:6]:
            prof_s = f"{r['prof_emodnet_m']:.1f}m" if r['prof_emodnet_m'] is not None else "—"
            print(f" {r['id']:<10} {r['plotter_dmm']:<26} {prof_s:<8} {r['dist_sonda_m']:<8.0f} {r['rumo_deg']:<6.0f}° {r['prob_max']*100:<7.0f}%")
        print("-" * 80)

    if report_files:
        print(" FICHEIROS NÁUTICOS GERADOS:")
        print(f" • Relatório PDF (5 páginas) : {report_files['pdf']}")
        print(f" • Ficheiro GPX para Plotter : {report_files['gpx']}")
        print(f" • Tabela de Coordenadas CSV : {report_files['csv']}")
        print(f" • Mapa de Alta Resolução    : {report_files['png_map']}")
        print(f" • Ráster GeoTIFF            : {results['files']['proba_tif']}")
        print(f" • Polígonos GeoJSON         : {results['files']['candidates_geojson']}")
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
