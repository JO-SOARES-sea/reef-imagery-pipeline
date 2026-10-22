"""
Gerador de Relatórios Náuticos em PDF (5 Páginas), GPX e CSV para Plotters de Barco.
Gera documentos idênticos ao padrão oceanográfico de alta precisão (ex: Pedra da Greta):
  - Página 1: Resumo Executivo, Anomalias Físicas e Coordenadas Principais
  - Página 2: Mapa Duplo (Cor Natural do Satélite vs Probabilidade de Recife com Cristas)
  - Página 3: Perfil Batimétrico e Morfométrico ao Longo da Crista
  - Página 4: Lista Completa de Coordenadas Formatadas para Plotter (DMM) e Decimal
  - Página 5: Método Estatístico, Parecer Cognitivo do TypeSafe Jev e Recomendações de Bordo
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from rasterio.warp import transform

from src.navigation.sonar_marks import to_dmm


def generate_nautical_report_package(
    scan_results: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Gera o pacote náutico completo:
      - Relatório PDF de 5 páginas estruturadas
      - Imagem do mapa de satélite (.png)
      - Ficheiro de waypoints para plotter (.gpx)
      - Ficheiro tabular (.csv)
    """
    name = scan_results["name"]
    clean_name = name.lower().replace(" ", "_")

    if output_dir is None:
        output_dir = Path(scan_results["files"]["proba_tif"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{clean_name}_relatorio_nautico.pdf"
    png_map_path = output_dir / f"{clean_name}_mapa_inspecao.png"
    csv_coords_path = output_dir / f"{clean_name}_coordenadas.csv"
    gpx_coords_path = output_dir / f"{clean_name}_coordenadas.gpx"

    # 1. Exporta CSV e GPX das coordenadas detetadas
    _export_coords_csv_and_gpx(scan_results, csv_coords_path, gpx_coords_path)

    # 2. Constrói o PDF de 5 páginas com PdfPages
    with PdfPages(pdf_path) as pdf:
        _render_page_1_summary(pdf, scan_results)
        _render_page_2_maps(pdf, scan_results, png_map_path)
        _render_page_3_profile(pdf, scan_results)
        _render_page_4_coordinates_table(pdf, scan_results)
        _render_page_5_methodology_and_jev(pdf, scan_results)

    return {
        "pdf": pdf_path,
        "png_map": png_map_path,
        "csv": csv_coords_path,
        "gpx": gpx_coords_path,
    }


def _render_page_1_summary(pdf: PdfPages, data: dict[str, Any]) -> None:
    """Página 1: Resumo e Coordenadas Principais."""
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150)  # A4 Landscape
    name = data["name"]
    sonda = data["sonda_mark"]
    eval_stat = data["validation_eval"]
    jev = data.get("jev_validation") or {}
    ridges = data["detected_ridges"]

    # Cabeçalho
    plt.text(0.05, 0.93, f"{name} — Deteção de Recife por Satélite e Validação de Sonda", fontsize=17, fontweight="bold")
    plt.text(
        0.05,
        0.88,
        f"ReefImageryPipeline · {datetime.now().strftime('%d/%m/%Y')} · Sentinel-2 L2A (10m) · Batimetria EMODnet (ZH)",
        fontsize=9,
        color="#555555",
    )

    # Pontos chave em texto
    bullets = [
        f"• Marca de Sonda Fornecida: {sonda['plotter_dmm']} (Profundidade EMODnet: {sonda['depth_emodnet_m']:.1f} m ZH).",
        f"• Área Marinha Inspecionada: {data['total_marine_km2']:.1f} km² num raio de {data['buffer_m']:.0f} m em redor da marca.",
        f"• Aglomerados de Recife Isolados: {data['candidate_count']} formações rochosas (totalizando {data['total_reef_ha']:.1f} ha).",
    ]

    if ridges:
        closest = min(ridges, key=lambda x: x["dist_sonda_m"])
        bullets.append(
            f"• Crista Rochosa Mais Próxima da Sonda: {closest['id']} a {closest['dist_sonda_m']:.0f} m "
            f"a Rumo {closest['rumo_deg']:.0f}° ({closest['plotter_dmm']})."
        )
    else:
        bullets.append("• Nenhuma crista com probabilidade >= 50% detetada no raio imediato de inspeção.")

    auc_str = f"{eval_stat['auc_exact']:.3f}" if eval_stat.get("auc_exact") is not None else "N/A"
    p_str = f"{eval_stat['p_value']:.2e}" if eval_stat.get("p_value") is not None else "N/A"
    delta_str = f"+{eval_stat['delta_contrast']*100:.1f}%" if eval_stat.get("delta_contrast") is not None else "N/A"
    bullets.append(
        f"• Validação Estatística contra Fundo (>150m): ROC-AUC Exato = {auc_str} (p = {p_str}), "
        f"Contraste Líquido = {delta_str}."
    )

    if jev and jev.get("executed"):
        bullets.append(
            f"• Parecer Cognitivo TypeSafe Jev: {jev.get('decision_text')} "
            f"(P(recife): {jev.get('p_reef', 0)*100:.1f}%, Confiança: {jev.get('confidence', 0)*100:.1f}%, Latência: {jev.get('latency_ms', 0):.1f} ms)."
        )

    y_pos = 0.81
    for b in bullets:
        plt.text(0.05, y_pos, b, fontsize=10.5, wrap=True)
        y_pos -= 0.055

    # Tabela com as Coordenadas Mais Importantes
    plt.text(0.05, 0.45, "Coordenadas Mais Importantes (Lista completa na página 4 e em ficheiros .csv / .gpx):", fontsize=11, fontweight="bold")

    table_data = [
        ["Ponto / Alvo", "Origem", "Plotter (Graus e Minutos)", "Lat (°)", "Lon (°)", "Prof. (m)", "Distância", "Rumo"],
        [
            name.upper()[:10],
            "Sonda Barco",
            sonda["plotter_dmm"],
            f"{sonda['lat']:.6f}",
            f"{sonda['lon']:.6f}",
            f"{sonda['depth_emodnet_m']:.1f}",
            "0 m (Marca)",
            "—",
        ],
    ]

    for r in ridges[:5]:
        table_data.append([
            r["id"],
            "Satélite S2",
            r["plotter_dmm"],
            f"{r['lat']:.6f}",
            f"{r['lon']:.6f}",
            f"{r['prof_emodnet_m']:.1f}" if r["prof_emodnet_m"] is not None else "—",
            f"{r['dist_sonda_m']:.0f} m",
            f"{r['rumo_deg']:.0f}°",
        ])

    table = plt.table(
        cellText=table_data,
        cellLoc="center",
        loc="bottom",
        bbox=[0.05, 0.08, 0.90, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _render_page_2_maps(pdf: PdfPages, data: dict[str, Any], png_save_path: Path) -> None:
    """Página 2: Mapa Duplo (Cor Natural vs Probabilidade de Recife)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=150)
    name = data["name"]
    sonda = data["sonda_mark"]
    b02 = data["arrays"]["b02"]
    b03 = data["arrays"]["b03"]
    b04 = data["arrays"]["b04"]
    reef_proba = data["arrays"]["reef_proba"]
    profile = data["profile"]
    ridges = data["detected_ridges"]

    # Pixel da sonda
    xs, ys = transform("EPSG:4326", profile["crs"], [sonda["lon"]], [sonda["lat"]])
    px_s, py_s = ~profile["transform"] * (xs[0], ys[0])

    # 1. Composição de Cor Natural (RGB = B04, B03, B02)
    rgb = np.stack([b04, b03, b02], axis=-1)
    rgb_valid = rgb[np.isfinite(rgb) & (rgb > 0)]
    if len(rgb_valid) > 0:
        p2, p98 = np.percentile(rgb_valid, (2, 98))
        rgb_disp = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    else:
        rgb_disp = np.zeros_like(rgb)

    ax1.imshow(rgb_disp)
    ax1.set_title("Cor Natural Sentinel-2 (B04, B03, B02)", fontsize=11, fontweight="bold")
    ax1.plot(px_s, py_s, marker="o", markersize=9, color="#FFD700", markeredgecolor="black", markeredgewidth=2)
    ax1.text(
        px_s + 6, py_s - 6, f"SONDA: {name}", color="#FFD700", fontsize=9, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )
    ax1.axis("off")

    # 2. Mapa de Probabilidade com Cristas Detetadas
    im2 = ax2.imshow(reef_proba, cmap="magma", vmin=0.0, vmax=1.0)
    ax2.set_title("Probabilidade de Recife Natural e Cristas Detetadas", fontsize=11, fontweight="bold")
    ax2.plot(px_s, py_s, marker="o", markersize=9, color="#FFD700", markeredgecolor="black", markeredgewidth=2)

    # Marca os cumes das cristas
    for r in ridges[:7]:
        c_xs, c_ys = transform("EPSG:4326", profile["crs"], [r["lon"]], [r["lat"]])
        c_px, c_py = ~profile["transform"] * (c_xs[0], c_ys[0])
        ax2.plot(c_px, c_py, marker="x", markersize=8, color="#00FFFF", markeredgewidth=2)
        ax2.text(
            c_px + 5, c_py - 5, r["id"], color="#00FFFF", fontsize=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")]
        )

    plt.colorbar(im2, ax=ax2, orientation="horizontal", fraction=0.046, pad=0.04, label="P(recife natural)")
    ax2.axis("off")

    fig.suptitle(f"{name} — Mapa de Inspeção Submarina (Janela {data['buffer_m']*2/1000:.1f} × {data['buffer_m']*2/1000:.1f} km)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Grava cópia PNG isolada
    fig.savefig(png_save_path, dpi=200, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)


def _render_page_3_profile(pdf: PdfPages, data: dict[str, Any]) -> None:
    """Página 3: Perfil Batimétrico e Morfométrico."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 8.27), dpi=150, sharex=True)
    name = data["name"]
    sonda = data["sonda_mark"]
    depth_m = data["arrays"]["depth_m"]
    reef_proba = data["arrays"]["reef_proba"]
    profile = data["profile"]

    # Extrai transecto radial SE para NW através da marca de sonda
    xs, ys = transform("EPSG:4326", profile["crs"], [sonda["lon"]], [sonda["lat"]])
    px_s, py_s = ~profile["transform"] * (xs[0], ys[0])
    px_s, py_s = int(round(px_s)), int(round(py_s))

    h, w = depth_m.shape
    # Cria linha de perfil diagonal (-100 px a +100 px em redor da sonda)
    idx_range = np.arange(-100, 101)
    pts_x = np.clip(px_s + idx_range, 0, w - 1)
    pts_y = np.clip(py_s - idx_range, 0, h - 1)  # SE para NW

    dists_m = idx_range * 10.0  # 10m por píxel
    prof_vals = depth_m[pts_y, pts_x]
    proba_vals = reef_proba[pts_y, pts_x]

    # Gráfico Superior: Probabilidade de Recife ao longo do Transecto
    ax1.plot(dists_m, proba_vals, color="#E63946", linewidth=2.0, label="P(recife) ao longo do alinhamento")
    ax1.axhline(0.50, color="#555555", linestyle="--", linewidth=1.2, label="Limiar de Deteção (0.50)")
    ax1.axvline(0.0, color="#FFD700", linestyle="-.", linewidth=1.5, label=f"Marca de Sonda ({name})")
    ax1.set_ylabel("Probabilidade de Recife", fontsize=10, fontweight="bold")
    ax1.set_title("Perfil de Probabilidade de Recife ao Longo do Alinhamento Submarino (SE para NW)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Gráfico Inferior: Profundidade EMODnet (ZH)
    ax2.plot(dists_m, prof_vals, color="#1D3557", linewidth=2.0, label="Profundidade EMODnet (m ZH)")
    ax2.axvline(0.0, color="#FFD700", linestyle="-.", linewidth=1.5)
    ax2.set_xlabel("Distância à Marca de Sonda ao Longo da Crista (metros)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Profundidade (metros)", fontsize=10, fontweight="bold")
    ax2.invert_yaxis()  # Mais fundo para baixo
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(f"{name} — Perfis Morfológicos e Batimétricos", fontsize=14, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_page_4_coordinates_table(pdf: PdfPages, data: dict[str, Any]) -> None:
    """Página 4: Tabela Completa de Coordenadas para Plotter."""
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150)
    name = data["name"]
    sonda = data["sonda_mark"]
    ridges = data["detected_ridges"]

    plt.text(0.05, 0.93, f"{name} — Tabela de Coordenadas para Carregamento em Plotter", fontsize=16, fontweight="bold")
    plt.text(
        0.05,
        0.89,
        "Coordenadas formatadas para plotters marítimos Furuno, Garmin, Raymarine e Simrad (Graus e Minutos Decimais DMM).",
        fontsize=9.5,
        color="#555555",
    )

    table_data = [
        ["Identificador", "Origem", "Plotter (Graus e Minutos)", "Latitude", "Longitude", "Prof. (m)", "Dist. Sonda", "Rumo", "Nota Operacional"],
        [
            name.upper()[:10],
            "Sonda Barco",
            sonda["plotter_dmm"],
            f"{sonda['lat']:.6f}",
            f"{sonda['lon']:.6f}",
            f"{sonda['depth_emodnet_m']:.1f}",
            "0 m",
            "—",
            "Marca original de sonda",
        ],
    ]

    for idx, r in enumerate(ridges[:12], start=1):
        table_data.append([
            r["id"],
            "Satélite S2",
            r["plotter_dmm"],
            f"{r['lat']:.6f}",
            f"{r['lon']:.6f}",
            f"{r['prof_emodnet_m']:.1f}" if r["prof_emodnet_m"] is not None else "—",
            f"{r['dist_sonda_m']:.0f} m",
            f"{r['rumo_deg']:.0f}°",
            f"P={r['prob_max']*100:.0f}%, Área={r['area_m2']:.0f}m²",
        ])

    table = plt.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.12, 0.90, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _render_page_5_methodology_and_jev(pdf: PdfPages, data: dict[str, Any]) -> None:
    """Página 5: Metodologia, Validação Cognitiva Jev e Procedimento de Bordo."""
    fig = plt.figure(figsize=(11.69, 8.27), dpi=150)
    name = data["name"]
    sonda = data["sonda_mark"]
    eval_stat = data["validation_eval"]
    jev = data.get("jev_validation") or {}

    plt.text(0.05, 0.93, f"{name} — Metodologia, Validação Cognitiva e Procedimento no Mar", fontsize=16, fontweight="bold")

    sec1 = [
        "1. Metodologia Científica e Processamento Espectral:",
        "• Sentinel-2 L2A BOA (Bottom of Atmosphere): refletâncias B02 (Azul), B03 (Verde), B04 (Vermelho) e B08 (NIR).",
        "• Eliminação de Superfície: máscara combinada de NDWI (>0.0) e limiar adaptativo de NIR para supressão de esteiras, espuma e sunglint.",
        "• Batimetria Calibrada: EMODnet DTM referido ao Zero Hidrográfico (ZH ≈ LAT). Cálculo de SDB calibrada em metros.",
        "• Anomalia e Morfometria: cálculo de rugosidade 3D em float64 (VRM 50m), elevação de crista BPI (100m) e passa-alto espectral.",
    ]

    sec2 = [
        "2. Validação Estatística contra Fundo Arenoso (>150 m de exclusão):",
        f"• ROC-AUC Exato no Pixel: {eval_stat.get('auc_exact', 0.0):.3f} (separação contra 2000 pontos aleatórios de fundo marinho limpo).",
        f"• Teste de Significância Mann-Whitney U: p = {eval_stat.get('p_value', 1.0):.2e} (H1: cumes têm probabilidade estritamente superior ao fundo).",
        f"• Separação de Contraste Líquido: +{eval_stat.get('delta_contrast', 0.0)*100:.1f}% (mediana dos cumes vs mediana do fundo arenoso).",
    ]

    sec3 = [
        "3. Parecer e Decisão Cognitiva do TypeSafe Jev:",
        f"• Decisão do Agente: {jev.get('decision_text', 'NÃO EXECUTADO')}",
        f"• Probabilidade Atribuída P(recife): {jev.get('p_reef', 0.0)*100:.1f}% | Grau de Confiança: {jev.get('confidence', 0.0)*100:.1f}%",
        f"• Latência do Sistema: {jev.get('latency_ms', 0.0):.1f} ms | Request ID: {jev.get('request_id', 'local')}",
        f"• Síntese: {jev.get('prompt_summary', 'Validação concluída com base nos parâmetros empíricos de relevo e refletância.')}",
    ]

    sec4 = [
        "4. Procedimento Operacional para Confirmação no Mar:",
        f"• Rota de Passagem com Sonda: partir da marca {sonda['plotter_dmm']} navegando aos rumos indicados na tabela.",
        "• Verificação com a Sonda: registar a hora UTC para compensação da tabela de marés do Instituto Hidrográfico.",
        "• Mergulho / ROV: orientar a descida com base no GPX exportado para validação da bio-cobertura calcária/algas.",
    ]

    sections = [sec1, sec2, sec3, sec4]
    y_pos = 0.86
    for s in sections:
        plt.text(0.05, y_pos, s[0], fontsize=11, fontweight="bold", color="#1D3557")
        y_pos -= 0.032
        for line in s[1:]:
            plt.text(0.07, y_pos, line, fontsize=9.2, wrap=True)
            y_pos -= 0.026
        y_pos -= 0.015

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _export_coords_csv_and_gpx(data: dict[str, Any], csv_path: Path, gpx_path: Path) -> None:
    """Exporta ficheiros CSV e GPX das coordenadas."""
    name = data["name"]
    sonda = data["sonda_mark"]
    ridges = data["detected_ridges"]

    records = [
        {
            "id": name.upper()[:10],
            "origem": "sonda_barco",
            "plotter_dmm": sonda["plotter_dmm"],
            "lat": f"{sonda['lat']:.6f}",
            "lon": f"{sonda['lon']:.6f}",
            "prof_emodnet_m": f"{sonda['depth_emodnet_m']:.1f}",
            "dist_sonda_m": "0",
            "rumo_deg": "0",
            "nota": "Marca de sonda fornecida",
        }
    ]

    for r in ridges:
        records.append({
            "id": r["id"],
            "origem": "satelite_s2",
            "plotter_dmm": r["plotter_dmm"],
            "lat": f"{r['lat']:.6f}",
            "lon": f"{r['lon']:.6f}",
            "prof_emodnet_m": f"{r['prof_emodnet_m']:.1f}" if r["prof_emodnet_m"] is not None else "",
            "dist_sonda_m": f"{r['dist_sonda_m']:.0f}",
            "rumo_deg": f"{r['rumo_deg']:.0f}",
            "nota": f"Crista com P={r['prob_max']*100:.0f}%",
        })

    # Grava CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "origem", "plotter_dmm", "lat", "lon", "prof_emodnet_m", "dist_sonda_m", "rumo_deg", "nota"])
        writer.writeheader()
        writer.writerows(records)

    # Grava GPX
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="ReefImageryPipeline" xmlns="http://www.topografix.com/GPX/1/1">',
        f'  <metadata><name>{name} - Deteção de Recifes</name></metadata>',
    ]
    for r in records:
        cmt = f"{r['plotter_dmm']} | Sonda: {r['prof_emodnet_m']}m | {r['nota']}"
        xml_lines.append(f'  <wpt lat="{r["lat"]}" lon="{r["lon"]}">')
        xml_lines.append(f'    <name>{r["id"]}</name>')
        xml_lines.append(f'    <cmt>{cmt}</cmt>')
        xml_lines.append(f'    <sym>Rock</sym>')
        xml_lines.append('  </wpt>')
    xml_lines.append('</gpx>')
    gpx_path.write_text("\n".join(xml_lines), encoding="utf-8")
