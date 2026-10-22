"""Gerador do Relatório Náutico Completo: Complexo Pedra da Greta, Lírios e Lages.

Gera:
  1. Relatório Náutico PDF de 5 páginas (outputs/ e UnderwaterVisibility/reports/2026-10-21_greta_recife/);
  2. Mapa de Alta Resolução Náutico (2 painéis, estilo noite/plotter) com linha fina da Greta (lw=1.2);
  3. Mapa do 'Melhor Dia de Satélite' (3 painéis: Cor Natural, Passa-Alto B03 do dia, Passa-Alto B03 mediana);
  4. Ficheiros GPX e CSV para carregamento direto em plotters GPS (Furuno/Garmin/Raymarine).
"""
import sys, pathlib, csv, math, rasterio, numpy as np
from datetime import date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi

ROOT_UV = pathlib.Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility")
sys.path.insert(0, str(ROOT_UV / "reports" / "2026-09-27_ilha_faro_13m"))
sys.path.insert(0, str(ROOT_UV / "reports" / "2026-10-21_greta_recife"))

from if_analyse import ll2rc, rc2ll, WORK
from zones import Z

BEST_DAY = "2019-01-10_S2B_R037"
halo = [pe.withStroke(linewidth=3.2, foreground="black")]
halo_white = [pe.withStroke(linewidth=3.0, foreground="white")]


def dmm(lat: float, lon: float) -> str:
    def _fmt(val: float, deg_w: int, pos_h: str, neg_h: str) -> str:
        h = pos_h if val >= 0 else neg_h
        abs_v = abs(val)
        d = int(abs_v)
        m = (abs_v - d) * 60.0
        return f"{d:0{deg_w}d}°{m:06.3f}'".replace(".", ",") + f" {h}"
    return f"{_fmt(lat, 2, 'N', 'S')} {_fmt(lon, 3, 'W', 'E')}"


def pt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def dist_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    r_lat1 = math.radians(lat1)
    r_lon1 = math.radians(lon1)
    r_lat2 = math.radians(lat2)
    r_lon2 = math.radians(lon2)
    d_lat = r_lat2 - r_lat1
    d_lon = r_lon2 - r_lon1
    a = math.sin(d_lat / 2)**2 + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist_m = 6371000.0 * c
    y = math.sin(d_lon) * math.cos(r_lat2)
    x = math.cos(r_lat1) * math.sin(r_lat2) - math.sin(r_lat1) * math.cos(r_lat2) * math.cos(d_lon)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return dist_m, bearing


def main():
    q = np.load(WORK / "if_maps.npz")
    med = q["R037_med_B03"]
    sea = q["sea"]
    depth = q["depth"]
    H, W = med.shape

    # 1. Carrega cor natural do melhor dia límpido (10/01/2019)
    with rasterio.open(WORK / "crops" / f"{BEST_DAY}.tif") as ds:
        arr = ds.read().astype(float)
        c_off = int(round((ds.bounds.left - Z["grid"]["x0"]) / 10))

    nat = np.full((H, W, 3), 0.9)
    wtr = arr[5] == 6
    b = arr / 1e4
    rgb = np.dstack([b[2], b[1], b[0]])
    lo, hi = np.percentile(rgb[wtr], 1), np.percentile(rgb[wtr], 99.7)
    nat_rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8
    nat[:, c_off:c_off + rgb.shape[1]] = nat_rgb

    # Carrega passa-alto individual do melhor dia
    prep_p = WORK / "prep" / f"prep_{BEST_DAY}.npz"
    zday = np.where(sea, np.load(prep_p)["z_B03"].astype(float), np.nan) if prep_p.exists() else med

    # 2. Definição precisa de Coordenadas
    # Usando as posições empíricas validadas do catálogo e sonda
    coords = {
        "PEDRA0": (36.989633, -8.003817, "barco", "PEDRA0", "Ponta NW extrema / pedra conhecida"),
        "Greta NW": (36.984876, -7.996629, "satelite", "CRISTNW", "Ponta Noroeste da crista detetada"),
        "Greta Crista Max": (36.975013, -7.981591, "satelite", "CRISTMX", "Ponto mais escuro da crista"),
        "Greta SE": (36.972730, -7.978251, "satelite", "CRISTSE", "Ponta Sudeste (Nascente) da crista"),
        "LIRIOS": (36.964217, -7.976833, "sonda", "LIRIOS", "Marca de sonda Lírios (sul da Greta)"),
        "LAGES": (36.964967, -7.959650, "sonda", "LAGES", "Marca de sonda Lages (este da Greta)"),
        "GRT0": (36.995600, -8.006283, "barco", "GRT0", "Coordenada histórica catálogo DL-043"),
    }

    grt_extras = {
        "GRT1": (36.994633, -8.003150, "barco", "GRT1", "Marca norte Furuno"),
        "GRT2": (36.991217, -8.001650, "barco", "GRT2", "Marca próx. PEDRA0 / crista NW"),
        "GRT3": (36.996950, -8.006117, "barco", "GRT3", "Marca Furuno"),
        "GRT4": (36.995350, -8.004783, "barco", "GRT4", "Marca Furuno"),
        "GRT5": (36.995583, -8.003083, "barco", "GRT5", "Marca Furuno"),
        "GRT6": (36.993633, -8.002267, "barco", "GRT6", "Marca Furuno"),
    }
    coords.update(grt_extras)

    # Tabela estruturada
    rows = []
    for nome, (la, lo, tipo, p_nome, nota) in coords.items():
        r, c = ll2rc(la, lo)
        ri, ci = int(round(r)), int(round(c))
        dep_val = float(depth[ri, ci]) if (0 <= ri < H and 0 <= ci < W) else 0.0
        b03_z = float(med[ri, ci]) if (0 <= ri < H and 0 <= ci < W) else 0.0
        d_se, b_se = dist_bearing(coords["Greta SE"][0], coords["Greta SE"][1], la, lo)

        rows.append({
            "nome": nome,
            "plotter_nome": p_nome[:8],
            "tipo": tipo,
            "lat": la,
            "lon": lo,
            "dmm": dmm(la, lo),
            "prof_m": round(dep_val, 1),
            "b03_z": round(b03_z, 2),
            "dist_se_m": int(round(d_se)),
            "rumo_se_deg": int(round(b_se)),
            "nota": nota
        })

    # Pastas de saída
    out_reef = pathlib.Path("outputs/inspections")
    out_uv_greta = ROOT_UV / "reports" / "2026-10-21_greta_recife"
    out_uv_melhor = ROOT_UV / "reports" / "2026-10-21_melhor_dia_satelite"

    out_reef.mkdir(parents=True, exist_ok=True)
    out_uv_greta.mkdir(parents=True, exist_ok=True)
    out_uv_melhor.mkdir(parents=True, exist_ok=True)

    # 3. Exporta CSV e GPX
    for p_csv in (out_reef / "complexo_greta_lirios_lages.csv", out_uv_greta / "complexo_greta_lirios_lages.csv"):
        with open(p_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    gpx_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="reef-imagery-pipeline & UnderwaterVisibility" xmlns="http://www.topografix.com/GPX/1/1">',
    ]
    for r in rows:
        gpx_lines.append(
            f'  <wpt lat="{r["lat"]:.6f}" lon="{r["lon"]:.6f}">'
            f'<name>{r["plotter_nome"]}</name>'
            f'<desc>{r["nome"]} | {r["dmm"]} | Prof: {r["prof_m"]}m | {r["nota"]}</desc>'
            f'</wpt>'
        )
    gpx_lines.append('</gpx>\n')
    gpx_text = "\n".join(gpx_lines)
    (out_reef / "complexo_greta_lirios_lages.gpx").write_text(gpx_text)
    (out_uv_greta / "complexo_greta_lirios_lages.gpx").write_text(gpx_text)

    # 4. Janela ótima de visualização em píxeis
    # Coordenadas em RC
    r_p0, c_p0 = ll2rc(coords["PEDRA0"][0], coords["PEDRA0"][1])
    r_gnw, c_gnw = ll2rc(coords["Greta NW"][0], coords["Greta NW"][1])
    r_gse, c_gse = ll2rc(coords["Greta SE"][0], coords["Greta SE"][1])
    r_lir, c_lir = ll2rc(coords["LIRIOS"][0], coords["LIRIOS"][1])
    r_lag, c_lag = ll2rc(coords["LAGES"][0], coords["LAGES"][1])

    # Janela enquadrada com segurança dentro dos limites da imagem
    r_top = max(0, int(round(r_p0 - 45)))       # ~340
    r_bot = min(H, int(round(max(r_lir, r_lag) + 50)))  # ~685
    c_left = max(0, int(round(c_p0 - 45)))      # ~355
    c_right = min(W, int(round(c_lag + 55)))    # ~815

    sl = (slice(r_top, r_bot), slice(c_left, c_right))
    H_win = sl[0].stop - sl[0].start
    W_win = sl[1].stop - sl[1].start

    # =========================================================================
    # MAPA 1: Duplo Painel Náutico ( outputs/ e 2026-10-21_greta_recife/ )
    # =========================================================================
    fig_map, axs = plt.subplots(1, 2, figsize=(18, 8.8), dpi=160)
    fig_map.patch.set_facecolor("#0a0f1d")

    fig_map.suptitle(
        "Complexo Submarino: Greta, Lírios e Lages — Carta de Alta Resolução Náutica\n"
        "Sentinel-2 L2A (10 m/pixel) · Isóbatas EMODnet (ZH) · Costa de Faro",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.96
    )

    axs[0].imshow(nat[sl], interpolation="bilinear")
    axs[0].set_title("1. Cor Natural de Alta Visibilidade (2019-01-10)\nÁgua Hiper-Límpida pós-Inverno (NIR B08 ≈ 0,006; fundo visível até >18 m)", fontsize=11, color="#e2e8f0", pad=8)

    axs[1].imshow(np.where(sea, med, np.nan)[sl], cmap="gray", vmin=-2.8, vmax=2.8, interpolation="bilinear")
    axs[1].set_title("2. Passa-Alto B03 Multi-Temporal (Mediana 22 Passagens 2017–2026)\nFundo Escuro Persistente · Alinhamento da Crista e Continuidade", fontsize=11, color="#e2e8f0", pad=8)

    def draw_nautical_map(ax):
        ax.set_facecolor("#0a0f1d")
        
        # 1. Isóbatas EMODnet com espessura fina (lw=1.2)
        cs = ax.contour(
            ndi.gaussian_filter(np.nan_to_num(np.where(sea, depth, np.nan)[sl]), 2.2),
            levels=[5, 10, 15, 20],
            colors=["#60a5fa", "#fbbf24", "#f43f5e", "#38bdf8"],
            linewidths=1.2,
            alpha=0.85
        )
        ax.clabel(cs, fmt="%d m", fontsize=8.5, colors="white")

        # 2. LINHA FINA DA GRETA (lw=1.2, exatamente como solicitado pelo utilizador)
        # a) Início junto a PEDRA0 até Greta NW (linha fina contínua)
        ax.plot(
            [c_p0 - sl[1].start, c_gnw - sl[1].start],
            [r_p0 - sl[0].start, r_gnw - sl[0].start],
            "-",
            color="#22d3ee",
            lw=1.2,
            alpha=0.95,
            path_effects=halo
        )

        # b) Crista Principal da Greta (Greta NW -> Greta SE, 2,1 km): linha fina contínua
        ax.plot(
            [c_gnw - sl[1].start, c_gse - sl[1].start],
            [r_gnw - sl[0].start, r_gse - sl[0].start],
            "-",
            color="#00e5ff",
            lw=1.2,
            alpha=1.0,
            path_effects=halo
        )

        # c) Continuidade até LÍRIOS (Greta SE -> Lírios, 955 m): linha fina tracejada
        ax.plot(
            [c_gse - sl[1].start, c_lir - sl[1].start],
            [r_gse - sl[0].start, r_lir - sl[0].start],
            "--",
            color="#38bdf8",
            lw=1.2,
            alpha=0.95,
            path_effects=halo
        )

        # d) Continuidade até LAGES (Greta SE -> Lages, 1 865 m): linha fina tracejada
        ax.plot(
            [c_gse - sl[1].start, c_lag - sl[1].start],
            [r_gse - sl[0].start, r_lag - sl[0].start],
            "--",
            color="#4ade80",
            lw=1.2,
            alpha=0.95,
            path_effects=halo
        )

        # e) Ligação Lírios <-> Lages (1 529 m, rumo 87°): linha fina pontilhada
        ax.plot(
            [c_lir - sl[1].start, c_lag - sl[1].start],
            [r_lir - sl[0].start, r_lag - sl[0].start],
            ":",
            color="#a3e635",
            lw=1.0,
            alpha=0.75,
            path_effects=halo
        )

        # 3. Marcas históricas do barco (GRT)
        for k in ("GRT1", "GRT2", "GRT3", "GRT4", "GRT5", "GRT6"):
            if k in coords:
                rg, cg = ll2rc(coords[k][0], coords[k][1])
                ax.plot(cg - sl[1].start, rg - sl[0].start, "o", mfc="none", mec="#fbbf24", ms=5, mew=1.5)

        r_g0, c_g0 = ll2rc(coords["GRT0"][0], coords["GRT0"][1])
        ax.plot(c_g0 - sl[1].start, r_g0 - sl[0].start, "o", mfc="none", mec="#fbbf24", ms=7, mew=2.0)
        ax.text(c_g0 - sl[1].start + 8, r_g0 - sl[0].start - 4, "GRT0", color="#fbbf24", fontsize=8.0, fontweight="bold", path_effects=halo)

        # 4. PEDRA0
        ax.plot(c_p0 - sl[1].start, r_p0 - sl[0].start, "o", mfc="none", mec="#f97316", ms=9, mew=2.2)
        ax.text(c_p0 - sl[1].start + 10, r_p0 - sl[0].start - 6, "PEDRA0\n(16,1 m)", color="#f97316", fontsize=8.5, fontweight="bold", path_effects=halo)

        # 5. Greta NW e SE
        ax.plot(c_gnw - sl[1].start, r_gnw - sl[0].start, "x", color="#00e5ff", ms=8, mew=2.0)
        ax.text(c_gnw - sl[1].start - 68, r_gnw - sl[0].start - 8, "Greta NW", color="#00e5ff", fontsize=8.5, fontweight="bold", path_effects=halo)

        ax.plot(c_gse - sl[1].start, r_gse - sl[0].start, "x", color="#00e5ff", ms=9, mew=2.0)
        ax.text(c_gse - sl[1].start - 120, r_gse - sl[0].start - 10, "Greta SE (Nascente)\n15,0 m (ZH)", color="#00e5ff", fontsize=8.5, fontweight="bold", path_effects=halo)

        # 6. LÍRIOS (Marca de Sonda)
        ax.plot(c_lir - sl[1].start, r_lir - sl[0].start, "o", mfc="none", mec="#38bdf8", ms=12, mew=2.4)
        ax.plot(c_lir - sl[1].start, r_lir - sl[0].start, "+", color="#38bdf8", ms=9, mew=2.0)
        # Rótulo Lírios deslocado para baixo e esquerda para não tapar as linhas
        ax.text(c_lir - sl[1].start - 110, r_lir - sl[0].start + 24, "LÍRIOS (Sonda)\n36°57,853' N  007°58,610' W\nProf: 17,6 m (ZH)", color="#38bdf8", fontsize=8.5, fontweight="bold", path_effects=halo)

        # 7. LAGES (Marca de Sonda)
        ax.plot(c_lag - sl[1].start, r_lag - sl[0].start, "o", mfc="none", mec="#4ade80", ms=12, mew=2.4)
        ax.plot(c_lag - sl[1].start, r_lag - sl[0].start, "+", color="#4ade80", ms=9, mew=2.0)
        # Rótulo Lages deslocado para a direita
        ax.text(c_lag - sl[1].start + 14, r_lag - sl[0].start + 4, "LAGES (Sonda)\n36°57,898' N  007°57,579' W\nProf: 13,6 m (ZH)", color="#4ade80", fontsize=8.5, fontweight="bold", path_effects=halo)

        # 8. Badges de Navegação (Rumo e Distância)
        # Greta SE -> Lírios (955 m, rumo 172°)
        mid_c_lir = (c_gse + c_lir) / 2 - sl[1].start
        mid_r_lir = (r_gse + r_lir) / 2 - sl[0].start
        ax.text(
            mid_c_lir + 22,
            mid_r_lir - 4,
            "955 m\n(172°)",
            color="#fef08a",
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#020617", edgecolor="#38bdf8", alpha=0.92, lw=1.1),
            path_effects=halo
        )

        # Greta SE -> Lages (1 865 m, rumo 118°)
        mid_c_lag = (c_gse + c_lag) / 2 - sl[1].start
        mid_r_lag = (r_gse + r_lag) / 2 - sl[0].start
        ax.text(
            mid_c_lag,
            mid_r_lag - 22,
            "1 865 m (1,01 NM) · Rumo 118°",
            color="#fef08a",
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#020617", edgecolor="#4ade80", alpha=0.92, lw=1.1),
            path_effects=halo
        )

        # Lírios -> Lages (1 529 m, rumo 87°)
        mid_c_ll = (c_lir + c_lag) / 2 - sl[1].start
        mid_r_ll = (r_lir + r_lag) / 2 - sl[0].start
        ax.text(
            mid_c_ll,
            mid_r_ll + 20,
            "1 529 m · 87°",
            color="#bef264",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#020617", edgecolor="#a3e635", alpha=0.88, lw=1.0),
            path_effects=halo
        )

        # Escala Gráfica de 1 km
        y_scale = H_win - 18
        ax.plot([20, 120], [y_scale, y_scale], color="white", lw=4.0, path_effects=halo)
        ax.text(70, y_scale - 7, "1 km (1 000 m)", color="white", ha="center", fontsize=9.0, fontweight="bold", path_effects=halo)

        # Seta do Norte
        ax.annotate(
            "N",
            xy=(25, 25),
            xytext=(25, 55),
            arrowprops=dict(facecolor="white", edgecolor="black", width=2.2, headwidth=7),
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color="white",
            path_effects=halo
        )

        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs:
        draw_nautical_map(ax)

    plt.tight_layout(rect=(0.01, 0.04, 0.99, 0.93))

    fig_map.text(
        0.5,
        0.015,
        "Alinhamento Submarino: Linha fina ciano (lw=1.2) assinala a crista da Greta desde PEDRA0 até à Ponta SE e continuidade até LÍRIOS (sul) e LAGES (este). "
        "Isóbatas EMODnet: 5m (azul claro), 10m (âmbar), 15m (rosa), 20m (azul profundo).",
        ha="center",
        fontsize=8.5,
        color="#94a3b8"
    )

    fig_map.savefig(out_reef / "complexo_greta_lirios_lages_mapa.png", dpi=160, facecolor=fig_map.get_facecolor(), edgecolor="none")
    fig_map.savefig(out_uv_greta / "complexo_greta_lirios_lages_mapa.png", dpi=160, facecolor=fig_map.get_facecolor(), edgecolor="none")
    plt.close(fig_map)

    # =========================================================================
    # MAPA 2: Formato 3 Painéis ( 2026-10-21_melhor_dia_satelite )
    # =========================================================================
    fig_3p, axs_3p = plt.subplots(1, 3, figsize=(24, 8.5), facecolor="#fcfcfb", dpi=140)
    fig_3p.suptitle(
        "Complexo Pedra da Greta, Lírios e Lages — O Melhor Dia de Satélite (10/01/2019, S2B R037)\n"
        "Costa de Faro (Ilha de Faro) · Linha fina da crista desde PEDRA0 até Lírios e Lages",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        y=0.96
    )

    panels_3p = [
        (nat[sl], "1. Cor Natural de Alta Visibilidade\n(10/01/2019 · NIR B08 ≈ 0,006)", None),
        (zday[sl], "2. Passa-Alto B03 do Próprio Dia\n(fundo escuro instantâneo)", "gray"),
        (np.where(sea, med, np.nan)[sl], "3. Passa-Alto B03 Mediana Multi-Temporal\n(22 passagens R037 · fundo persistente 2017–2026)", "gray")
    ]

    for ax, (img, title, cm) in zip(axs_3p, panels_3p):
        if cm:
            ax.imshow(img, cmap=cm, vmin=-2.8, vmax=2.8, interpolation="bilinear")
        else:
            ax.imshow(img, interpolation="bilinear")
        
        # Isóbatas finas
        cs = ax.contour(
            ndi.gaussian_filter(np.nan_to_num(np.where(sea, depth, np.nan)[sl]), 2.2),
            levels=[5, 10, 15, 20],
            colors=["#60a5fa", "#fbbf24", "#f43f5e", "#38bdf8"],
            linewidths=1.2,
            alpha=0.85
        )
        ax.clabel(cs, fmt="%d m", fontsize=8)

        # Linha fina da Greta (lw=1.2)
        ax.plot([c_p0 - sl[1].start, c_gnw - sl[1].start], [r_p0 - sl[0].start, r_gnw - sl[0].start], "-", color="#06b6d4", lw=1.2, alpha=0.95, path_effects=halo)
        ax.plot([c_gnw - sl[1].start, c_gse - sl[1].start], [r_gnw - sl[0].start, r_gse - sl[0].start], "-", color="#00e5ff", lw=1.2, alpha=1.0, path_effects=halo)
        ax.plot([c_gse - sl[1].start, c_lir - sl[1].start], [r_gse - sl[0].start, r_lir - sl[0].start], "--", color="#38bdf8", lw=1.2, alpha=0.95, path_effects=halo)
        ax.plot([c_gse - sl[1].start, c_lag - sl[1].start], [r_gse - sl[0].start, r_lag - sl[0].start], "--", color="#22c55e", lw=1.2, alpha=0.95, path_effects=halo)

        # Marcas
        ax.plot(c_p0 - sl[1].start, r_p0 - sl[0].start, "o", mfc="none", mec="#f97316", ms=8, mew=2.0)
        ax.text(c_p0 - sl[1].start + 9, r_p0 - sl[0].start - 4, "PEDRA0", color="#f97316", fontsize=8.5, fontweight="bold", path_effects=halo)

        ax.plot(c_gse - sl[1].start, r_gse - sl[0].start, "x", color="#00e5ff", ms=8, mew=2.0)
        ax.text(c_gse - sl[1].start - 105, r_gse - sl[0].start - 8, "Greta SE", color="#00e5ff", fontsize=8.5, fontweight="bold", path_effects=halo)

        ax.plot(c_lir - sl[1].start, r_lir - sl[0].start, "o", mfc="none", mec="#0284c7", ms=11, mew=2.2)
        ax.text(c_lir - sl[1].start - 105, r_lir - sl[0].start + 20, "LÍRIOS (17,6m)", color="#0284c7", fontsize=8.5, fontweight="bold", path_effects=halo)

        ax.plot(c_lag - sl[1].start, r_lag - sl[0].start, "o", mfc="none", mec="#16a34a", ms=11, mew=2.2)
        ax.text(c_lag - sl[1].start + 12, r_lag - sl[0].start + 4, "LAGES (13,6m)", color="#16a34a", fontsize=8.5, fontweight="bold", path_effects=halo)

        # Escala e Norte
        ax.plot([20, 120], [H_win - 18, H_win - 18], color="white", lw=4.0, path_effects=halo)
        ax.text(70, H_win - 25, "1 km", color="white", ha="center", fontsize=9.0, fontweight="bold", path_effects=halo)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=(0.01, 0.04, 0.99, 0.94))
    fig_3p.text(0.5, 0.015, "Relatório 2026-10-21_melhor_dia_satelite · Linha fina assinala o corredor recifal desde PEDRA0 -> Greta -> Lírios e Lages.", ha="center", fontsize=9, color="#64748b")
    fig_3p.savefig(out_uv_melhor / "greta_lirios_lages_melhor_dia.png", dpi=140, facecolor="#fcfcfb")
    plt.close(fig_3p)

    # =========================================================================
    # RELATÓRIO NÁUTICO PDF (5 PÁGINAS)
    # =========================================================================
    pdf_path_reef = out_reef / "complexo_greta_lirios_lages_relatorio_nautico.pdf"
    pdf_path_uv = out_uv_greta / "complexo_greta_lirios_lages_relatorio_nautico.pdf"

    with PdfPages(pdf_path_reef) as pdf:
        # PÁGINA 1: Sumário Executivo
        fig1 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        fig1.patch.set_facecolor("white")
        fig1.text(0.05, 0.935, "Complexo de Recifes: Pedra da Greta, Lírios e Lages", fontsize=18, fontweight="bold", color="#0f172a")
        fig1.text(0.05, 0.905, f"Relatório Náutico Integrado · {date.today().strftime('%d/%m/%Y')} · Sentinel-2 Multi-Temporal & EMODnet · Marcas de Sonda", fontsize=9.5, color="#64748b")

        sum_texts = [
            "• Continuidade Estrutural da Greta: A Pedra da Greta inicia-se junto à marca histórica PEDRA0 e prolonga-se por 2,94 km no rumo 129°–130°,",
            "  constituindo uma crista rochosa estreita, nítida e saliente sobre a isóbata dos 14–15 m (ZH).",
            f"• Relação com os LÍRIOS: A marca de sonda 'Lírios' (36°57,853' N 007°58,610' W, prof: 17,6 m) situa-se a apenas 955 m a sul (rumo 172°)",
            "  da ponta SE da Greta, constituindo o flanco mais profundo da formação submarina na transição para a vasa/areia profunda.",
            f"• Relação com as LAGES: A marca de sonda 'Lages' (36°57,898' N 007°57,579' W, prof: 13,6 m) situa-se a 1 865 m a ESE (rumo 118°) da Greta",
            "  e a 1 529 m dos Lírios (rumo 87°), no prolongamento batimétrico natural do recife a nascente.",
            "• Deteção Ótica e Critérios Físicos: Crista confirmada pela textura escura multi-temporal persistente em 22 passagens R037 (2017–2026).",
            "  Profundidades referidas ao Zero Hidrográfico (ZH ≈ LAT); em operação náutica de mergulho ou pesca com sonda somar a altura da maré."
        ]
        for i, st in enumerate(sum_texts):
            fig1.text(0.05, 0.84 - i * 0.032, st, fontsize=9.8, color="#1e293b")

        fig1.text(0.05, 0.54, "Coordenadas Náuticas Principais para o Plotter (DMM — Graus e Minutos Decimais)", fontsize=12, fontweight="bold", color="#0f172a")

        key_pts = ["PEDRA0", "Greta NW", "Greta SE", "LIRIOS", "LAGES", "GRT0"]
        table_rows = []
        for kp in key_pts:
            r_item = next(r for r in rows if r["nome"] == kp)
            table_rows.append([
                f"{r_item['nome']} ({r_item['plotter_nome']})",
                r_item["dmm"],
                f"{r_item['lat']:.6f}°",
                f"{r_item['lon']:.6f}°",
                f"{pt(r_item['prof_m'])} m",
                "—" if kp == "Greta SE" else f"{r_item['dist_se_m']} m ({r_item['rumo_se_deg']}°)",
                r_item["nota"]
            ])

        ax_tbl = fig1.add_axes([0.05, 0.08, 0.90, 0.43])
        ax_tbl.axis("off")
        t1 = ax_tbl.table(
            cellText=table_rows,
            colLabels=["Ponto (Plotter)", "Formato Plotter (DMM)", "Latitude", "Longitude", "Prof. (ZH)", "Da Ponta SE", "Descrição / Função"],
            loc="upper left",
            cellLoc="left",
            colWidths=[0.20, 0.22, 0.10, 0.10, 0.09, 0.13, 0.28]
        )
        t1.auto_set_font_size(False)
        t1.set_fontsize(9.5)
        t1.scale(1, 1.85)
        for (i, j), cell in t1.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight="bold", color="#0f172a")
                cell.set_facecolor("#e2e8f0")
            elif i % 2 == 1:
                cell.set_facecolor("#f8fafc")

        pdf.savefig(fig1)
        plt.close(fig1)

        # PÁGINA 2: Mapa Náutico
        fig_map_pdf, axs_pdf = plt.subplots(1, 2, figsize=(11.69, 8.27), dpi=150)
        fig_map_pdf.patch.set_facecolor("#0a0f1d")
        fig_map_pdf.suptitle("Complexo Submarino: Greta, Lírios e Lages — Carta Náutica de Alta Resolução", fontsize=13.5, fontweight="bold", color="white", y=0.96)
        axs_pdf[0].imshow(nat[sl], interpolation="bilinear")
        axs_pdf[0].set_title("Cor Natural (10/01/2019)", fontsize=10, color="#e2e8f0")
        axs_pdf[1].imshow(np.where(sea, med, np.nan)[sl], cmap="gray", vmin=-2.8, vmax=2.8, interpolation="bilinear")
        axs_pdf[1].set_title("Passa-Alto B03 Mediana (2017–2026)", fontsize=10, color="#e2e8f0")
        for ax in axs_pdf:
            draw_nautical_map(ax)
        plt.tight_layout(rect=(0.01, 0.04, 0.99, 0.93))
        pdf.savefig(fig_map_pdf)
        plt.close(fig_map_pdf)

        # PÁGINA 3: Perfis Batimétricos e de Sinal Submarino
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(11.69, 8.27), dpi=150)
        fig3.patch.set_facecolor("white")

        path_pts = [
            ("PEDRA0", coords["PEDRA0"][:2]),
            ("Greta NW", coords["Greta NW"][:2]),
            ("Greta SE", coords["Greta SE"][:2]),
            ("LIRIOS", coords["LIRIOS"][:2]),
        ]

        all_dists, all_depths, all_b03 = [], [], []
        cum_dist = 0.0
        for idx in range(len(path_pts) - 1):
            p1_name, (la1, lo1) = path_pts[idx]
            p2_name, (la2, lo2) = path_pts[idx + 1]
            seg_dist, _ = dist_bearing(la1, lo1, la2, lo2)
            steps = max(5, int(seg_dist / 15.0))
            for s in range(steps):
                frac = s / float(steps)
                curr_lat = la1 + frac * (la2 - la1)
                curr_lon = lo1 + frac * (lo2 - lo1)
                r_c, c_c = ll2rc(curr_lat, curr_lon)
                ri, ci = int(round(r_c)), int(round(c_c))
                d_val = float(depth[ri, ci]) if (0 <= ri < H and 0 <= ci < W) else 15.0
                b_val = float(med[ri, ci]) if (0 <= ri < H and 0 <= ci < W) else 0.0
                all_dists.append((cum_dist + frac * seg_dist) / 1000.0)
                all_depths.append(d_val)
                all_b03.append(b_val)
            cum_dist += seg_dist

        rf, cf = ll2rc(path_pts[-1][1][0], path_pts[-1][1][1])
        all_dists.append(cum_dist / 1000.0)
        all_depths.append(float(depth[int(round(rf)), int(round(cf))]))
        all_b03.append(float(med[int(round(rf)), int(round(cf))]))

        ax3a.plot(all_dists, all_b03, color="#0284c7", lw=1.8, label="Passa-Alto B03 Mediana (z-score)")
        ax3a.axhline(0.0, color="#94a3b8", ls=":", lw=1.0)
        ax3a.axhline(-0.5, color="#f43f5e", ls="--", lw=1.2, label="Limiar Fundo Escuro Anómalo (z <= -0.5)")
        ax3a.set_ylabel("z-score B03 (escuro < 0)", fontsize=10, fontweight="bold")
        ax3a.set_title("Perfil Ótico e Batimétrico: PEDRA0 -> Greta NW -> Greta SE -> LÍRIOS (~3,9 km)", fontsize=12, fontweight="bold", pad=8)
        ax3a.grid(True, linestyle="--", alpha=0.5)
        ax3a.legend(loc="upper right", fontsize=9)

        cum_marker = 0.0
        for idx in range(len(path_pts)):
            name, _ = path_pts[idx]
            ax3a.axvline(cum_marker / 1000.0, color="#64748b", ls="--", lw=1.0)
            ax3a.text(cum_marker / 1000.0 + 0.03, -1.8, name, fontsize=8.5, fontweight="bold", rotation=90)
            if idx < len(path_pts) - 1:
                seg_d, _ = dist_bearing(path_pts[idx][1][0], path_pts[idx][1][1], path_pts[idx + 1][1][0], path_pts[idx + 1][1][1])
                cum_marker += seg_d

        ax3b.plot(all_dists, all_depths, color="#059669", lw=2.2, label="Profundidade EMODnet DTM (ZH)")
        ax3b.fill_between(all_dists, all_depths, 22, color="#d1fae5", alpha=0.6)
        ax3b.invert_yaxis()
        ax3b.set_ylabel("Profundidade (m)", fontsize=10, fontweight="bold")
        ax3b.set_xlabel("Distância ao longo do corredor a partir de PEDRA0 (km)", fontsize=10, fontweight="bold")
        ax3b.grid(True, linestyle="--", alpha=0.5)
        ax3b.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig3)
        plt.close(fig3)

        # PÁGINA 4: Tabela Completa de Waypoints
        fig4 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        fig4.patch.set_facecolor("white")
        fig4.text(0.04, 0.94, "Catálogo Geral de Waypoints e Coordenadas para Plotter Náutico", fontsize=14, fontweight="bold", color="#0f172a")
        fig4.text(0.04, 0.915, "Compatível com Furuno, Garmin, Raymarine, Simrad e Lowrance (ficheiros .gpx e .csv anexos)", fontsize=9.5, color="#64748b")

        full_table_data = []
        for r in rows:
            full_table_data.append([
                r["plotter_nome"],
                r["nome"],
                r["tipo"].capitalize(),
                r["dmm"],
                f"{r['lat']:.6f}°",
                f"{r['lon']:.6f}°",
                f"{pt(r['prof_m'])} m",
                f"{r['dist_se_m']} m",
                f"{r['rumo_se_deg']}°",
                r["nota"]
            ])

        ax4 = fig4.add_axes([0.03, 0.06, 0.94, 0.83])
        ax4.axis("off")
        t4 = ax4.table(
            cellText=full_table_data,
            colLabels=["Plotter", "Nome Completo", "Origem", "Formato Náutico (DMM)", "Lat Dec", "Lon Dec", "Prof.", "Dist Greta SE", "Rumo", "Observações"],
            loc="upper left",
            cellLoc="left",
            colWidths=[0.09, 0.15, 0.08, 0.19, 0.08, 0.08, 0.06, 0.08, 0.06, 0.22]
        )
        t4.auto_set_font_size(False)
        t4.set_fontsize(8.2)
        t4.scale(1, 1.55)
        for (i, j), cell in t4.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight="bold", color="#0f172a")
                cell.set_facecolor("#e2e8f0")
            elif i % 2 == 1:
                cell.set_facecolor("#f8fafc")
        pdf.savefig(fig4)
        plt.close(fig4)

        # PÁGINA 5: Metodologia e Decisão TypeSafe Jev
        fig5 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        fig5.patch.set_facecolor("white")
        fig5.text(0.05, 0.935, "Metodologia Ótico-Espectral e Validação Cognitiva TypeSafe Jev", fontsize=15, fontweight="bold", color="#0f172a")

        meth_texts = [
            "Física da Radiação Eletromagnética Submarina:",
            "• Banda Azul (B02, ~490 nm): Máxima penetração na coluna de água costeira (até 15–20 m em dias calmos pós-inverno).",
            "• Banda Verde (B03, ~560 nm): Ótima discriminação de fundo entre 5 m e 18 m de profundidade. A areia reflete intensamente a luz verde,",
            "  enquanto as lajes rochosas e recifes cobertos de flora biogénica e macroalgas absorvem a luz, criando anomalias escuras persistentes.",
            "• Banda Vermelha (B04, ~665 nm): Rápida absorção pelos primeiros 3–5 m de água. Pixels a >10 m sem sinal no B04 comprovam que a anomalia",
            "  não é espuma, lixo flutuante ou esteira de embarcação à superfície.",
            "• Banda NIR (B08, ~842 nm): Absorção total pelos primeiros 10 cm de água limpa. Serve de filtro puro de sunglint e esteiras (B08 < 0,035).",
            "",
            "Mediana Multi-temporal (2017–2026):",
            "• Uma única cena pode ter artefactos temporários. Ao calcular a mediana em 22 passagens límpidas da mesma órbita (R037), sedimentos móveis,",
            "  marés vivas e esteiras são eliminados, restando unicamente as estruturas geológicas estáticas e permanentes do fundo marinho.",
            "",
            "Validação Empírica e Decisão Cognitiva TypeSafe Jev:",
            "• LAGES (36°57,898' N 007°57,579' W, 13,6 m): ROC-AUC exato = 0.962 (+39,9% contraste líquido contra o fundo arenoso).",
            "  Decisão Jev: RECIFE CONFIRMADO (P = 85.0%, Confiança = 70.0%).",
            "• LÍRIOS (36°57,853' N 007°58,610' W, 17,6 m): ROC-AUC exato = 0.686 (+10,4% contraste líquido contra o fundo circundante).",
            "  A profundidade maior (17,6 m) aproxima-se do limite de penetração ótica do Sentinel-2, exigindo confirmação in-situ na sonda.",
            "  Decisão Jev: INCERTO / REVISÃO MANUAL (P = 25.0%). Posição a 955 m a sul da Greta confirma continuação morfológica para águas mais fundas."
        ]

        for i, mt in enumerate(meth_texts):
            is_header = mt.endswith(":")
            fig5.text(0.05, 0.87 - i * 0.033, mt, fontsize=10.0 if not is_header else 11.0, fontweight="bold" if is_header else "normal", color="#0f172a" if is_header else "#334155")

        pdf.savefig(fig5)
        plt.close(fig5)

    import shutil
    shutil.copy2(pdf_path_reef, pdf_path_uv)

    print("Relatório náutico e mapas atualizados com sucesso:")
    print(" - PDF:", pdf_path_reef)
    print(" - Mapa Náutico PNG:", out_reef / "complexo_greta_lirios_lages_mapa.png")
    print(" - Mapa 3 Painéis PNG:", out_uv_melhor / "greta_lirios_lages_melhor_dia.png")
    print(" - GPX:", out_reef / "complexo_greta_lirios_lages.gpx")
    print(" - CSV:", out_reef / "complexo_greta_lirios_lages.csv")


if __name__ == "__main__":
    main()
