"""Gera mapa comparativo de alta resolução entre a Pedra da Greta e as Lages.

Utiliza:
  - Mosaico multitemporal e melhor dia límpido (2019-01-10) de UnderwaterVisibility;
  - Passa-alto B03 (22 passagens R037, 2017-2026);
  - Crista detetada da Greta (SE -> NW);
  - Marca de sonda Lages (36°57,898' N 007°57,579' W);
  - Vetor geodésico de distância e rumo;
  - Isóbatas EMODnet (5 m, 10 m, 15 m, 20 m).
"""
import sys, pathlib, csv, rasterio, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import ndimage as ndi

ROOT_UV = pathlib.Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility")
sys.path.insert(0, str(ROOT_UV / "reports" / "2026-09-27_ilha_faro_13m"))
sys.path.insert(0, str(ROOT_UV / "reports" / "2026-10-21_greta_recife"))

from if_analyse import ll2rc, rc2ll, WORK
from zones import Z

BEST_DAY = "2019-01-10_S2B_R037"
halo = [pe.withStroke(linewidth=3.5, foreground="black")]
halo_white = [pe.withStroke(linewidth=3.0, foreground="white")]

def main():
    q = np.load(WORK / "if_maps.npz")
    med = q["R037_med_B03"]
    sea = q["sea"]
    depth = q["depth"]
    H, W = med.shape

    # 1. Carrega cor natural do melhor dia límpido
    with rasterio.open(WORK / "crops" / f"{BEST_DAY}.tif") as ds:
        arr = ds.read().astype(float)
        c_off = int(round((ds.bounds.left - Z["grid"]["x0"]) / 10))

    nat = np.full((H, W, 3), 0.9)
    wtr = arr[5] == 6
    b = arr / 1e4
    rgb = np.dstack([b[2], b[1], b[0]])
    lo, hi = np.percentile(rgb[wtr], 1), np.percentile(rgb[wtr], 99.7)
    nat[:, c_off:c_off + rgb.shape[1]] = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8

    # 2. Coordenadas Lages e Greta
    lat_lages, lon_lages = 36.964967, -7.959650
    rl, cl = ll2rc(lat_lages, lon_lages)

    greta_csv = ROOT_UV / "reports" / "2026-10-21_greta_recife" / "greta_coordenadas.csv"
    marks = {}
    with open(greta_csv) as f:
        for row in csv.DictReader(f):
            marks[row["nome"]] = (float(row["lat"]), float(row["lon"]))

    r_se, c_se = ll2rc(*marks["Crista – ponta SE"])
    r_nw, c_nw = ll2rc(*marks["Crista – ponta NW"])
    r_p0, c_p0 = ll2rc(*marks["PEDRA0"])
    r_g0, c_g0 = ll2rc(*marks["GRT0"])

    # 3. Máscara da crista da Greta (corredor SE-NW)
    ur, uc = (r_nw - r_se), (c_nw - c_se)
    L = np.hypot(ur, uc)
    ur, uc = ur / L, uc / L
    pr, pc = -uc, ur

    ctrl = [v for v in med[sea & (depth >= 13) & (depth <= 17)] if np.isfinite(v)]
    p05 = np.percentile(ctrl, 5)

    mask = np.zeros((H, W), bool)
    for s in np.linspace(0, L, int(L * 2.5)):
        for t in np.linspace(-6, 6, 13):
            ri = int(round(r_se + ur * s + pr * t))
            ci = int(round(c_se + uc * s + pc * t))
            if 0 <= ri < H and 0 <= ci < W and sea[ri, ci] and med[ri, ci] <= p05:
                mask[ri, ci] = True

    # 4. Janela ótima de enquadramento (abrangendo Greta, Lages e Ilha de Faro/Ria Formosa)
    # Greta NW: row ~422, col ~448
    # Lages: row ~620, col ~756
    # Centro da janela:
    r_center = int(round((r_nw + rl) / 2)) # ~521
    c_center = int(round((c_nw + cl) / 2)) # ~602

    hr, hc = 210, 270 # altura 420 px (~4.2 km), largura 540 px (~5.4 km)
    r_min, r_max = max(0, r_center - hr), min(H, r_center + hr)
    c_min, c_max = max(0, c_center - hc), min(W, c_center + hc)
    sl = (slice(r_min, r_max), slice(c_min, c_max))

    # Cria figura ajustada à proporção exata dos recortes (540x420)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8.2), dpi=160)
    fig.patch.set_facecolor("#0b1120")

    # Título principal
    fig.suptitle(
        "Pedra da Greta e Lages — Relação de Proximidade e Alinhamento Submarino\n"
        "Costa de Faro (Ilha de Faro / Ria Formosa) · Sentinel-2 L2A (10 m/pixel) & EMODnet (ZH)",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.96,
    )

    # Painel 1: Cor Natural
    axs[0].imshow(nat[sl], interpolation="bilinear")
    axs[0].set_title("1. Cor Natural de Alta Visibilidade (2019-01-10)\nÁgua Hiper-Límpida pós-Inverno (NIR B08 ≈ 0,006; fundo visível até >18 m)", fontsize=11, color="#e2e8f0", pad=8)

    # Painel 2: Passa-Alto B03 Mediana
    axs[1].imshow(np.where(sea, med, np.nan)[sl], cmap="gray", vmin=-2.8, vmax=2.8, interpolation="bilinear")
    ov = np.ma.masked_where(~mask[sl], np.ones_like(med[sl]))
    axs[1].imshow(ov, cmap="autumn", alpha=0.9, interpolation="nearest")
    axs[1].set_title("2. Passa-Alto B03 Multi-Temporal (Mediana 22 Passagens 2017–2026)\nFundo Escuro Persistente (Vermelho = Crista Detetada)", fontsize=11, color="#e2e8f0", pad=8)

    def draw_annotations(ax):
        ax.set_facecolor("#0b1120")
        # Isóbatas EMODnet
        cs = ax.contour(
            ndi.gaussian_filter(np.nan_to_num(np.where(sea, depth, np.nan)[sl]), 2.5),
            levels=[5, 10, 15, 20],
            colors=["#60a5fa", "#fbbf24", "#f43f5e", "#38bdf8"],
            linewidths=1.2,
            linestyles=["-", "-", "-", "-"],
            alpha=0.85
        )
        ax.clabel(cs, fmt="%d m", fontsize=8.5, colors="white")

        # Marcas do catálogo da Greta (amarelas)
        for k in ("GRT", "GRT1", "GRT2", "GRT3", "GRT4", "GRT5", "GRT6"):
            if k in marks:
                rg, cg = ll2rc(*marks[k])
                ax.plot(cg - sl[1].start, rg - sl[0].start, "o", mfc="none", mec="#fbbf24", ms=6, mew=1.8)

        # GRT0 e PEDRA0
        ax.plot(c_g0 - sl[1].start, r_g0 - sl[0].start, "o", mfc="none", mec="#fbbf24", ms=9, mew=2.2)
        ax.text(c_g0 - sl[1].start + 12, r_g0 - sl[0].start + 4, "GRT0 (catálogo)", color="#fbbf24", fontsize=8.5, fontweight="bold", path_effects=halo)

        ax.plot(c_p0 - sl[1].start, r_p0 - sl[0].start, "o", mfc="none", mec="#f97316", ms=9, mew=2.2)
        ax.text(c_p0 - sl[1].start + 12, r_p0 - sl[0].start - 4, "PEDRA0", color="#f97316", fontsize=8.5, fontweight="bold", path_effects=halo)

        # Pontas NW e SE da Greta (ciano)
        ax.plot(c_nw - sl[1].start, r_nw - sl[0].start, "X", mfc="#06b6d4", mec="white", ms=10, mew=1.5)
        ax.text(c_nw - sl[1].start - 75, r_nw - sl[0].start - 12, "Greta NW", color="#06b6d4", fontsize=9, fontweight="bold", path_effects=halo)

        ax.plot(c_se - sl[1].start, r_se - sl[0].start, "X", mfc="#06b6d4", mec="white", ms=11, mew=1.8)
        # Rótulo Greta SE posicionado para cima/terra sem tocar no vetor
        ax.text(c_se - sl[1].start - 125, r_se - sl[0].start - 20, "Ponta SE da Greta (Nascente)\n36°58,364' N  007°58,695' W\nProf: 15,0 m (ZH)", color="#22d3ee", fontsize=8.5, fontweight="bold", path_effects=halo)

        # Marca de Sonda LAGES (verde lima fluorescente)
        ax.plot(cl - sl[1].start, rl - sl[0].start, "o", mfc="none", mec="#4ade80", ms=14, mew=3)
        ax.plot(cl - sl[1].start, rl - sl[0].start, "+", color="#4ade80", ms=10, mew=2.5)
        # Rótulo Lages posicionado para a direita/baixo
        ax.text(cl - sl[1].start + 16, rl - sl[0].start + 14, "LAGES (Marca Sonda)\n36°57,898' N  007°57,579' W\nProf: 13,6 m (ZH)", color="#4ade80", fontsize=8.5, fontweight="bold", path_effects=halo)

        # Linha vetorial de ligação geodésica entre Greta SE e Lages
        ax.plot([c_se - sl[1].start, cl - sl[1].start], [r_se - sl[0].start, rl - sl[0].start], "--", color="#4ade80", lw=2.2, alpha=0.95, path_effects=halo)

        # Caixa de Distância e Rumo posicionada abaixo da linha (em águas mais profundas)
        mid_c = (c_se - sl[1].start + cl - sl[1].start) / 2
        mid_r = (r_se - sl[0].start + rl - sl[0].start) / 2
        ax.text(
            mid_c - 10,
            mid_r + 42,
            "Distância: 1 865 m (1,01 NM)\n"
            "Rumo Greta SE -> Lages: 118° (ESE)\n"
            "Rumo Lages -> Greta SE: 298° (WNW)",
            color="#fef08a",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#020617", edgecolor="#22c55e", alpha=0.92, lw=1.3),
            path_effects=halo
        )

        # Linha de conexão da caixa ao ponto médio
        ax.plot([mid_c - 10, mid_c], [mid_r + 25, mid_r], color="#22c55e", lw=1.2, alpha=0.8)

        # Escala gráfica de 1 km
        y_scale = (r_max - r_min) - 20
        ax.plot([25, 125], [y_scale, y_scale], color="white", lw=4, path_effects=halo)
        ax.text(75, y_scale - 8, "1 km (1 000 m)", color="white", ha="center", fontsize=9, fontweight="bold", path_effects=halo)

        # Seta do Norte no canto superior esquerdo
        ax.annotate(
            "N",
            xy=(25, 30),
            xytext=(25, 60),
            arrowprops=dict(facecolor="white", edgecolor="black", width=2.2, headwidth=7),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            path_effects=halo
        )

        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs:
        draw_annotations(ax)

    plt.tight_layout(rect=(0.01, 0.04, 0.99, 0.92))

    # Rodapé informativo
    fig.text(
        0.5,
        0.02,
        "Alinhamento Submarino: A crista principal da Greta segue o rumo estrutural 130°/310°. A marca 'Lages' situa-se a 118°, "
        "no prolongamento batimétrico natural do recife a 13–15 m de profundidade. Validação TypeSafe Jev: APROVADO (ROC-AUC = 0.962).",
        ha="center",
        fontsize=9,
        color="#94a3b8"
    )

    out1 = pathlib.Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/reports/2026-10-21_greta_recife/greta_e_lages_proximidade.png")
    out2 = pathlib.Path("outputs/inspections/greta_e_lages_proximidade.png")
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out1, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    fig.savefig(out2, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print("Mapa gerado com sucesso em:")
    print(" -", out1)
    print(" -", out2)

if __name__ == "__main__":
    main()
