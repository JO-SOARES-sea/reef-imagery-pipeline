#!/usr/bin/env python3
"""
scripts/persistent_structures_ml.py — Base de estruturas persistentes do fundo e classificador rocha vs areia.

Ponto de partida: os mapas de persistência do UnderwaterVisibility (`data/raw/seabed/work/<zona>/<zona>_maps.npz`),
uma grelha de 10 m (UTM 29N) por zona com a mediana do passa-alto B03 de todas as passagens limpas da órbita R037
(2017–2026), as duas metades do arquivo, a fração de dias escuros e os controlos de superfície B04/B08.

1. Estruturas persistentes = componentes ligadas escuras na mediana (z ≤ −1) e nas duas metades (≤ −0,5), a 3–25 m.
2. Feições por estrutura: forma (área, alongamento, orientação face às isóbatas), persistência (z, metades, fração de
   dias escuros), controlos de superfície (B08/B04), contraste com o anel envolvente, profundidade, densidade de vizinhas.
3. Rótulos de fontes independentes do satélite:
     rocha  — crista da Greta e Zimbral (sonda do barco, confirmados pelo Rui); recifes da Santa Eulália (observação
              subaquática, DL-020); cumes CCMAR do Valado (multifeixe); afloramentos de Teixeira 2025 (sonar IH + 383
              mergulhos, AIA 3587).
     areia/lodo — estruturas persistentes dentro do mapa de Teixeira, a ≥ 100 m de rocha, plataforma de abrasão e
              recifes artificiais.
4. Validação: (a) blocos espaciais de 1 km dentro do troço com os dois rótulos; (b) transferência — modelo treinado
   só com Teixeira, aplicado às outras zonas: as rochas confirmadas ficam no topo do ranking da sua zona?
5. Saída: `data/external/persistent_structures/estruturas_persistentes.geojson` (+ CSV) com a probabilidade de rocha,
   a fonte do rótulo e as feições de todas as estruturas de todas as zonas, e `metrics.json`.

Persistente não é sinónimo de rocha: lodos escuros, limites de sedimento, formas de fundo e objetos humanos também
persistem. É isso que o classificador tenta separar — e por isso as probabilidades são para priorizar verificação no mar.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, shape
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("persistent_structures_ml")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UV = Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility")
WORK = UV / "data" / "raw" / "seabed" / "work"
ZONES_PY = UV / "reports" / "2026-09-27_ilha_faro_13m" / "zones.py"
TEIXEIRA = PROJECT_ROOT / "data" / "external" / "eia3587" / "fig20_teixeira2025_layers.geojson"
VALADO_CUMES = PROJECT_ROOT / "data" / "vector" / "valado_cumes_ccmar.csv"
OUT_DIR = PROJECT_ROOT / "data" / "external" / "persistent_structures"
CRS = "EPSG:32629"

# Rocha confirmada por fonte independente do satélite (lat, lon). Linhas: lista de vértices.
CONFIRMED_ROCK_LINES = {
    "greta_crista_sonda_rui": [(36.972730, -7.978251), (36.975013, -7.981591), (36.984876, -7.996629)],
    "zimbral_sonda_rui_DL022": [(36.95910, -7.93553), (36.95927, -7.93774), (36.95941, -7.93945)],
}
CONFIRMED_ROCK_POINTS = {  # Santa Eulália, observação subaquática (UV DL-020)
    "santa_eulalia_terreno_DL020": [(37.069071, -8.210492), (37.06777, -8.21079), (37.07005, -8.20908)],
}

# Sem profundidade nem distância às barras: no mapa de Teixeira a profundidade sozinha separa rocha de lodo (AUC 0,83,
# os lodos do paleo-estuário estão junto à costa) e o modelo aprendia "mais fundo = rocha". Ficam só forma,
# persistência, controlos de superfície e contexto; a profundidade fica como coluna descritiva.
FEATURES = [
    "area_px", "elong", "angle_to_isobath", "z_mean", "z_min", "zA_mean", "zB_mean", "half_diff", "frac_dark",
    "b08_mean", "b04_mean", "ring_contrast", "neighbours_500m", "fill_ratio",
]


def load_zones() -> dict:
    spec = importlib.util.spec_from_file_location("uv_zones", ZONES_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ZONES


def _to_utm(lat, lon):
    return Transformer.from_crs(4326, CRS, always_xy=True).transform(lon, lat)


def reference_geometries() -> dict[str, gpd.GeoDataFrame]:
    rows = []
    for name, pts in CONFIRMED_ROCK_LINES.items():
        rows.append({"source": name, "geometry": LineString([_to_utm(la, lo) for la, lo in pts]).buffer(50)})
    for name, pts in CONFIRMED_ROCK_POINTS.items():
        for la, lo in pts:
            rows.append({"source": name, "geometry": Point(_to_utm(la, lo)).buffer(50)})
    cumes = pd.read_csv(VALADO_CUMES)
    for _, r in cumes.iterrows():
        rows.append({"source": "valado_cumes_ccmar", "geometry": Point(_to_utm(r.lat, r.lon)).buffer(50)})
    confirmed = gpd.GeoDataFrame(rows, crs=CRS)

    lith = gpd.read_file(TEIXEIRA).to_crs(CRS)
    rock = lith[lith.layer.str.startswith("rocha")]
    not_sand = lith[lith.layer.str.startswith(("rocha", "plataforma", "recifes artificiais"))]
    hull = gpd.GeoDataFrame(geometry=[lith.unary_union.convex_hull.buffer(-100)], crs=CRS)
    return {"confirmed": confirmed, "teixeira_rock": rock, "teixeira_not_sand": not_sand, "teixeira_hull": hull}


def extract_structures(zone_key: str, z: dict, grid: dict, refs: dict, prefix: str = "R037") -> gpd.GeoDataFrame:
    T = from_origin(grid["x0"], grid["y1"], 10, 10)
    med, A, B = z[f"{prefix}_med_B03"], z[f"{prefix}_A_B03"], z[f"{prefix}_B_B03"]
    depth, sea = z["depth"], z["sea"].astype(bool)
    ok = sea & np.isfinite(med) & (depth >= 3) & (depth <= 25)
    cand = ok & (med <= -1.0) & (A <= -0.5) & (B <= -0.5)
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    if n == 0:
        return gpd.GeoDataFrame()
    objs = ndi.find_objects(lab)
    gy, gx = np.gradient(ndi.gaussian_filter(np.where(ok, depth, np.nanmedian(depth[ok])), 5))
    centroids = np.array(ndi.center_of_mass(cand, lab, range(1, n + 1)))

    recs = []
    for i, sl in enumerate(objs, start=1):
        m = lab[sl] == i
        npx = int(m.sum())
        if npx < 3:
            continue
        rr, cc = np.nonzero(m)
        rr = rr + sl[0].start; cc = cc + sl[1].start
        coords = np.column_stack([cc, -rr]).astype(float)
        if npx >= 3:
            ev, evec = np.linalg.eigh(np.cov(coords.T) + 1e-6 * np.eye(2))
            elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
            ax = evec[:, 1]
        r0, c0 = int(np.median(rr)), int(np.median(cc))
        iso = np.array([gy[r0, c0], -gx[r0, c0]])  # direção da isóbata (perpendicular ao gradiente)
        cosang = abs(ax @ iso) / max(np.linalg.norm(iso), 1e-9)
        ring = ndi.binary_dilation(lab == i, iterations=3) & ~(lab == i) & ok
        bbox_area = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        recs.append({
            "zone": zone_key, "obj_id": f"{zone_key}_{i}",
            "area_px": npx, "elong": elong, "angle_to_isobath": float(np.degrees(np.arccos(np.clip(cosang, 0, 1)))),
            "z_mean": float(med[rr, cc].mean()), "z_min": float(med[rr, cc].min()),
            "zA_mean": float(A[rr, cc].mean()), "zB_mean": float(B[rr, cc].mean()),
            "half_diff": float(abs(A[rr, cc].mean() - B[rr, cc].mean())),
            "frac_dark": float(np.nanmean(z[f"{prefix}_frac_dark"][rr, cc])),
            "b08_mean": float(np.nanmean(z[f"{prefix}_med_B08"][rr, cc])), "b04_mean": float(np.nanmean(z[f"{prefix}_med_B04"][rr, cc])),
            "ring_contrast": float(med[rr, cc].mean() - (np.nanmean(med[ring]) if ring.any() else 0.0)),
            "depth_mean": float(np.nanmean(depth[rr, cc])), "d_inlet_min": float(np.nanmin(z["d_inlet"][rr, cc])),
            "fill_ratio": npx / max(bbox_area, 1),
            "cy": centroids[i - 1][0], "cx": centroids[i - 1][1],
            "mask": (lab[sl] == i), "sl": sl,
        })
    if not recs:
        return gpd.GeoDataFrame()

    # Polígonos e vizinhança
    geoms = []
    for r in recs:
        sl = r.pop("sl"); m = r.pop("mask")
        sub_T = T * T.translation(sl[1].start, sl[0].start)
        from rasterio.features import shapes
        polys = [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=sub_T) if v == 1]
        geoms.append(max(polys, key=lambda p: p.area) if polys else Point(*(T * (r["cx"], r["cy"]))).buffer(5))
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=CRS)
    cent = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    d = np.hypot(cent[:, None, 0] - cent[None, :, 0], cent[:, None, 1] - cent[None, :, 1])
    gdf["neighbours_500m"] = (d < 500).sum(axis=1) - 1

    # Rótulos
    gdf["label"] = np.nan; gdf["label_source"] = ""
    conf = gpd.sjoin(gdf[["obj_id", "geometry"]], refs["confirmed"], predicate="intersects")
    for oid, src in conf.groupby("obj_id")["source"].first().items():
        gdf.loc[gdf.obj_id == oid, ["label", "label_source"]] = [1, src]
    rock_b = refs["teixeira_rock"].buffer(20).unary_union
    hull = refs["teixeira_hull"].geometry.iloc[0]
    notsand_far = refs["teixeira_not_sand"].buffer(100).unary_union
    for j, g in gdf.geometry.items():
        if gdf.at[j, "label_source"]:
            continue
        if g.intersection(rock_b).area >= 0.5 * g.area:
            gdf.loc[j, ["label", "label_source"]] = [1, "teixeira_rocha"]
        elif hull.contains(g) and not g.intersects(notsand_far):
            gdf.loc[j, ["label", "label_source"]] = [0, "teixeira_areia_lodo"]
    return gdf.drop(columns=["cx", "cy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Base de estruturas persistentes e classificador rocha vs areia")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zones = load_zones()
    refs = reference_geometries()
    parts = []
    for key, Z in zones.items():
        f = WORK / Z["work"] / f"{Z['prefix']}_maps.npz"
        if not f.exists():
            logger.warning(f"{key}: sem mapas ({f})"); continue
        g = extract_structures(key, dict(np.load(f)), Z["grid"], refs)
        logger.info(f"{key}: {len(g)} estruturas persistentes; rótulos: {g.label_source.value_counts().to_dict() if len(g) else {}}")
        parts.append(g)
    # Zonas sobrepostas (FZW/FZE, IF/ANCAO) geram duplicados: ficam ambos, mas com o mesmo bloco espacial
    S = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    S["block_1km"] = (S.geometry.centroid.x // 1000).astype(int).astype(str) + "_" + (S.geometry.centroid.y // 1000).astype(int).astype(str)
    X = S[FEATURES].to_numpy(float)
    lab = S.label.to_numpy()
    model = lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_depth=4, class_weight="balanced",
                                                   random_state=args.seed)
    metrics = {"n_structures": int(len(S)), "labels": S.label_source.value_counts().to_dict()}

    # (a) dentro do troço de Teixeira (os dois rótulos), blocos de 1 km
    tx = S.label_source.str.startswith("teixeira").to_numpy()
    Xt, yt, gt = X[tx], lab[tx].astype(int), S.block_1km.to_numpy()[tx]
    oof = np.full(len(yt), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(gt)))).split(Xt, yt, gt):
        oof[te] = model().fit(Xt[tr], yt[tr]).predict_proba(Xt[te])[:, 1]
    rng = np.random.default_rng(args.seed)
    blocks = np.unique(gt)
    boots = []
    for _ in range(args.n_boot):
        bs = rng.choice(blocks, len(blocks)); idx = np.concatenate([np.flatnonzero(gt == b) for b in bs])
        if len(set(yt[idx])) == 2:
            boots.append(roc_auc_score(yt[idx], oof[idx]))
    metrics["teixeira_spatial_cv"] = {"n_rock": int(yt.sum()), "n_sand_mud": int((yt == 0).sum()),
                                      "auc": float(roc_auc_score(yt, oof)), "auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    metrics["teixeira_baseline_z_mean_auc"] = float(roc_auc_score(yt, -Xt[:, FEATURES.index("z_mean")]))
    dt = S.depth_mean.to_numpy()[tx]
    metrics["teixeira_baseline_depth_only_auc"] = float(roc_auc_score(yt, dt))
    pairs = [float(oof[i] > oof[j]) + 0.5 * float(oof[i] == oof[j])
             for i in np.flatnonzero(yt == 1) for j in np.flatnonzero(yt == 0) if abs(dt[i] - dt[j]) <= 1.5]
    metrics["teixeira_spatial_cv"]["auc_depth_matched_1p5m"] = float(np.mean(pairs))

    # (b) transferência: treino só com Teixeira, aplicação às rochas confirmadas independentes de outras zonas
    m_tx = model().fit(Xt, yt)
    S["p_rock_teixeira_only"] = m_tx.predict_proba(X)[:, 1]
    transfer = {}
    for src in ("greta_crista_sonda_rui", "zimbral_sonda_rui_DL022", "santa_eulalia_terreno_DL020", "valado_cumes_ccmar"):
        sel = S.label_source == src
        if not sel.any():
            transfer[src] = "sem estruturas persistentes sobre a referência"; continue
        zone = S.loc[sel, "zone"].iloc[0]
        others = (S.zone == zone) & ~sel & S.label.isna()
        d_conf = S.loc[sel, "depth_mean"].to_numpy()
        y = np.r_[np.ones(sel.sum()), np.zeros(others.sum())]
        s = np.r_[S.loc[sel, "p_rock_teixeira_only"], S.loc[others, "p_rock_teixeira_only"]]
        transfer[src] = {"zone": zone, "n_confirmed": int(sel.sum()), "n_other_unlabelled": int(others.sum()),
                         "auc_vs_other_persistent": float(roc_auc_score(y, s)),
                         "median_percentile_in_zone": float(np.median([(S.loc[others, 'p_rock_teixeira_only'] < v).mean() * 100
                                                                       for v in S.loc[sel, 'p_rock_teixeira_only']])),
                         # só contra estruturas à mesma profundidade (±1,5 m): a profundidade não pode explicar o ranking
                         "median_percentile_same_depth": float(np.median([
                             (S.loc[others & ((S.depth_mean - d).abs() <= 1.5), "p_rock_teixeira_only"] < v).mean() * 100
                             for v, d in zip(S.loc[sel, "p_rock_teixeira_only"], d_conf)]))}
    metrics["transfer_from_teixeira"] = transfer

    # Modelo final com todos os rótulos
    has = ~np.isnan(lab)
    final = model().fit(X[has], lab[has].astype(int))
    S["p_rock"] = final.predict_proba(X)[:, 1]
    S["classe"] = np.select([S.p_rock >= 0.7, S.p_rock <= 0.3], ["provável rocha", "provável areia/lodo"], "incerta")
    S["area_m2"] = S.area_px * 100
    out = S.drop(columns=["block_1km"]).to_crs(4326)
    cent = out.geometry.centroid
    out["lat"], out["lon"] = cent.y.round(6), cent.x.round(6)
    out.to_file(OUT_DIR / "estruturas_persistentes.geojson", driver="GeoJSON")
    out.drop(columns="geometry").to_csv(OUT_DIR / "estruturas_persistentes.csv", index=False)
    metrics["classes"] = out.classe.value_counts().to_dict()
    metrics["by_zone"] = out.groupby("zone").classe.value_counts().unstack(fill_value=0).to_dict(orient="index")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
