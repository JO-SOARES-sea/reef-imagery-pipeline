#!/usr/bin/env python3
"""
scripts/build_structures_db.py — Base costeira de estruturas persistentes e lista de verificação para a sonda.

Aplica o modelo congelado (`--model`, de run_h4_classifier.py --save-model) a todas as zonas com pilha de
desenvolvimento e gera, em data/external/persistent_structures/:
  - estruturas_v2.geojson / .csv / .gpx — uma linha por estrutura (componente ligado com P ≥ 0,5), com
    p_rocha calibrada (média e máxima), classe (rocha / areia-lodo / artificial / incerta), relevo medido onde há COSMO
    (nível a) ou "não estimável" (o relevo NÃO se estima a partir de satélite nem do sonar do IH: ledger E005),
    anos de persistência (anos com mediana anual do mesmo sinal e |z| ≥ 0,5), nível de evidência e fonte;
  - verificacao_sonda.gpx / .csv — as N estruturas com mais valor de informação (incertas, grandes, sem confirmação),
    nomes de 8 caracteres no máximo, e colunas vazias para devolver: hora, profundidade, pedra/areia.
Estruturas dentro do holdout só entram com `--include-holdout` (depois do teste final).

Níveis de evidência: (a) confirmação independente (COSMO rocha estável, cumes CCMAR, sonda do Rui com marca,
observação subaquática, rocha de Teixeira); (b) conhecimento local (marcas do plotter a ≤ 50 m); (c) só satélite.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, shape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.constants import HOLDOUT_BBOX_WGS84  # noqa: E402

_s = importlib.util.spec_from_file_location("h4", PROJECT_ROOT / "scripts" / "run_h4_classifier.py")
h4 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(h4)
PDIR = h4.PDIR
OUT = PROJECT_ROOT / "data" / "external" / "persistent_structures"
TRUTH = PROJECT_ROOT / "data" / "processed" / "cosmo_truth"
ZONES = ["valado", "se", "ancao", "if", "fzw", "fze", "cacela"]
MAX_STRUCTURE_M2 = 500_000   # acima disto não é uma estrutura discreta, é uma zona extensa (fora da lista de verificação)
MAX_VERIFY_PER_ZONE = 6

CONFIRMED_LINES = {"greta_crista_sonda_rui": [(36.972730, -7.978251), (36.975013, -7.981591), (36.984876, -7.996629)]}
CONFIRMED_POINTS = {
    "zimbral_sonda_rui": [(36.960233, -7.944583), (36.959667, -7.940767), (36.959483, -7.939667), (36.959317, -7.937933), (36.959083, -7.936083)],
    "pedra_do_alto_gps_rui": [(37.05815, -8.209817)],
    "vilha_cosmo": [(37.003483, -8.005033)],
    "santa_eulalia_terreno_DL020": [(37.069071, -8.210492), (37.06777, -8.21079), (37.07005, -8.20908)],
}


def dmm(v: float, pos: str, neg: str, w: int) -> str:
    s = pos if v >= 0 else neg
    v = abs(v)
    d = int(v)
    return f"{d:0{w}d}°{(v - d) * 60:06.3f}'".replace(".", ",") + f" {s}"


def years_persistent(zone: str, zmed: np.ndarray) -> np.ndarray:
    stack = np.load(PDIR / f"{zone}_dev_stack_B03.npy", mmap_mode="r")
    years = pd.read_csv(PDIR / f"{zone}_dev_stack_passes.csv").date.str[:4].astype(int).to_numpy()
    cnt = np.zeros(zmed.shape, np.int8)
    for yv in np.unique(years):
        sub = np.asarray(stack[years == yv], np.float32)
        with np.errstate(all="ignore"):
            m = np.nanmedian(sub, axis=0)
        n = np.isfinite(sub).sum(0)
        cnt += ((n >= 3) & (np.sign(m) == np.sign(zmed)) & (np.abs(m) >= 0.5)).astype(np.int8)
    return cnt


def evidence_layers():
    t = lambda la, lo: Transformer.from_crs(4326, 32629, always_xy=True).transform(lo, la)
    rows = []
    for k, pts in CONFIRMED_LINES.items():
        rows.append({"src": k, "geometry": LineString([t(*p) for p in pts]).buffer(60)})
    for k, pts in CONFIRMED_POINTS.items():
        rows += [{"src": k, "geometry": Point(t(*p)).buffer(60)} for p in pts]
    c = pd.read_csv(PROJECT_ROOT / "data" / "vector" / "valado_cumes_ccmar.csv")
    rows += [{"src": "valado_cumes_ccmar", "geometry": Point(t(la, lo)).buffer(60)} for la, lo in zip(c.lat, c.lon)]
    lith = gpd.read_file(PROJECT_ROOT / "data" / "external" / "eia3587" / "fig20_teixeira2025_layers.geojson").to_crs(32629)
    rows += [{"src": "teixeira_rocha", "geometry": g} for g in lith[lith.layer.str.startswith("rocha")].geometry]
    conf = gpd.GeoDataFrame(rows, crs=32629)
    art = list(gpd.read_file(PROJECT_ROOT / "data" / "external" / "psoem" / "lyr13_complexos_recifais_faro_quarteira.geojson").to_crs(32629).geometry)
    art += list(lith[lith.layer.str.startswith("recifes artificiais")].geometry)
    gps = pd.read_csv(PROJECT_ROOT / "data" / "vector" / "GPS_10_2026.csv").drop_duplicates(["latitude", "longitude"])
    gx, gy = Transformer.from_crs(4326, 32629, always_xy=True).transform(gps.longitude.values, gps.latitude.values)
    marks = gpd.GeoDataFrame({"name": gps.name.values}, geometry=gpd.points_from_xy(gx, gy), crs=32629)
    return conf, gpd.GeoSeries(art, crs=32629).unary_union, marks


def main() -> None:
    p = argparse.ArgumentParser(description="Base costeira de estruturas persistentes + lista para a sonda")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--n-verify", type=int, default=30)
    p.add_argument("--include-holdout", action="store_true")
    a = p.parse_args()
    bundle = joblib.load(a.model)
    m, feats = bundle["model"], bundle["features"]
    conf, art, marks = evidence_layers()
    W, S, E, N = HOLDOUT_BBOX_WGS84
    hbox = gpd.GeoSeries.from_xy([W, E], [S, N], crs=4326).to_crs(32629).unary_union.envelope
    inv = Transformer.from_crs(32629, 4326, always_xy=True)
    recs = []
    for zone in ZONES:
        if not (PDIR / f"{zone}_dev_maps.npz").exists():
            continue
        z = h4.zone_features(zone, bundle.get("with_h2", False))
        ok = z["sea"].astype(bool) & (z["depth"] >= 3) & (z["depth"] <= 25)
        for f in feats:
            ok &= np.isfinite(z["F"][f])
        P = np.full(z["shape"], np.nan, np.float32)
        ii = np.flatnonzero(ok.ravel())
        P.ravel()[ii] = m.predict_proba(np.column_stack([z["F"][f].ravel()[ii] for f in feats]))[:, 1]
        np.save(PDIR / f"{zone}_frozen_prob.npy", P)
        ny = years_persistent(zone, z["ALL_med_B03"])
        lab, n = ndi.label(np.nan_to_num(P) >= 0.5, structure=np.ones((3, 3)))
        rel_maps = []
        for f in TRUTH.glob("*_truth10m.tif"):
            with rasterio.open(f) as src:
                r_ = np.full(z["shape"], np.nan, np.float32)
                reproject(src.read(2), r_, src_transform=src.transform, src_crs=src.crs, dst_transform=z["T"], dst_crs="EPSG:32629",
                          resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
                rel_maps.append((r_, f.stem.split("_")[0]))
        geoms_by_label: dict[int, list] = {}
        for geom_d, v in shapes(lab.astype(np.int32), mask=lab > 0, transform=z["T"]):
            geoms_by_label.setdefault(int(v), []).append(shape(geom_d))
        objs = ndi.find_objects(lab)
        for v, parts in geoms_by_label.items():
            from shapely.ops import unary_union
            g = unary_union(parts)
            sl = objs[v - 1]
            sel = np.zeros(z["shape"], bool)
            sel[sl] = lab[sl] == v
            if sel.sum() < 2:
                continue
            if g.intersects(hbox) and not a.include_holdout:
                continue
            pm, px = float(np.nanmean(P[sel])), float(np.nanmax(P[sel]))
            c = g.centroid
            lon, lat = inv.transform(c.x, c.y)
            hits = conf[conf.intersects(g)]
            near = marks[marks.distance(g) <= 50]
            level, src = ("a", ";".join(sorted(set(hits.src)))) if len(hits) else (("b", ";".join(sorted(set(near.name.astype(str))))[:120]) if len(near) else ("c", "satélite"))
            is_art = bool(g.intersects(art))
            relief = None
            for r_, sector in rel_maps:
                vals = r_[sel]
                if np.isfinite(vals).any():
                    relief = (float(np.nanmax(vals)), sector)
            classe = "artificial" if is_art else ("rocha" if pm >= 0.7 else ("areia-lodo" if pm <= 0.3 else "incerta"))
            recs.append({"zona": zone, "lat": round(lat, 6), "lon": round(lon, 6), "plotter": f"{dmm(lat, 'N', 'S', 2)} {dmm(lon, 'E', 'W', 3)}",
                         "prof_emodnet_m": round(float(np.nanmedian(z['depth'][sel])), 1), "area_m2": int(sel.sum() * 100),
                         "p_rocha_media": round(pm, 3), "p_rocha_max": round(px, 3), "classe": classe,
                         "relevo_m": round(relief[0], 2) if relief else None, "relevo_fonte": f"COSMO {relief[1]} (medido)" if relief else "não estimável (E005)",
                         "anos_persistencia": int(np.nanmax(ny[sel])), "sinal_b03": "escuro" if np.nanmedian(z["ALL_med_B03"][sel]) < 0 else "claro",
                         "nivel_evidencia": level, "fonte_rotulo": src, "recife_artificial": is_art,
                         "zona_extensa": bool(sel.sum() * 100 > MAX_STRUCTURE_M2),
                         "holdout": bool(g.intersects(hbox)), "geometry": g})
    D = gpd.GeoDataFrame(recs, crs=32629).to_crs(4326)
    D.insert(0, "id", [f"E{i:05d}" for i in range(1, len(D) + 1)])
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_file(OUT / "estruturas_v2.geojson", driver="GeoJSON")
    D.drop(columns="geometry").to_csv(OUT / "estruturas_v2.csv", index=False)
    pts = D.copy(); pts["geometry"] = gpd.points_from_xy(pts.lon, pts.lat)
    pts[["id", "geometry"]].rename(columns={"id": "name"}).to_file(OUT / "estruturas_v2.gpx", driver="GPX", layer="waypoints")
    # lista de verificação: incerteza × tamanho, mais as rochas prováveis só com evidência de satélite
    cand = D[(~D.recife_artificial) & (~D.zona_extensa)].copy()
    unc = 1 - np.abs(2 * cand.p_rocha_media - 1)
    cand["valor_informacao"] = unc * np.log10(cand.area_m2 + 10) + 0.5 * ((cand.nivel_evidencia == "c") & (cand.p_rocha_media >= 0.7))
    V = (cand.sort_values("valor_informacao", ascending=False).groupby("zona", group_keys=False).head(MAX_VERIFY_PER_ZONE)
         .sort_values("valor_informacao", ascending=False).head(a.n_verify).copy())
    V["nome_gpx"] = [f"V{zz[:2].upper()}{i:03d}" for i, zz in enumerate(V.zona, 1)]
    for col in ("hora_utc", "profundidade_sonda_m", "pedra_ou_areia", "notas"):
        V[col] = ""
    V.drop(columns="geometry").to_csv(OUT / "verificacao_sonda.csv", index=False)
    vp = V.copy(); vp["geometry"] = gpd.points_from_xy(vp.lon, vp.lat)
    vp[["nome_gpx", "geometry"]].rename(columns={"nome_gpx": "name"}).to_file(OUT / "verificacao_sonda.gpx", driver="GPX", layer="waypoints")
    summ = {"n_estruturas": int(len(D)), "por_classe": D.classe.value_counts().to_dict(), "por_nivel": D.nivel_evidencia.value_counts().to_dict(),
            "por_zona": D.zona.value_counts().to_dict(), "modelo": str(a.model), "n_verificacao": int(len(V))}
    (OUT / "estruturas_v2_resumo.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
