#!/usr/bin/env python3
"""
scripts/build_persistence_map.py — Pilha e mapa de persistência do fundo (passa-alto B03 por passagem).

Reproduz o método do UnderwaterVisibility (`reports/2026-09-27_ilha_faro_13m`, DL-020/DL-024), que deteta estruturas
estreitas do fundo melhor do que uma composição de reflectância:
  1. todas as passagens Sentinel-2 L2A no intervalo de datas (por omissão o de DESENVOLVIMENTO, 2017–2024: as passagens
     ≥ 2025-01-01 estão reservadas pelo pré-registo `docs/HOLDOUT_PREREGISTO.md`), sem reprocessamentos duplicados;
  2. por passagem: máscara de água limpa (NDWI, SCL sem nuvem/sombra, B08 sem glint/esteiras); só passagens com
     ≥ ``--min-valid`` da água válida;
  3. passa-alto DoG (σ 1–10 px = 10–100 m) de B03, B04, B08 e B11 com convolução normalizada sobre a água válida;
  4. correção de glint residual tipo Hedley no passa-alto: B03 − k·B11, com k estimado em água funda, aplicada só se
     reduzir o ruído em água funda (como o UV);
  5. normalização z pelo ruído da água funda (mediana e desvio robusto a ≥ ``--deep-m`` m);
  6. mediana de todas as passagens, das duas metades (anos ímpares / pares) e fração de dias escuros (z ≤ −2).

Saídas em `data/processed/persistence/`:
  - `<nome>_maps.npz` com as chaves do UV (prefixo ``ALL_``) para `scripts/persistent_structures_ml.py`;
  - `<nome>_maps.json` com a grelha, as passagens usadas e a versão;
  - com ``--save-stack``: `<nome>_stack_{B03,B04,B08}.npy` (float16, passagens × linhas × colunas) e
    `<nome>_stack_passes.csv` (data, órbita, fração válida, ruído em água funda, k do deglint) — a pilha que as
    hipóteses H1 (decomposição temporal) e H3 (mobilidade) usam.

Uso:
    python scripts/build_persistence_map.py --uv-zone IF --name if_dev --save-stack
    python scripts/build_persistence_map.py --bbox -7.60 37.07 -7.44 37.16 --name cacela \\
        --emodnet data/processed/EMODnet_batimetria_Cacela.tif
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage as ndi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_persistence_map")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.constants import DEV_DATE_RANGE, HOLDOUT_FIRST_DATE  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "persistence"
UV_ZONES_PY = Path("/Volumes/4G_WD_BLACK/dev/UnderwaterVisibility/reports/2026-09-27_ilha_faro_13m/zones.py")
VERSION = "2026-10-26.3"

_spec = importlib.util.spec_from_file_location("bcw", PROJECT_ROOT / "scripts" / "build_clear_water_composite.py")
bcw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcw)


def _dog(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """DoG σ1−σ10 px com convolução normalizada (os píxeis inválidos não contaminam o filtro)."""
    a = np.where(valid, arr, 0.0)
    w = valid.astype(np.float32)

    def g(s):
        return ndi.gaussian_filter(a, s) / np.maximum(ndi.gaussian_filter(w, s), 1e-6)

    return np.where(valid, g(1) - g(10), np.nan)


def _robust_z(hp: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = hp[ref & np.isfinite(hp)]
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    return (hp - med) / max(s, 1e-9)


def _deglint(hp3: np.ndarray, hp11: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """B03 − k·B11 no passa-alto (Hedley et al. 2005), só se reduzir o ruído em água funda."""
    m = ref & np.isfinite(hp3) & np.isfinite(hp11)
    if m.sum() < 500:
        return hp3, 0.0
    x, y = hp11[m], hp3[m]
    k = float(np.cov(x, y)[0, 1] / max(np.var(x), 1e-12))
    corr = hp3 - k * hp11
    return (corr, k) if np.nanstd(corr[m]) < np.nanstd(y) else (hp3, 0.0)


def process_pass(item, grid, sea, deep, glint_nir, min_valid, prescreen=False):
    try:
        off = bcw._boa_offset_dn(item)

        def rd(b, rs=Resampling.bilinear):
            return bcw._read_on_grid(item.assets[b].href, grid, rs)

        scl = rd("SCL", Resampling.nearest)
        if prescreen:
            # limite superior exato da fração válida (a máscara final é um subconjunto): evita ler as bandas de 10 m
            bound = (sea & ~np.isin(scl, bcw.SCL_INVALID)).sum() / max(sea.sum(), 1)
            if bound < min_valid:
                return {"id": item.id, "used": False, "valid_frac": float(bound)}
        b3, b4, b8, b11 = (bcw._to_reflectance(rd(b), off) for b in ("B03", "B04", "B08", "B11"))
    except Exception as exc:
        logger.warning(f"{item.id}: leitura falhou ({exc})")
        return None
    valid = sea & (b3 > 0) & ~np.isin(scl, bcw.SCL_INVALID) & (b8 < glint_nir) & (b3 > b8)
    frac = valid.sum() / max(sea.sum(), 1)
    if frac < min_valid or (valid & deep).sum() < 500:
        return {"id": item.id, "used": False, "valid_frac": float(frac)}
    ref = valid & deep
    hp3, k = _deglint(_dog(b3, valid), _dog(b11, valid), ref)
    return {
        "id": item.id, "used": True, "valid_frac": float(frac), "date": item.datetime.strftime("%Y-%m-%d"),
        "year": item.datetime.year, "orbit": item.properties.get("sat:relative_orbit"),
        "deep_noise": float(np.nanstd(hp3[ref])), "deglint_k": k,
        "B03": _robust_z(hp3, ref).astype(np.float16),
        "B04": _robust_z(_dog(b4, valid), ref).astype(np.float16),
        "B08": _robust_z(_dog(b8, valid), ref).astype(np.float16),
    }


def _load_reuse(name: str, grid: dict) -> dict:
    """Passagens já calculadas noutra pilha da MESMA grelha e parâmetros (só muda --min-valid): id -> resultado."""
    meta = json.loads((OUT_DIR / f"{name}_maps.json").read_text())
    T = grid["transform"]
    if (meta["x0"], meta["y1"]) != (T.c, T.f):
        raise SystemExit(f"--reuse-stack {name}: grelha diferente")
    stacks = {b: np.load(OUT_DIR / f"{name}_stack_{b}.npy", mmap_mode="r") for b in ("B03", "B04", "B08")}
    if stacks["B03"].shape[1:] != (grid["height"], grid["width"]):
        raise SystemExit(f"--reuse-stack {name}: forma diferente")
    out = {}
    with open(OUT_DIR / f"{name}_stack_passes.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            i = int(row["idx"])
            out[row["id"]] = {"id": row["id"], "used": True, "valid_frac": float(row["valid_frac"]), "date": row["date"],
                              "year": int(row["date"][:4]), "orbit": int(row["orbit"]), "deep_noise": float(row["deep_noise"]),
                              "deglint_k": float(row["deglint_k"]), "_idx": i, "_stacks": stacks}
    return out


def _materialize(r: dict) -> dict:
    if "_stacks" in r:
        st, i = r.pop("_stacks"), r.pop("_idx")
        r.update({b: np.asarray(st[b][i], np.float16) for b in ("B03", "B04", "B08")})
    return r


def _uv_grid(zone: str) -> tuple[dict, tuple[float, float, float, float]]:
    spec = importlib.util.spec_from_file_location("uvz", UV_ZONES_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = mod.ZONES[zone]["grid"]
    crs = rasterio.crs.CRS.from_epsg(32629)
    T = from_origin(g["x0"], g["y1"], 10, 10)
    grid = {"crs": crs, "transform": T, "width": g["nx"], "height": g["ny"]}
    bbox = rasterio.warp.transform_bounds(crs, "EPSG:4326", g["x0"], g["y1"] - 10 * g["ny"], g["x0"] + 10 * g["nx"], g["y1"])
    return grid, bbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilha e mapa de persistência do fundo (passa-alto B03 por passagem)")
    loc = parser.add_mutually_exclusive_group(required=True)
    loc.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    loc.add_argument("--uv-zone", help="Usa a grelha exata de uma zona do UV (VALADO, SE, ANCAO, IF, FZW, FZE).")
    parser.add_argument("--name", required=True)
    parser.add_argument("--emodnet", type=Path, default=None, help="Raster EMODnet (cota). Por omissão é descarregado.")
    parser.add_argument("--date-range", default=DEV_DATE_RANGE)
    parser.add_argument("--max-cloud", type=float, default=15.0)
    parser.add_argument("--min-valid", type=float, default=0.9)
    parser.add_argument("--glint-nir", type=float, default=0.03)
    parser.add_argument("--deep-m", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-passes", type=int, default=400, help="Mantém só as N passagens válidas mais limpas (memória).")
    parser.add_argument("--save-stack", action="store_true", help="Grava a pilha por passagem (H1/H3).")
    parser.add_argument("--scl-prescreen", action="store_true",
                        help="Lê primeiro só a SCL e salta as passagens cujo limite superior da fração válida < --min-valid.")
    parser.add_argument("--reuse-stack", default=None,
                        help="Reutiliza as passagens de outra pilha com a mesma grelha (ex.: se_dev), sem as voltar a ler.")
    args = parser.parse_args()

    end = args.date_range.split("/")[-1]
    if end >= HOLDOUT_FIRST_DATE:
        logger.warning(f"O intervalo {args.date_range} inclui passagens do holdout (≥ {HOLDOUT_FIRST_DATE}): "
                       "só para a avaliação final pré-registada.")

    if args.uv_zone:
        grid, bbox = _uv_grid(args.uv_zone)
    else:
        bbox = tuple(args.bbox)
        grid, _ = bcw._grid_from_bbox(bbox)
    shape = (grid["height"], grid["width"])

    emod_path = args.emodnet
    if emod_path is None:
        from src.emodnet_bathymetry import fetch_depth

        emod_path = PROJECT_ROOT / "data" / "processed" / f"EMODnet_batimetria_{args.name}.tif"
        if not emod_path.exists():
            fetch_depth(tuple(round(v, 4) for v in (bbox[0] - .01, bbox[1] - .01, bbox[2] + .01, bbox[3] + .01)), emod_path)
    depth = np.full(shape, np.nan, np.float32)
    with rasterio.open(emod_path) as e:
        src = e.read(1).astype(np.float32)
        if e.nodata is not None:
            src[src == e.nodata] = np.nan
        reproject(src, depth, src_transform=e.transform, src_crs=e.crs, dst_transform=grid["transform"], dst_crs=grid["crs"],
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    depth = -depth  # EMODnet: cota negativa no mar -> profundidade positiva
    sea = np.isfinite(depth) & (depth > 0.5)
    deep = sea & (depth >= args.deep_m)
    if deep.sum() < 2000:
        deep = sea & (depth >= np.nanpercentile(depth[sea], 75))
        logger.warning(f"Pouca água ≥ {args.deep_m} m: referência de ruído = quartil mais fundo ({np.nanmin(depth[deep]):.1f} m+)")

    client = Client.open(bcw.STAC_URL, modifier=pc.sign_inplace)
    items = list(client.search(collections=["sentinel-2-l2a"], bbox=list(bbox), datetime=args.date_range,
                               query={"eo:cloud_cover": {"lt": args.max_cloud}}, max_items=2000).items())
    items = bcw._latest_processing_per_acquisition(items)
    logger.info(f"{len(items)} aquisições com nuvens < {args.max_cloud}% em {args.date_range}")

    reuse = _load_reuse(args.reuse_stack, grid) if args.reuse_stack else {}
    if reuse:
        logger.info(f"{sum(it.id in reuse for it in items)} passagens reutilizadas de {args.reuse_stack}")

    def work(it):
        if it.id in reuse:
            return _materialize(dict(reuse[it.id]))
        return process_pass(it, grid, sea, deep, args.glint_nir, args.min_valid, args.scl_prescreen)

    used, n_read, n_failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(work, items):
            n_read += 1
            if r is None:
                n_failed += 1
                continue
            if r["used"]:
                used.append(r)
                if len(used) > args.max_passes:  # descarta a passagem mais ruidosa para limitar a memória
                    used.remove(max(used, key=lambda q: q["deep_noise"]))
            if n_read % 100 == 0:
                logger.info(f"  {n_read}/{len(items)} lidas, {len(used)} limpas guardadas, {n_failed} falhas")
    if n_failed > 0.05 * len(items):
        raise SystemExit(f"{n_failed}/{len(items)} leituras falharam (rede?): pilha não gravada.")
    used.sort(key=lambda r: r["date"])
    logger.info(f"Passagens limpas usadas: {len(used)} de {n_read} lidas")
    if len(used) < 6:
        raise SystemExit("Passagens limpas insuficientes para um mapa de persistência.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"depth": depth, "sea": sea, "d_inlet": np.full(shape, 1e4, np.float32)}
    years = np.array([r["year"] for r in used])
    for band in ("B03", "B04", "B08"):
        stack = np.stack([r[band] for r in used])  # float16
        if args.save_stack:
            np.save(OUT_DIR / f"{args.name}_stack_{band}.npy", stack)
        st = stack.astype(np.float32)
        with np.errstate(all="ignore"):
            out[f"ALL_med_{band}"] = np.nanmedian(st, axis=0).astype(np.float32)
            if band == "B03":
                out["ALL_A_B03"] = np.nanmedian(st[years % 2 == 1], axis=0).astype(np.float32)
                out["ALL_B_B03"] = np.nanmedian(st[years % 2 == 0], axis=0).astype(np.float32)
                out["ALL_frac_dark"] = (np.nansum(st <= -2.0, axis=0) / np.maximum(np.sum(np.isfinite(st), axis=0), 1)).astype(np.float32)
                out["ALL_n_valid"] = np.sum(np.isfinite(st), axis=0).astype(np.int16)
        del st, stack
    if args.save_stack:
        with open(OUT_DIR / f"{args.name}_stack_passes.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["idx", "id", "date", "orbit", "valid_frac", "deep_noise", "deglint_k"])
            for i, r in enumerate(used):
                w.writerow([i, r["id"], r["date"], r["orbit"], round(r["valid_frac"], 4), round(r["deep_noise"], 6), round(r["deglint_k"], 4)])
    T = grid["transform"]
    meta = {"name": args.name, "version": VERSION, "uv_zone": args.uv_zone, "bbox_wgs84": list(bbox), "crs": str(grid["crs"]),
            "x0": T.c, "y1": T.f, "res": T.a, "date_range": args.date_range, "n_passes": len(used),
            "orbits": sorted({str(r["orbit"]) for r in used}), "passes": [r["id"] for r in used],
            "params": {k: v for k, v in vars(args).items() if k not in ("emodnet",)}, "emodnet": str(emod_path)}
    np.savez_compressed(OUT_DIR / f"{args.name}_maps.npz", **out)
    (OUT_DIR / f"{args.name}_maps.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info(f"Gravado {OUT_DIR / (args.name + '_maps.npz')} ({len(used)} passagens, órbitas {meta['orbits']})")


if __name__ == "__main__":
    main()
