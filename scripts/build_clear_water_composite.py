#!/usr/bin/env python3
"""
scripts/build_clear_water_composite.py — Composição mediana multi-temporal de dias com água límpida.

Uma cena isolada escolhida só por nebulosidade traz sunglint, esteiras, ondulação e turbidez que
o detetor confunde com relevo do fundo. O fundo rochoso é estático; os artefactos de superfície não.
Este script (regra 1 do AGENTS.md):

  1. Procura cenas Sentinel-2 L2A (Planetary Computer) sobre a grelha da cena de amostra da zona.
  2. Exige que a cena cubra toda a grelha (descarta tiles vizinhos com cobertura parcial) e, entre
     reprocessamentos da mesma aquisição, mantém o mais recente.
  3. Classifica cada cena pela perturbação da superfície: percentil 90 do NIR (B08) sobre a água
     (esteiras, espuma e ondulação estão na cauda alta; a mediana satura em água escura) e
     fração de água válida pela SCL (sem nuvem/sombra).
  4. Mantém as ``--n-best`` cenas mais límpidas e calcula a mediana por pixel de B02/B03/B04/B08,
     mascarando em cada cena nuvens/sombras (SCL) e pixels com NIR acima de ``--glint-nir``.

Saída: data/raw/<zona>_composite.tif, com a mesma grelha, bandas e escala da amostra
(reflectância BOA 0-1, nodata 0), e as cenas usadas nas tags ``scenes``.

Uso:
    python scripts/build_clear_water_composite.py --zone santa_eulalia
    python scripts/build_clear_water_composite.py --zone all --n-best 12
    python scripts/build_clear_water_composite.py --bbox -8.175 36.965 -7.995 37.085 --name quarteira_garrao
"""

from __future__ import annotations

import argparse
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
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clear_water_composite")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

ZONE_SAMPLES = {
    "valado": "pedra_do_valado",
    "greta": "pedra_da_greta",
    "santa_eulalia": "santa_eulalia",
    "quarteira": "quarteira",
}

BANDS = ["B02", "B03", "B04", "B08"]
BAND_DESCRIPTIONS = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"]

# SCL: 3 sombra de nuvem, 8/9 nuvem média/alta, 10 cirros, 11 neve
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)

_READ_FAILED = object()  # sentinela: leitura de rede falhou (distinto de "cena não elegível")


def composite_path(sample_name: str) -> Path:
    return RAW_DIR / f"{sample_name}_composite.tif"


def _boa_offset_dn(item) -> float:
    """Baseline >= 04.00 (a partir de 2022-01-25) acrescenta BOA_ADD_OFFSET = -1000 DN."""
    baseline = item.properties.get("s2:processing_baseline")
    try:
        return 1000.0 if float(baseline) >= 4.0 else 0.0
    except (TypeError, ValueError):
        return 1000.0 if item.datetime.year >= 2022 else 0.0


def _read_on_grid(href: str, grid: dict, resampling: Resampling) -> np.ndarray:
    with rasterio.open(href) as src, WarpedVRT(
        src,
        crs=grid["crs"],
        transform=grid["transform"],
        width=grid["width"],
        height=grid["height"],
        resampling=resampling,
        nodata=0,
    ) as vrt:
        return vrt.read(1).astype(np.float32)


def _to_reflectance(dn: np.ndarray, offset: float) -> np.ndarray:
    return np.where(dn > 0, np.maximum((dn - offset) * 0.0001, 0.0001), 0.0).astype(np.float32)


def _score_scene(item, grid: dict, glint_nir: float, min_coverage: float = 0.98) -> dict | None:
    """Mede a limpidez da superfície da água numa cena, sem ler as bandas que não são precisas."""
    try:
        offset = _boa_offset_dn(item)
        b03 = _to_reflectance(_read_on_grid(item.assets["B03"].href, grid, Resampling.bilinear), offset)
        b08 = _to_reflectance(_read_on_grid(item.assets["B08"].href, grid, Resampling.bilinear), offset)
        scl = _read_on_grid(item.assets["SCL"].href, grid, Resampling.nearest)
    except Exception as exc:  # cena inacessível ou corrompida: ignora
        logger.warning(f"{item.id}: leitura falhou ({exc}); cena ignorada.")
        return _READ_FAILED

    coverage = float(np.mean(b03 > 0))
    if coverage < min_coverage:
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (b03 - b08) / (b03 + b08)
    water = (ndwi > 0.0) & (b03 > 0)
    if water.sum() < 1000:
        return None

    valid_water = water & ~np.isin(scl, SCL_INVALID)
    return {
        "item": item,
        "date": item.datetime.strftime("%Y-%m-%d"),
        "valid_water_frac": float(valid_water.sum() / water.sum()),
        "coverage": coverage,
        "nir_water_median": float(np.median(b08[valid_water])) if valid_water.any() else np.inf,
        "nir_water_p90": float(np.percentile(b08[valid_water], 90)) if valid_water.any() else np.inf,
        "glint_frac": float(np.mean(b08[valid_water] > glint_nir)) if valid_water.any() else 1.0,
    }


def _latest_processing_per_acquisition(items: list) -> list:
    """Uma aquisição pode estar publicada várias vezes (reprocessamentos); fica a mais recente."""
    latest: dict[tuple, object] = {}
    for it in items:
        key = (it.datetime, it.properties.get("s2:mgrs_tile"))
        if key not in latest or it.id > latest[key].id:  # o sufixo do id é a data de processamento
            latest[key] = it
    return list(latest.values())


def _read_scene_stack(item, grid: dict, glint_nir: float) -> np.ndarray:
    """Lê B02/B03/B04/B08 e devolve (4, H, W) com NaN onde há nuvem, sombra ou glint."""
    offset = _boa_offset_dn(item)
    stack = np.stack(
        [_to_reflectance(_read_on_grid(item.assets[b].href, grid, Resampling.bilinear), offset) for b in BANDS]
    )
    scl = _read_on_grid(item.assets["SCL"].href, grid, Resampling.nearest)
    bad = np.isin(scl, SCL_INVALID) | (stack[3] > glint_nir) | (stack[0] <= 0)
    stack[:, bad] = np.nan
    return stack


def _grid_from_bbox(bbox: tuple[float, float, float, float], res: float = 10.0) -> tuple[dict, dict]:
    """Grelha UTM 29N de 10 m, alinhada a múltiplos de 10 m (como os tiles Sentinel-2), para uma bbox WGS84."""
    from rasterio.transform import from_origin

    crs = rasterio.crs.CRS.from_epsg(32629)
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox)
    left, top = np.floor(left / res) * res, np.ceil(top / res) * res
    width, height = int(np.ceil((right - left) / res)), int(np.ceil((top - bottom) / res))
    transform = from_origin(left, top, res, res)
    grid = {"crs": crs, "transform": transform, "width": width, "height": height}
    profile = {"driver": "GTiff", "crs": crs, "transform": transform, "width": width, "height": height}
    return grid, profile


def build_composite(
    zone: str,
    date_range: str,
    max_cloud: float,
    n_best: int,
    glint_nir: float,
    min_valid_water: float,
    workers: int,
    max_failed_frac: float = 0.05,
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    if zone in ZONE_SAMPLES:
        sample_name = ZONE_SAMPLES[zone]
        with rasterio.open(RAW_DIR / f"{sample_name}_sample.tif") as s:
            grid = {"crs": s.crs, "transform": s.transform, "width": s.width, "height": s.height}
            profile = s.profile.copy()
            bbox_wgs84 = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    else:
        sample_name = zone
        if bbox is None:
            raise ValueError(f"Zona '{zone}' sem cena de amostra: indique --bbox.")
        grid, profile = _grid_from_bbox(bbox)
        bbox_wgs84 = bbox

    client = Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = list(
        client.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox_wgs84),
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=400,
        ).items()
    )
    logger.info(f"[{zone}] {len(items)} cenas com nuvens < {max_cloud}% em {date_range}.")
    items = _latest_processing_per_acquisition(items)
    logger.info(f"[{zone}] {len(items)} aquisições únicas (tile + data) após remover reprocessamentos.")
    if not items:
        raise RuntimeError(f"Nenhuma cena encontrada para {zone}.")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda it: _score_scene(it, grid, glint_nir), items))
    n_failed = sum(r is _READ_FAILED for r in results)
    if n_failed > max_failed_frac * len(items):
        # Com muitas leituras falhadas a escolha das "melhores" cenas é feita sobre uma lista truncada
        raise RuntimeError(f"[{zone}] {n_failed}/{len(items)} leituras falharam (rede?); composição não gerada.")
    scored = [r for r in results if r is not None and r is not _READ_FAILED]

    eligible = [s for s in scored if s["valid_water_frac"] >= min_valid_water]
    eligible.sort(key=lambda s: (s["nir_water_p90"], s["glint_frac"]))
    chosen = eligible[:n_best]
    if len(chosen) < 3:
        raise RuntimeError(
            f"[{zone}] só {len(chosen)} cenas límpidas (>= {min_valid_water:.0%} de água válida); "
            "alargue --date-range ou --max-cloud."
        )
    for s in chosen:
        logger.info(
            f"[{zone}] {s['date']} {s['item'].id}: NIR água p50={s['nir_water_median']:.4f} p90={s['nir_water_p90']:.4f} "
            f"glint={s['glint_frac']:.1%} água válida={s['valid_water_frac']:.1%}"
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        stacks = list(pool.map(lambda s: _read_scene_stack(s["item"], grid, glint_nir), chosen))

    with np.errstate(all="ignore"):
        median = np.nanmedian(np.stack(stacks), axis=0)
        n_obs = np.sum(np.isfinite(np.stack(stacks)[:, 0]), axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)

    out = composite_path(sample_name)
    profile.update(driver="GTiff", count=len(BANDS), dtype=rasterio.float32, nodata=0, compress="deflate")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(median)
        for i, desc in enumerate(BAND_DESCRIPTIONS, start=1):
            dst.set_band_description(i, desc)
        dst.update_tags(
            scenes=json.dumps(
                [{"id": s["item"].id, "date": s["date"], "nir_water_p90": s["nir_water_p90"]} for s in chosen]
            ),
            n_scenes=str(len(chosen)),
            glint_nir=str(glint_nir),
            min_obs_per_pixel=str(int(n_obs.min())),
        )
    logger.info(f"[{zone}] Composição de {len(chosen)} cenas gravada em {out} (obs/pixel mediana: {int(np.median(n_obs))}).")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Composição mediana Sentinel-2 de dias com água límpida")
    parser.add_argument("--zone", choices=[*ZONE_SAMPLES, "all"], default="all")
    parser.add_argument(
        "--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
        help="Zona nova sem cena de amostra: bbox WGS84 (grelha UTM 29N de 10 m). Requer --name.",
    )
    parser.add_argument("--name", help="Nome da zona nova (ficheiro data/raw/<nome>_composite.tif).")
    parser.add_argument("--date-range", default="2019-01-01/2025-12-31")
    parser.add_argument("--max-cloud", type=float, default=10.0)
    parser.add_argument("--n-best", type=int, default=10, help="Número de cenas mais límpidas na mediana.")
    parser.add_argument("--glint-nir", type=float, default=0.03, help="Reflectância B08 acima da qual um pixel de água é glint/esteira.")
    parser.add_argument("--min-valid-water", type=float, default=0.90, help="Fração mínima de água sem nuvem/sombra por cena.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.bbox:
        if not args.name:
            parser.error("--bbox requer --name")
        zones = [args.name]
    else:
        zones = list(ZONE_SAMPLES) if args.zone == "all" else [args.zone]
    failed = []
    for zone in zones:
        try:
            build_composite(
                zone, args.date_range, args.max_cloud, args.n_best, args.glint_nir, args.min_valid_water, args.workers,
                bbox=tuple(args.bbox) if args.bbox else None,
            )
        except Exception as exc:
            logger.error(f"[{zone}] composição falhou: {exc}", exc_info=True)
            failed.append(zone)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
