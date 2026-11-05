#!/usr/bin/env python3
"""
demo_bathy_live.py
Demonstração ao vivo da integração IH Isobatas + SDB Stumpf
com os TIFs reais do projeto (reef_Output_Master).
"""
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout
)

ROOT = Path(__file__).parent.parent  # scripts/ -> raiz
MASTER = ROOT / "reef_Output_Master"

# ── spots a testar ──────────────────────────────────────────────────────────
SPOTS = [
    {
        "name":  "Pedra do Alto (2024-09-30)",
        "dir":   MASTER / "reef_output_ai_prediction_spot",
        "b02":   "S2_B02_20240930.tif",
        "b03":   "S2_B03_20240930.tif",
        "lat":   37.0636,
        "lon":  -8.2193,
    },
    {
        "name":  "Mar 2022 New Spot (2022-02-28)",
        "dir":   MASTER / "reef_output_mar_2022_new_spot",
        "b02":   "S2_B02_20220228.tif",
        "b03":   "S2_B03_20220228.tif",
        "lat":   37.0581,
        "lon":  -8.2098,
    },
    {
        "name":  "Aug 2022 Target (2022-08-12)",
        "dir":   MASTER / "reef_output_aug_2022_target1",
        "b02":   "S2_B02_20220812.tif",
        "b03":   "S2_B03_20220812.tif",
        "lat":   37.0636,
        "lon":  -8.2193,
    },
]

BANNER = "=" * 66


def load_raster_wgs84(tif_path):
    """Load raster as float32 [0..1] and return (array, profile, wgs84_bounds)."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        crs = src.crs

        # Convert raster bounds to WGS84
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326",
            src.bounds.left, src.bounds.bottom,
            src.bounds.right, src.bounds.top
        )

    # Normalise to BOA reflectance [0..1]
    if arr.max() > 2.0:
        arr = arr / 10000.0

    # bounds_wgs84 in the format bathy_calibrator expects: (min_lat, min_lon, max_lat, max_lon)
    bounds_wgs84 = (south, west, north, east)
    return arr, profile, bounds_wgs84


def run_spot(spot):
    from src.bathy_calibrator import (
        fetch_isobaths_for_bbox,
        classify_benthic_zone,
        calibrate_stumpf_from_isobaths,
        validate_sdb_vs_chart,
    )
    from src.reef_ml_predictor_acolite import stumpf_sdb

    name = spot["name"]
    lat, lon = spot["lat"], spot["lon"]
    b02_path = spot["dir"] / spot["b02"]
    b03_path = spot["dir"] / spot["b03"]

    if not b02_path.exists() or not b03_path.exists():
        print(f"\n⚠️  Ficheiros não encontrados: {b02_path}")
        return

    print(f"\n{BANNER}")
    print(f"  {name}")
    print(f"  Lat: {lat}  Lon: {lon}")
    print(BANNER)

    # ── 1. Carregar rasters ──────────────────────────────────────────────────
    b02, profile, bounds = load_raster_wgs84(b02_path)
    b03, _,       _      = load_raster_wgs84(b03_path)
    min_lat, min_lon, max_lat, max_lon = bounds

    print("\n📦 Raster carregado")
    print(f"   Shape    : {b02.shape}")
    print(f"   BOA B02  : min={b02[b02>0].min():.4f}  max={b02.max():.4f}  "
          f"mean={b02[b02>0].mean():.4f}")
    print(f"   Bounds   : lon [{min_lon:.4f} → {max_lon:.4f}]  "
          f"lat [{min_lat:.4f} → {max_lat:.4f}]")

    # ── 2. Buscar isóbatas IH ────────────────────────────────────────────────
    print("\n🌊 A consultar serviço IH DGRM ...")
    DEG = 3000 / 111_000
    features = fetch_isobaths_for_bbox(
        min_lon=lon - DEG, min_lat=lat - DEG,
        max_lon=lon + DEG, max_lat=lat + DEG,
    )
    depths_found = sorted(set(f["depth"] for f in features))
    print(f"   Isóbatas disponíveis na área : {depths_found} m")
    print(f"   Total de segmentos           : {len(features)}")

    # ── 3. Classificação de zona ─────────────────────────────────────────────
    zone = classify_benthic_zone(lon, lat, features)
    print("\n📍 Classificação de Zona Bentónica")
    print(f"   Zona              : {zone['zone']}")
    print(f"   Isóbata mais próx.: {zone['nearest_isobath']}")
    print(f"   Dist. → 10m       : {zone['dist_10m_m']} m")
    print(f"   Dist. → 20m       : {zone['dist_20m_m']} m")
    print(f"   Dist. → 30m       : {zone['dist_30m_m']} m")
    print(f"   Viável Sentinel-2 : {'✅ Sim' if zone['optically_viable'] else '❌ Não'}")
    print(f"   Nota              : {zone['note']}")

    # ── 4. Calibração Stumpf ─────────────────────────────────────────────────
    print("\n🔧 Calibração Stumpf SDB com isóbatas IH ...")
    m0, m1, cal = calibrate_stumpf_from_isobaths(b02, b03, features, bounds)
    if cal["calibrated"]:
        print(f"   ✅ Calibrado com {cal['n_samples']} amostras")
        print(f"   m0 = {m0:.3f}  (default: -16.0)")
        print(f"   m1 = {m1:.3f}  (default:  20.0)")
        print(f"   RMSE calibração = {cal['rmse_m']} m")
        print(f"   Isóbatas usadas = {cal['isobaths_used']} m")
    else:
        print(f"   ⚠️  Não calibrado ({cal['n_samples']} amostras) — "
              f"a usar defaults m0={m0:.1f}  m1={m1:.1f}")
        print("   Motivo: poucos pixels próximos das isóbatas no raster")

    # ── 5. Gerar mapa SDB ────────────────────────────────────────────────────
    print("\n🗺️  A gerar mapa SDB (Stumpf log-ratio) ...")
    sdb = stumpf_sdb(b02, b03, m0=m0, m1=m1)
    valid = sdb[sdb > 0]
    print(f"   Pixels válidos : {len(valid):,} / {sdb.size:,} "
          f"({100*len(valid)/sdb.size:.1f}%)")
    print(f"   Profundidade   : min={valid.min():.1f}m  "
          f"max={valid.max():.1f}m  média={valid.mean():.1f}m  "
          f"mediana={np.median(valid):.1f}m")

    # Distribuição por classes de profundidade
    bins = [(0,5,"0–5m"), (5,10,"5–10m"), (10,20,"10–20m"),
            (20,30,"20–30m"), (30,40,"30–40m")]
    print("\n   Distribuição de profundidade estimada:")
    for lo, hi, label in bins:
        n = ((sdb >= lo) & (sdb < hi)).sum()
        pct = 100 * n / max(1, len(valid))
        bar = "█" * int(pct / 2)
        print(f"   {label:8s}  {bar:<25s}  {pct:5.1f}%  ({n:,} px)")

    # ── 6. Validação SDB vs carta náutica ────────────────────────────────────
    print("\n📐 Validação SDB vs carta náutica IH ...")
    val = validate_sdb_vs_chart(sdb, features, bounds)
    per = val.get("per_isobath", {})
    ov  = val.get("overall", {})

    if per:
        print(f"   {'Isóbata':10s} {'N amostras':>12s} {'Bias (m)':>10s} "
              f"{'RMSE (m)':>10s} {'MAE (m)':>10s}")
        print(f"   {'-'*56}")
        for iso, stats in sorted(per.items()):
            if stats["n_samples"] > 0:
                print(f"   {iso:10s} {stats['n_samples']:>12d} "
                      f"{stats['bias_m']:>10.2f} {stats['rmse_m']:>10.2f} "
                      f"{stats['mae_m']:>10.2f}")
            else:
                print(f"   {iso:10s} {'—':>12s} {'—':>10s} {'—':>10s} {'—':>10s}")

    if ov:
        print(f"\n   {'GLOBAL':10s} {ov['n_total']:>12d} "
              f"{ov['overall_bias_m']:>10.2f} {ov['overall_rmse_m']:>10.2f} "
              f"{ov['overall_mae_m']:>10.2f}")
        bias = ov['overall_bias_m']
        interp = ("subestima" if bias > 0 else "sobrestima") + \
                 f" profundidade em {abs(bias):.2f}m em média"
        print(f"\n   📊 Interpretação: O SDB {interp}")
    else:
        print("   ⚠️  Sem sobreposição entre mapa SDB e isóbatas → "
              "bbox muito pequeno ou spot offshore")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*66}")
    print("  DEMO: IH Isobatas + Calibração Stumpf SDB + Validação")
    print("  Serviço: DGRM/IH ArcGIS REST  —  Escala 1:150.000")
    print(f"{'='*66}")

    for spot in SPOTS:
        try:
            run_spot(spot)
        except Exception as e:
            print(f"\n❌ Erro em {spot['name']}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*66}")
    print("  Demo concluído.")
    print(f"{'='*66}\n")
