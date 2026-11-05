#!/usr/bin/env python3
"""
coastal_features_vm.py — Extract Algarve coastal-topography features from DGT
50 cm LiDAR by STREAMING directly from the DGT MinIO bucket (/vsis3/), windowed,
NO downloads.  Designed to run INSIDE the DGT JupyterHub VM (the S3 bucket policy
is IP-restricted to the VM network).

Run via:  python scripts/jupyterhub_control.py run-code "$(cat scripts/coastal_features_vm.py)"

It prints each site's stats as a `CSV::` line (so the caller can grep them out)
and also writes `coastal_features_vm.csv` in the VM home dir.  Slope/aspect use
the exact gradient formula from src/coastal_topography.py for consistency.
"""
import os, time

# --- map the VM's AWS_*2 creds to the names GDAL/rasterio expect ---
_ep = os.environ["AWS_ENDPOINT_URL2"].replace("https://", "").replace("http://", "")
os.environ["AWS_S3_ENDPOINT"] = _ep
os.environ["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY_ID2"]
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_ACCESS_KEY2"]
os.environ["AWS_HTTPS"] = "YES"
os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"

import numpy as np
import requests
import boto3
import rasterio
from rasterio.merge import merge
from pyproj import Transformer

BUCKET = os.environ.get("AWS_S3_BUCKET", "lidar")
STAC = "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items"
BUFFER_M = 3000        # feature-stats radius; reaches the emerged coast for offshore points
ANALYSIS_RES = 1.0     # metres; decimate 50cm→1m to cap RAM (still 30x finer than GLO-30)
SEARCH_HALF_DEG = 0.05 # ~5 km STAC search box around each site

_s3 = boto3.client(
    "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL2"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID2"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY2"],
)
_to3763 = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
_to4326 = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)

SITES = [
    ("tavira_west", 37.1050, -7.6800), ("ilha_de_tavira", 37.0950, -7.7200),
    ("fuseta", 37.0600, -7.7600), ("olhao_offshore", 37.0350, -7.8200),
    ("faro_east", 37.0100, -7.8800), ("praia_de_faro", 36.9800, -7.9400),
    ("ancão_peninsula", 36.9650, -8.0000), ("quarteira", 37.0650, -8.1000),
    ("vilamoura", 37.0750, -8.1300), ("olhos_de_agua", 37.0900, -8.1600),
    ("pedra_sta_eulalia", 37.069081, -8.210242), ("albufeira_reef", 37.0690, -8.2105),
    ("galé", 37.0560, -8.2296), ("salgados", 37.0950, -8.3000),
    ("armacao_de_pera", 37.0700, -8.3600),
]

COLS = ["site_name", "latitude", "longitude", "buffer_m",
        "slope_min", "slope_max", "slope_mean", "slope_std", "slope_median", "slope_percentile_90",
        "aspect_min", "aspect_max", "aspect_mean", "aspect_std", "aspect_median"]


def s3_keys_for(lat, lon, half_deg=SEARCH_HALF_DEG):
    """STAC-query tiles intersecting the site, resolve to exact S3 keys (any version)."""
    bbox = f"{lon-half_deg},{lat-half_deg},{lon+half_deg},{lat+half_deg}"
    r = requests.get(STAC, params={"bbox": bbox, "limit": 400, "f": "json"}, timeout=60)
    ids = [f["id"] for f in r.json().get("features", [])]
    keys = []
    for iid in ids:
        rr = _s3.list_objects_v2(Bucket=BUCKET, Prefix=f"MDT50cm/{iid}")
        keys += [o["Key"] for o in rr.get("Contents", []) if o["Key"].endswith(".tif")]
    return sorted(set(keys))


def find_coast_anchor(lat, lon, search_m=8000, res=4.0, min_elev=0.5):
    """For an offshore site, find the nearest emerged-land pixel (the coast anchor).

    Streams a coarse DEM over a large box, masks solid land (elev > min_elev),
    and returns (anchor_lat, anchor_lon, distance_m) of the closest land pixel
    to the site — or None if no land is found within the search box.
    """
    keys = s3_keys_for(lat, lon, half_deg=0.10)
    if not keys:
        return None
    x, y = _to3763.transform(lon, lat)
    bounds = (x - search_m, y - search_m, x + search_m, y + search_m)
    srcs = []
    try:
        for k in keys:
            try:
                srcs.append(rasterio.open(f"/vsis3/{BUCKET}/{k}"))
            except Exception:
                pass
        if not srcs:
            return None
        mosaic, tr = merge(srcs, bounds=bounds, res=res, nodata=-999.0)
    finally:
        for s in srcs:
            s.close()
    dem = mosaic[0].astype("float32")
    land = np.isfinite(dem) & (dem > min_elev) & (dem < 200)
    if not land.any():
        return None
    rr, cc = np.where(land)
    xs = tr.c + (cc + 0.5) * tr.a
    ys = tr.f + (rr + 0.5) * tr.e
    d = np.hypot(xs - x, ys - y)
    i = int(np.argmin(d))
    alon, alat = _to4326.transform(float(xs[i]), float(ys[i]))
    return alat, alon, float(d[i])


def stats_for(lat, lon):
    keys = s3_keys_for(lat, lon)
    if not keys:
        return None
    x, y = _to3763.transform(lon, lat)
    bounds = (x - BUFFER_M, y - BUFFER_M, x + BUFFER_M, y + BUFFER_M)
    srcs = []
    try:
        for k in keys:
            try:
                srcs.append(rasterio.open(f"/vsis3/{BUCKET}/{k}"))
            except Exception:
                pass
        if not srcs:
            return None
        mosaic, transform = merge(srcs, bounds=bounds, res=ANALYSIS_RES, nodata=-999.0)
    finally:
        for s in srcs:
            s.close()

    dem = mosaic[0].astype("float32")
    dem[dem <= -100] = np.nan
    if not np.isfinite(dem).any():
        return None

    # exact formula from src/coastal_topography.py
    dz_dy, dz_dx = np.gradient(dem, ANALYSIS_RES, ANALYSIS_RES)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.degrees(np.arctan2(dz_dx, -dz_dy))
    aspect = np.where(aspect < 0, 360 + aspect, aspect)

    # circular buffer mask centred on the site
    rows, cols = dem.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    col0 = (x - transform.c) / transform.a
    row0 = (y - transform.f) / transform.e
    dist = np.hypot((xx - col0) * ANALYSIS_RES, (yy - row0) * ANALYSIS_RES)
    m = (dist <= BUFFER_M) & np.isfinite(slope)
    if int(m.sum()) < 50:
        return None
    sl, asp = slope[m], aspect[m]
    return {
        "slope_min": float(sl.min()), "slope_max": float(sl.max()), "slope_mean": float(sl.mean()),
        "slope_std": float(sl.std()), "slope_median": float(np.median(sl)),
        "slope_percentile_90": float(np.percentile(sl, 90)),
        "aspect_min": float(asp.min()), "aspect_max": float(asp.max()), "aspect_mean": float(asp.mean()),
        "aspect_std": float(asp.std()), "aspect_median": float(np.median(asp)),
        "_valid_px": int(m.sum()),
    }


def _degenerate(st):
    """True if the buffer caught too little land or flat constant data (offshore)."""
    return (st is None
            or st.get("_valid_px", 0) < 1_000_000
            or st.get("slope_max", 0.0) < 1.0)


def main():
    print("CSV::" + ",".join(COLS))
    out_rows = [",".join(COLS)]
    n_ok = 0
    for i, (name, lat, lon) in enumerate(SITES, 1):
        t0 = time.time()
        try:
            st = stats_for(lat, lon)
        except Exception as e:
            print(f"ERR {name}: {type(e).__name__}: {str(e)[:120]}")
            st = None

        note = ""
        if _degenerate(st):
            # Offshore reef point — snap to the nearest emerged coast and retry.
            try:
                anc = find_coast_anchor(lat, lon)
            except Exception as e:
                print(f"  snap-err {name}: {type(e).__name__}: {str(e)[:90]}")
                anc = None
            if anc:
                alat, alon, dist = anc
                try:
                    st2 = stats_for(alat, alon)
                except Exception:
                    st2 = None
                if not _degenerate(st2):
                    st = st2
                    note = f" [snapped {dist/1000:.1f}km to {alat:.4f},{alon:.4f}]"

        if _degenerate(st):
            print(f"  [{i}/15] {name}: NO USABLE LAND (even after snap)")
            continue

        # Keep the original reef coords; features describe the nearest coast.
        row = [name, f"{lat}", f"{lon}", f"{BUFFER_M}"] + [f"{st[c]:.6f}" for c in COLS[4:]]
        line = ",".join(row)
        out_rows.append(line)
        n_ok += 1
        print("CSV::" + line)
        print(f"  [{i}/15] {name}: slope_mean={st['slope_mean']:.1f} "
              f"slope_max={st['slope_max']:.1f} aspect_mean={st['aspect_mean']:.1f} "
              f"px={st['_valid_px']}{note} ({time.time()-t0:.1f}s)")
    with open(os.path.expanduser("~/coastal_features_vm.csv"), "w") as fh:
        fh.write("\n".join(out_rows) + "\n")
    print(f"DONE — {n_ok}/{len(SITES)} sites written to ~/coastal_features_vm.csv")


main()
