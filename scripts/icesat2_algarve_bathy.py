#!/usr/bin/env python3
"""
icesat2_algarve_bathy.py
═══════════════════════════════════════════════════════════════════════════════
Search ICESat-2 bathymetric tracks along the Algarve coast (0-16m depth zone)
Sites: Pedra do Alto, Armação de Atuns, Pedra Sta Eulália + extended coast

Uses NASA EarthData (earthaccess) to search ATL03 granules.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime

try:
    import earthaccess
except ImportError:
    print("Installing earthaccess...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "earthaccess", "-q"])
    import earthaccess

# ═══════════════════════════════════════════════════════════════════════════
# ALGARVE COAST BOUNDING BOX (expanded to catch all tracks)
# From Sagres (west) to Vila Real de Santo António (east)
# Only the shallow zone (coast to ~16m depth = ~1-2km offshore)
# ═══════════════════════════════════════════════════════════════════════════

# Algarve south coast — approximate bounding box
# West: Sagres ~8.95°W, East: VRSA ~7.4°W
# North: coast ~37.08°N, South: ~1.5km offshore ~37.02°N (covers 16m isobath)

BBOX_WEST = -8.95   # Sagres
BBOX_EAST = -7.40   # Vila Real Sto António
BBOX_SOUTH = 37.00  # offshore (~2km from coast)
BBOX_NORTH = 37.10  # onshore (includes coast)

# Specific dive sites
SITES = {
    "Pedra do Alto":      {"lat": 37.05815, "lon": -8.20982, "depth": 16},  # 37°03.489'N 008°12.589'W
    "Armação de Atuns":   {"lat": 37.04678, "lon": -7.66038, "depth": 10},
    "Pedra Sta Eulália":  {"lat": 37.068978, "lon": -8.210328, "depth": 12},
}

print("="*80)
print("  ICESat-2 BATIMETRIA LIDAR — Costa Algarvia (0-16m)")
print(f"  Bounding Box: {BBOX_WEST}°W to {BBOX_EAST}°W, {BBOX_SOUTH}°N to {BBOX_NORTH}°N")
print("  Period: 2018-10-01 to present")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Login to NASA EarthData (anonymous search OK)
# ═══════════════════════════════════════════════════════════════════════════

print("\n[1] Connecting to NASA EarthData...")
try:
    earthaccess.login(strategy="environment")
    print("    ✓ Authenticated via environment")
except Exception:
    try:
        earthaccess.login(strategy="interactive")
        print("    ✓ Authenticated interactively")
    except Exception:
        print("    ⚠ Running without authentication (search only)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Search ATL03 (Geolocated Photon Data) — raw photons
# ═══════════════════════════════════════════════════════════════════════════

print("\n[2] Searching ICESat-2 ATL03 granules over Algarve coast...")
print("    Product: ATL03 (Global Geolocated Photon Data)")
print("    This contains raw photon returns including seafloor reflections")

try:
    results_atl03 = earthaccess.search_data(
        short_name="ATL03",
        bounding_box=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        temporal=("2018-10-01", datetime.now().strftime("%Y-%m-%d")),
        count=500,
    )
    print(f"    ✓ Found {len(results_atl03)} ATL03 granules")
except Exception as e:
    print(f"    ✗ ATL03 search failed: {e}")
    results_atl03 = []

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Search ATL24 (Bathymetric Profiles) — processed bathymetry
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3] Searching ICESat-2 ATL24 (Inland Water Bathymetry)...")
print("    This is the NEW processed bathymetry product (released 2024)")

try:
    results_atl24 = earthaccess.search_data(
        short_name="ATL24",
        bounding_box=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        temporal=("2018-10-01", datetime.now().strftime("%Y-%m-%d")),
        count=500,
    )
    print(f"    ✓ Found {len(results_atl24)} ATL24 granules")
except Exception as e:
    print(f"    ⚠ ATL24 search: {e}")
    results_atl24 = []

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Search ATL12 (Ocean Surface Height) — for reference
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4] Searching ICESat-2 ATL12 (Ocean Surface Height)...")

try:
    results_atl12 = earthaccess.search_data(
        short_name="ATL12",
        bounding_box=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        temporal=("2018-10-01", datetime.now().strftime("%Y-%m-%d")),
        count=500,
    )
    print(f"    ✓ Found {len(results_atl12)} ATL12 granules")
except Exception as e:
    print(f"    ⚠ ATL12 search: {e}")
    results_atl12 = []

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Analyze granule dates and proximity to dive sites
# ═══════════════════════════════════════════════════════════════════════════

print("\n[5] Analyzing temporal coverage...")

def extract_dates(results, product_name):
    """Extract dates from granules."""
    dates = []
    for r in results:
        try:
            # earthaccess granule metadata
            umm = r.get("umm", r) if isinstance(r, dict) else r
            # Try different metadata access patterns
            if hasattr(r, 'data_links'):
                # earthaccess DataGranule object
                t_start = r.get("umm", {}).get("TemporalExtent", {}).get(
                    "RangeDateTime", {}).get("BeginningDateTime", "")
                if not t_start:
                    # Try properties
                    t_start = str(getattr(r, 'properties', {}).get('start_datetime', ''))
            else:
                t_start = str(r)
            
            if "20" in str(t_start)[:10]:
                date_str = str(t_start)[:10]
                dates.append(date_str)
        except Exception:
            continue
    return sorted(set(dates))

# Get dates for ATL03
atl03_info = []
for r in results_atl03:
    try:
        # Get temporal info
        props = {}
        if hasattr(r, '__getitem__'):
            props = r.get('umm', {}).get('TemporalExtent', {}).get('RangeDateTime', {})
        
        # Try to get the granule ID for date extraction
        granule_id = str(r) if not hasattr(r, 'native_id') else r.native_id
        # Extract date from granule ID (format: ATL03_YYYYMMDDHHMMSS_...)
        parts = granule_id.split("_")
        date_str = None
        for p in parts:
            if len(p) >= 8 and p[:4].isdigit() and p[4:6].isdigit():
                try:
                    date_str = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
                    pd.Timestamp(date_str)
                    break
                except Exception:
                    date_str = None
        
        if date_str:
            atl03_info.append({"date": date_str, "granule": granule_id[:60]})
    except Exception:
        continue

if atl03_info:
    df_atl03 = pd.DataFrame(atl03_info).drop_duplicates("date").sort_values("date")
    print(f"\n  ATL03 — {len(df_atl03)} unique passes over Algarve coast:")
    print(f"  Period: {df_atl03['date'].iloc[0]} to {df_atl03['date'].iloc[-1]}")
    print("\n  Dates of ICESat-2 passes:")
    for i, row in df_atl03.iterrows():
        print(f"    • {row['date']}")
    
    # Group by year
    df_atl03["year"] = pd.to_datetime(df_atl03["date"]).dt.year
    yearly = df_atl03.groupby("year").size()
    print("\n  Passes per year:")
    for y, n in yearly.items():
        print(f"    {y}: {n} passes")
else:
    print("\n  Could not extract date information from granules")
    print(f"  Total granules found: ATL03={len(results_atl03)}")
    if results_atl03:
        print(f"  Sample granule info: {str(results_atl03[0])[:200]}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Summary and recommendations
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("  RESUMO — ICESat-2 sobre Costa Algarvia")
print(f"{'='*80}")

print(f"""
  Dados disponíveis:
    ATL03 (fotões brutos):     {len(results_atl03)} granules
    ATL24 (batimetria):        {len(results_atl24)} granules  
    ATL12 (superfície oceano): {len(results_atl12)} granules

  Zona de interesse:
    Costa: Sagres → Vila Real Sto António (~150km)
    Profundidade: 0–16m (zona fótica bentônica)
    
  Sites de mergulho:""")
for name, cfg in SITES.items():
    print(f"    • {name}: {cfg['lat']:.5f}°N, {abs(cfg['lon']):.5f}°W ({cfg['depth']}m)")

print("""
  Potencial para batimetria lidar:
    • Kd490 médio: 0.045–0.080 m⁻¹ (águas claras)
    • Profundidade máx. detetável: ~20–40m (em dias ótimos)
    • A 16m: ~50-70% dos fotões atingem o fundo
    
  Para aceder aos dados:
    1. https://openaltimetry.org → visualização rápida dos tracks
    2. NASA EarthData: https://search.earthdata.nasa.gov/
       → Procurar "ATL03" + bounding box Algarve
    3. Python: earthaccess.download(results) → ficheiros HDF5
    
  Nota: ICESat-2 dá PERFIS (linhas), não imagens.
  Útil para: validar batimetria derivada de Sentinel-2 (Stumpf SDB)
""")

print("="*80)
