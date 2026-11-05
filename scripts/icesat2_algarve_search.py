#!/usr/bin/env python3
"""
icesat2_algarve_search.py
═══════════════════════════════════════════════════════════════════════════════
Search ICESat-2 tracks over Algarve coast using NASA CMR API (no auth needed)
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import httpx

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

# Algarve south coast bounding box (shallow zone 0-16m)
BBOX_WEST = -8.95
BBOX_EAST = -7.40
BBOX_SOUTH = 37.00
BBOX_NORTH = 37.10

# Dive sites
SITES = {
    "Pedra do Alto":      {"lat": 37.05815, "lon": -8.20982, "depth": 16},  # 37°03.489'N 008°12.589'W
    "Armação de Atuns":   {"lat": 37.04678, "lon": -7.66038, "depth": 10},
    "Pedra Sta Eulália":  {"lat": 37.068978, "lon": -8.210328, "depth": 12},
}

# NASA CMR (Common Metadata Repository) — public, no auth
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

print("="*80)
print("  ICESat-2 BATIMETRIA LIDAR — Costa Algarvia (0-16m)")
print(f"  BBox: [{BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH}]")
print("  Período: 2018-10 → presente")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# SEARCH NASA CMR for ATL03 granules
# ═══════════════════════════════════════════════════════════════════════════

def search_cmr(short_name, page_size=200):
    """Search NASA CMR for ICESat-2 granules over Algarve."""
    params = {
        "short_name": short_name,
        "bounding_box": f"{BBOX_WEST},{BBOX_SOUTH},{BBOX_EAST},{BBOX_NORTH}",
        "temporal": "2018-10-01T00:00:00Z,2026-12-31T00:00:00Z",
        "page_size": page_size,
        "sort_key": "-start_date",
    }
    
    all_granules = []
    page = 1
    
    with httpx.Client(timeout=30) as client:
        while True:
            params["page_num"] = page
            resp = client.get(CMR_URL, params=params)
            if resp.status_code != 200:
                print(f"    ✗ CMR error: {resp.status_code}")
                break
            
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])
            if not entries:
                break
                
            all_granules.extend(entries)
            if len(entries) < page_size:
                break
            page += 1
    
    return all_granules

# Search ATL03 (raw photons — best for bathymetry)
print("\n[1] Pesquisando ATL03 (fotões brutos — batimetria)...")
granules_atl03 = search_cmr("ATL03")
print(f"    ✓ {len(granules_atl03)} granules ATL03 encontrados")

# Parse granule metadata
atl03_records = []
for g in granules_atl03:
    title = g.get("title", "")
    t_start = g.get("time_start", "")
    t_end = g.get("time_end", "")
    granule_id = g.get("producer_granule_id", g.get("id", title))
    
    # Extract spatial
    boxes = g.get("boxes", [])
    polygons = g.get("polygons", [])
    
    date_str = t_start[:10] if t_start else ""
    
    atl03_records.append({
        "date": date_str,
        "time": t_start[11:19] if len(t_start) > 11 else "",
        "granule_id": granule_id[:80],
        "boxes": boxes,
    })

# Search ATL12 (ocean surface)
print("\n[2] Pesquisando ATL12 (superfície oceano)...")
granules_atl12 = search_cmr("ATL12")
print(f"    ✓ {len(granules_atl12)} granules ATL12 encontrados")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("  RESULTADOS — ICESat-2 sobre Algarve")
print(f"{'='*80}")

if atl03_records:
    df = pd.DataFrame(atl03_records)
    df = df[df["date"] != ""].drop_duplicates("date").sort_values("date")
    
    print("\n  ATL03 — Passes únicas sobre costa Algarvia:")
    print(f"  Total: {len(df)} datas")
    print(f"  Período: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    
    # By year
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    
    print("\n  Passes por ano:")
    for y, grp in df.groupby("year"):
        print(f"    {y}: {len(grp)} passes")
    
    # Best months (need clear water = low CHL)
    print("\n  Passes por mês (melhores para batimetria = Set/Out/Jan):")
    for m in range(1, 13):
        n = (df["month"] == m).sum()
        best = "⭐" if m in [1, 9, 10, 12] else ""
        print(f"    Mês {m:>2}: {n:>2} passes {best}")
    
    # List all dates
    print("\n  Todas as datas com cobertura ICESat-2:")
    print(f"  {'Date':<12} {'Granule ID':<60}")
    print("  " + "-"*72)
    for _, r in df.iterrows():
        print(f"  {r['date']:<12} {r['granule_id']}")

    # Check proximity to dive sites
    print("\n  Notas sobre proximidade aos dive sites:")
    print("  ⚠️  ICESat-2 tem tracks com ~14m de largura e espaçamento ~3km")
    print("     A probabilidade de um track passar EXATAMENTE sobre um")
    print("     dive site é baixa. Os dados são úteis para:")
    print("     • Validar batimetria Sentinel-2 (Stumpf SDB)")
    print("     • Mapa de profundidade da zona costeira")
    print("     • Identificar formações rochosas submersas")
    
    # Cross-reference with visibility model
    # Best dates = ICESat-2 pass + clear water (Sep/Oct/Jan)
    good_months = df[df["month"].isin([1, 9, 10, 12])]
    print("\n  🎯 PASSES EM MESES DE ÁGUA CLARA (Set/Out/Jan/Dez):")
    print(f"     → {len(good_months)} passes com potencial batimétrico máximo")
    for _, r in good_months.iterrows():
        print(f"     • {r['date']}")

else:
    print("\n  Nenhum granule encontrado com os parâmetros de pesquisa.")
    print(f"  Granules ATL03: {len(granules_atl03)}")

# OpenAltimetry link
print(f"\n{'='*80}")
print("  COMO ACEDER AOS DADOS")
print(f"{'='*80}")
print(f"""
  1. VISUALIZAÇÃO RÁPIDA (browser):
     https://openaltimetry.org/data/icesat2/
     → Zoom para Algarve sul
     → Selecionar produto: ATL03
     → Ver tracks sobre a costa

  2. DOWNLOAD (NASA EarthData — requer registo gratuito):
     https://search.earthdata.nasa.gov/
     → Pesquisar: "ATL03"
     → Bounding Box: [{BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH}]
     → Download HDF5

  3. PYTHON (após registo):
     import earthaccess
     earthaccess.login()
     results = earthaccess.search_data(
         short_name="ATL03",
         bounding_box=({BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH}),
         temporal=("2018-10-01", "2026-01-01")
     )
     files = earthaccess.download(results, "./icesat2_data/")

  4. OPENALTIMETRY API (sem registo):
     https://openaltimetry.org/data/api/icesat2/
     → ATL03 photon data para coordenadas específicas

  Formato: HDF5 com grupos por beam (gt1l, gt1r, gt2l, gt2r, gt3l, gt3r)
  Variáveis chave:
    /gtXX/heights/h_ph        → altura de cada fotão (m)
    /gtXX/heights/lat_ph      → latitude
    /gtXX/heights/lon_ph      → longitude
    /gtXX/heights/signal_conf_ph → confiança (1-4, 4=alta)
  
  Para batimetria: filtrar fotões abaixo da superfície com signal_conf ≥ 3
""")
print("="*80)
