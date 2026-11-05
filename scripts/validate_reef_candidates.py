#!/usr/bin/env python3
"""
Validação de Reef Candidates — Real vs Ruído
=============================================

Analisa os polígonos detetados e atribui scores de confiança baseados em:
- Área (recifes reais têm áreas consistentes)
- Rugosidade (TRI) — recifes são mais rugosos que sedimentos
- BPI (Bathymetric Position Index) — recifes elevados vs surroundings
- Relação com batimetria vizinha

Uso:
    python scripts/validate_reef_candidates.py \
        --candidates /tmp/reef_12m_detection/reef_candidates_20260524.geojson \
        --bathy /tmp/reef_12m_detection/bathy_emodnet_20260524.tif \
        --tri /tmp/reef_12m_detection/bathy_emodnet_20260524_tri.tif \
        --bpi /tmp/reef_12m_detection/bathy_emodnet_20260524_bpi_broad.tif

Output: GeoJSON com campo 'confidence_score' (0-100) e 'validation_notes'
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize

# Add src to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    import geopandas as gpd
    from shapely.geometry import shape
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("Error: geopandas and shapely required. Install: pip install geopandas shapely")
    sys.exit(1)


def compute_polygon_statistics(geom, bathy_arr, tri_arr, bpi_arr, transform):
    """Compute statistics for a single polygon from raster arrays."""
    # Rasterize polygon to mask
    mask = rasterize(
        [(geom, 1)],
        out_shape=bathy_arr.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    
    if mask.sum() == 0:
        return None
    
    # Extract values
    bathy_vals = bathy_arr[mask == 1]
    tri_vals = tri_arr[mask == 1]
    bpi_vals = bpi_arr[mask == 1]
    
    # Filter valid
    bathy_valid = bathy_vals[np.isfinite(bathy_vals)]
    tri_valid = tri_vals[np.isfinite(tri_vals)]
    bpi_valid = bpi_vals[np.isfinite(bpi_vals)]
    
    if len(bathy_valid) == 0:
        return None
    
    return {
        'depth_mean': float(np.mean(bathy_valid)),
        'depth_std': float(np.std(bathy_valid)),
        'tri_mean': float(np.mean(tri_valid)) if len(tri_valid) > 0 else 0,
        'tri_max': float(np.max(tri_valid)) if len(tri_valid) > 0 else 0,
        'bpi_mean': float(np.mean(bpi_valid)) if len(bpi_valid) > 0 else 0,
        'pixel_count': int(mask.sum()),
    }


def validate_candidate(stats, area_m2):
    """Validate a candidate and return confidence score (0-100) and notes."""
    score = 50  # Base score
    notes = []
    
    # 1. Area check (very small = suspicious)
    if area_m2 < 10000:  # < 1 hectare
        score -= 20
        notes.append("Small area (<1 ha) — may be noise or artifact")
    elif area_m2 > 50000:  # > 5 hectares
        score += 10
        notes.append("Large area — consistent with reef structure")
    
    # 2. Depth consistency (high std = suspicious)
    depth_range = stats['depth_std']
    if depth_range > 5:  # > 5m variation
        score -= 15
        notes.append(f"High depth variation ({depth_range:.1f}m) — check if uniform structure")
    else:
        score += 10
        notes.append("Consistent depth — good")
    
    # 3. Rugosity (TRI)
    tri_mean = stats['tri_mean']
    if tri_mean > 2.0:
        score += 15
        notes.append(f"High rugosity (TRI={tri_mean:.1f}) — characteristic of hard substrate")
    elif tri_mean > 1.0:
        score += 5
        notes.append(f"Moderate rugosity (TRI={tri_mean:.1f})")
    else:
        score -= 10
        notes.append(f"Low rugosity (TRI={tri_mean:.1f}) — may be sediment, not reef")
    
    # 4. BPI (elevation relative to surroundings)
    bpi_mean = stats['bpi_mean']
    if bpi_mean > 2.0:
        score += 15
        notes.append(f"Positive BPI ({bpi_mean:.1f}) — elevated structure, likely reef")
    elif bpi_mean > 0.5:
        score += 5
        notes.append(f"Weakly positive BPI ({bpi_mean:.1f})")
    else:
        score -= 5
        notes.append(f"Near-zero/negative BPI ({bpi_mean:.1f}) — not elevated")
    
    # 5. Pixel count vs area (check for fragmentation)
    pixels = stats['pixel_count']
    expected_pixels = area_m2 / (115 * 115)  # EMODnet ~115m resolution
    if pixels < expected_pixels * 0.5:
        score -= 10
        notes.append("Fragmented/irregular shape — possible artifact")
    
    # Clamp score
    score = max(0, min(100, score))
    
    # Classification
    if score >= 70:
        classification = "HIGH CONFIDENCE — Likely real reef"
    elif score >= 50:
        classification = "MODERATE — Needs verification"
    elif score >= 30:
        classification = "LOW — Probably noise/artifact"
    else:
        classification = "REJECT — Likely false positive"
    
    return {
        'score': score,
        'classification': classification,
        'notes': " | ".join(notes) if notes else "No specific issues"
    }


def main():
    parser = argparse.ArgumentParser(description='Validate reef candidates — detect real vs noise')
    parser.add_argument('--candidates', required=True, help='Input GeoJSON with reef candidates')
    parser.add_argument('--bathy', required=True, help='Bathymetry raster (tif)')
    parser.add_argument('--tri', required=True, help='TRI roughness raster (tif)')
    parser.add_argument('--bpi', required=True, help='BPI broad raster (tif)')
    parser.add_argument('--output', default=None, help='Output validated GeoJSON (default: overwrite input)')
    parser.add_argument('--min-score', type=int, default=0, help='Minimum score to keep (0-100)')
    
    args = parser.parse_args()
    
    # Read inputs
    print(f"Reading candidates: {args.candidates}")
    gdf = gpd.read_file(args.candidates)
    print(f"  → {len(gdf)} candidates found")
    
    print("Reading rasters...")
    with rasterio.open(args.bathy) as src:
        bathy_arr = src.read(1).astype(np.float32)
        transform = src.transform
        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            bathy_arr = np.where(bathy_arr == nodata, np.nan, bathy_arr)
    
    with rasterio.open(args.tri) as src:
        tri_arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            tri_arr = np.where(tri_arr == nodata, np.nan, tri_arr)
    
    with rasterio.open(args.bpi) as src:
        bpi_arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            bpi_arr = np.where(bpi_arr == nodata, np.nan, bpi_arr)
    
    print(f"  → Rasters loaded: {bathy_arr.shape}")
    
    # Validate each candidate
    print("\nValidating candidates...")
    results = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        area_m2 = row.get('area_m2', geom.area * (111320**2))  # Approx if not present
        
        stats = compute_polygon_statistics(geom, bathy_arr, tri_arr, bpi_arr, transform)
        
        if stats is None:
            validation = {
                'score': 0,
                'classification': 'ERROR — No valid data',
                'notes': 'Could not extract raster statistics'
            }
        else:
            validation = validate_candidate(stats, area_m2)
        
        results.append({
            'id': idx,
            'area_m2': area_m2,
            **validation,
            'stats': stats
        })
        
        print(f"\n  Candidate {idx+1}/{len(gdf)}:")
        print(f"    Area: {area_m2:,.0f} m²")
        print(f"    Score: {validation['score']}/100")
        print(f"    Status: {validation['classification']}")
        if stats:
            print(f"    Depth: {stats['depth_mean']:.1f}±{stats['depth_std']:.1f}m")
            print(f"    TRI: {stats['tri_mean']:.2f} (max {stats['tri_max']:.2f})")
            print(f"    BPI: {stats['bpi_mean']:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    high = sum(1 for r in results if r['score'] >= 70)
    moderate = sum(1 for r in results if 50 <= r['score'] < 70)
    low = sum(1 for r in results if 30 <= r['score'] < 50)
    reject = sum(1 for r in results if r['score'] < 30)
    
    print(f"  HIGH confidence (≥70):     {high} candidates")
    print(f"  MODERATE (50-69):          {moderate} candidates")
    print(f"  LOW (30-49):               {low} candidates")
    print(f"  REJECT (<30):              {reject} candidates")
    
    # Add validation fields to GeoDataFrame
    gdf['confidence_score'] = [r['score'] for r in results]
    gdf['validation_class'] = [r['classification'] for r in results]
    gdf['validation_notes'] = [r['notes'] for r in results]
    
    # Filter by min score if requested
    if args.min_score > 0:
        gdf_filtered = gdf[gdf['confidence_score'] >= args.min_score].copy()
        print(f"\n  Filtered to {len(gdf_filtered)}/{len(gdf)} candidates (score ≥ {args.min_score})")
    else:
        gdf_filtered = gdf
    
    # Save output
    output_path = args.output or args.candidates.replace('.geojson', '_validated.geojson')
    gdf_filtered.to_file(output_path, driver='GeoJSON')
    print(f"\n✓ Output saved: {output_path}")
    
    # Print recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if high > 0:
        print(f"\n✓ {high} HIGH confidence candidates are likely real reefs.")
        print("  → Recommended for field validation or ROV survey")
    
    if moderate > 0:
        print(f"\n⚠ {moderate} MODERATE confidence candidates need verification:")
        print("  → Cross-check with historical charts or local knowledge")
        print("  → Review in QGIS with orthophoto overlay")
    
    if low + reject > 0:
        print(f"\n✗ {low + reject} LOW/REJECT candidates are likely artifacts:")
        print("  → May be data gaps, noise, or sediment mounds")
        print("  → Consider excluding from further analysis")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Open the validated GeoJSON in QGIS")
    print("2. Overlay with OrtoSat2023 (30cm) or DGT orthophotos")
    print("3. Look for visual correlation:")
    print("   - Dark patches = possible reef shadows")
    print("   - Irregular texture = hard substrate")
    print("   - Uniform sand = false positive")
    print("4. Cross-reference with:")
    print("   - Historical nautical charts (IHM)")
    print("   - Local fishing knowledge")
    print("   - ROV/diver surveys if available")


if __name__ == '__main__':
    main()
