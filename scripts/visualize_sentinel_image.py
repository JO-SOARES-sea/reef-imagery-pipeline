#!/usr/bin/env python3
"""
Visualize Sentinel-2 JP2 bands for the Algarve reef site.
Loads B02, B03, and TCI bands and creates visualization plots.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Site coordinates (Algarve reef)
SITE_LAT = 37.0555
SITE_LON = 8.2296

# Paths to JP2 files
SENTINEL_DIR = Path("sentinel_images/20240930")
B02_FILE = SENTINEL_DIR / "T29SNB_20240930T112119_B02_10m.jp2"
B03_FILE = SENTINEL_DIR / "T29SNB_20240930T112119_B03_10m.jp2"
TCI_FILE = SENTINEL_DIR / "T29SNB_20240930T112119_TCI_10m.jp2"

def load_jp2_band(filepath, window=None):
    """Load JP2 band with optional window subset."""
    try:
        with rasterio.open(filepath) as src:
            print(f"\n📊 Band: {filepath.name}")
            print(f"   Shape: {src.shape}")
            print(f"   CRS: {src.crs}")
            print(f"   Bounds: {src.bounds}")
            print(f"   NoData: {src.nodata}")
            
            if window:
                data = src.read(1, window=window)
            else:
                data = src.read(1)
            
            return data, src
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None, None

def create_visualizations():
    """Load bands and create visualization plots."""
    
    # Load B02 (Blue, 490nm)
    b02_data, b02_src = load_jp2_band(B02_FILE)
    
    # Load B03 (Green, 560nm)
    b03_data, b03_src = load_jp2_band(B03_FILE)
    
    # Load TCI (True Color Composite)
    tci_data, tci_src = load_jp2_band(TCI_FILE)
    
    if b02_data is None or b03_data is None or tci_data is None:
        print("❌ Failed to load one or more bands")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Sentinel-2 T29SNB - September 30, 2024 | Algarve Reef Site", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: B02 (Blue band)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(b02_data, cmap='Blues', vmin=np.percentile(b02_data, 2), 
                     vmax=np.percentile(b02_data, 98))
    ax1.set_title("B02 - Blue (490nm)\n[Deep water penetration]", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Pixels (10m resolution)")
    ax1.set_ylabel("Pixels (10m resolution)")
    plt.colorbar(im1, ax=ax1, label="Reflectance (DN)")
    
    # Plot 2: B03 (Green band)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(b03_data, cmap='Greens', vmin=np.percentile(b03_data, 2), 
                     vmax=np.percentile(b03_data, 98))
    ax2.set_title("B03 - Green (560nm)\n[Best substrate contrast]", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Pixels (10m resolution)")
    ax2.set_ylabel("Pixels (10m resolution)")
    plt.colorbar(im2, ax=ax2, label="Reflectance (DN)")
    
    # Plot 3: TCI (True Color Composite)
    ax3 = axes[1, 0]
    tci_display = np.clip(tci_data.transpose(1, 2, 0) / 255.0, 0, 1)
    ax3.imshow(tci_display)
    ax3.set_title("TCI - True Color Composite\n[Human eye view]", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Pixels (10m resolution)")
    ax3.set_ylabel("Pixels (10m resolution)")
    
    # Plot 4: B02/B03 Ratio (Water clarity indicator)
    ax4 = axes[1, 1]
    ratio = np.divide(b03_data.astype(float), b02_data.astype(float) + 1e-8)
    im4 = ax4.imshow(ratio, cmap='RdYlBu_r', vmin=np.percentile(ratio, 2), 
                     vmax=np.percentile(ratio, 98))
    ax4.set_title("B03/B02 Ratio\n[Water clarity index]", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Pixels (10m resolution)")
    ax4.set_ylabel("Pixels (10m resolution)")
    plt.colorbar(im4, ax=ax4, label="Ratio")
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("sentinel_images/sentinel_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # Print statistics
    print("\n📈 Band Statistics:")
    print(f"   B02 - Min: {b02_data.min():.0f}, Max: {b02_data.max():.0f}, Mean: {b02_data.mean():.1f}")
    print(f"   B03 - Min: {b03_data.min():.0f}, Max: {b03_data.max():.0f}, Mean: {b03_data.mean():.1f}")
    print(f"   B03/B02 Ratio - Mean: {ratio.mean():.3f}")
    
    # Water quality assessment
    print("\n💧 Water Quality Assessment:")
    b02_mean = b02_data.mean()
    b03_mean = b03_data.mean()
    
    if b02_mean > 500 and b03_mean > 400:
        print("   Status: ✅ EXCELLENT - Low turbidity, high water clarity")
        print("   → Benthic substrate should be visible at moderate depths")
    elif b02_mean > 300 and b03_mean > 250:
        print("   Status: ✅ GOOD - Moderate clarity")
        print("   → Suitable for substrate detection at shallow-moderate depths")
    else:
        print("   Status: ⚠️  MODERATE/POOR - High turbidity")
        print("   → Limited benthic visibility, may require special processing")
    
    print("\n🎯 Physics Validation Summary:")
    print("   B02 (490nm) transmittance at 22m: 39.1% (theory)")
    print("   B03 (560nm) transmittance at 22m: 37.2% (theory)")
    print("   Both bands should penetrate well for benthic imaging")
    
    plt.show()

if __name__ == "__main__":
    print("🛰️  Sentinel-2 Image Visualization")
    print("=" * 50)
    print(f"Site: {SITE_LAT}°N, {SITE_LON}°E (Algarve)")
    print("Tile: T29SNB (September 30, 2024)")
    print("=" * 50)
    
    create_visualizations()
