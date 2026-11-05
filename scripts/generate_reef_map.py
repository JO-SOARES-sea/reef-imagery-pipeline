import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import plotting_extent

BATHY_PATH = "outputs/santa_eulalia_12m/bathy_emodnet_20260529.tif"
TRI_PATH = "outputs/santa_eulalia_12m/bathy_emodnet_20260529_tri.tif"
BPI_PATH = "outputs/santa_eulalia_12m/bathy_emodnet_20260529_bpi_broad.tif"
REEF_PATH = "outputs/santa_eulalia_12m/reef_candidates_20260529_validated.geojson"
OUT_MAIN = "outputs/santa_eulalia_12m/reef_detection_map.png"
OUT_DETAIL = "outputs/santa_eulalia_12m/reef_detection_detail.png"

PEDRA_LAT, PEDRA_LON = 37.068978, -8.210328


def confidence_color(score):
    if score >= 70:
        return "red"
    elif score >= 50:
        return "gold"
    else:
        return "gray"


def confidence_label(score):
    if score >= 70:
        return "High"
    elif score >= 50:
        return "Moderate"
    else:
        return "Low"


reef_gdf = gpd.read_file(REEF_PATH)

with rasterio.open(BATHY_PATH) as src:
    bathy = src.read(1)
    bathy_extent = plotting_extent(src)
    bathy_transform = src.transform
    bathy_nodata = src.nodata

with rasterio.open(TRI_PATH) as src:
    tri = src.read(1)
    tri_extent = plotting_extent(src)

with rasterio.open(BPI_PATH) as src:
    bpi = src.read(1)
    bpi_extent = plotting_extent(src)

bathy_plot = np.where(bathy == bathy_nodata, np.nan, bathy) if bathy_nodata else bathy

fig, axes = plt.subplots(2, 2, figsize=(18, 16))

ax1 = axes[0, 0]
im1 = ax1.imshow(bathy_plot, extent=bathy_extent, cmap="Blues_r", aspect="auto")
reef_gdf.plot(ax=ax1, facecolor="none", edgecolor=[
    confidence_color(s) for s in reef_gdf["confidence_score"]
], linewidth=1.2)
cb1 = fig.colorbar(im1, ax=ax1, shrink=0.7, label="Depth (m)")
ax1.set_title("EMODnet Bathymetry + Reef Candidates", fontsize=13, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")

ax2 = axes[0, 1]
tri_plot = np.where(np.isnan(tri), np.nan, tri)
im2 = ax2.imshow(tri_plot, extent=tri_extent, cmap="YlOrRd", aspect="auto")
reef_gdf.plot(ax=ax2, facecolor="none", edgecolor=[
    confidence_color(s) for s in reef_gdf["confidence_score"]
], linewidth=1.2)
fig.colorbar(im2, ax=ax2, shrink=0.7, label="TRI")
ax2.set_title("TRI (Rugosity) + Reef Candidates", fontsize=13, fontweight="bold")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")

ax3 = axes[1, 0]
bpi_plot = np.where(np.isnan(bpi), np.nan, bpi)
vmax_bpi = np.nanmax(np.abs(bpi_plot))
im3 = ax3.imshow(bpi_plot, extent=bpi_extent, cmap="RdBu_r", vmin=-vmax_bpi, vmax=vmax_bpi, aspect="auto")
reef_gdf.plot(ax=ax3, facecolor="none", edgecolor=[
    confidence_color(s) for s in reef_gdf["confidence_score"]
], linewidth=1.2)
fig.colorbar(im3, ax=ax3, shrink=0.7, label="BPI")
ax3.set_title("BPI (Broad) + Reef Candidates", fontsize=13, fontweight="bold")
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")

ax4 = axes[1, 1]
ax4.imshow(bathy_plot, extent=bathy_extent, cmap="Blues_r", alpha=0.6, aspect="auto")
colors = [confidence_color(s) for s in reef_gdf["confidence_score"]]
reef_gdf.plot(ax=ax4, facecolor=colors, edgecolor="black", linewidth=0.8, alpha=0.5)
ax4.plot(PEDRA_LON, PEDRA_LAT, marker="*", color="cyan", markersize=18,
         markeredgecolor="black", markeredgewidth=1.2, zorder=10, label="Pedra Sta Eulália")
ax4.annotate("Pedra Sta Eulália", xy=(PEDRA_LON, PEDRA_LAT),
             xytext=(PEDRA_LON + 0.01, PEDRA_LAT + 0.01),
             fontsize=9, fontweight="bold", color="white",
             arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="navy", alpha=0.7))

top3 = reef_gdf.nlargest(3, "confidence_score")
for idx, row in top3.iterrows():
    centroid = row.geometry.centroid
    label = f"d={abs(row['depth_min_m']):.0f}m, c={row['confidence_score']}"
    ax4.annotate(label, xy=(centroid.x, centroid.y),
                 xytext=(centroid.x + 0.008, centroid.y + 0.008),
                 fontsize=8, color="white", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="red", edgecolor="black", label="High confidence (≥70)"),
    Patch(facecolor="gold", edgecolor="black", label="Moderate (50-69)"),
    Patch(facecolor="gray", edgecolor="black", label="Low (<50)"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="cyan",
               markeredgecolor="black", markersize=14, label="Pedra Sta Eulália"),
]
ax4.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)
ax4.set_title("Reef Detection — Pedra Sta Eulália (12m)", fontsize=13, fontweight="bold")
ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")

plt.tight_layout()
fig.savefig(OUT_MAIN, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_MAIN}")

fig2, ax_detail = plt.subplots(1, 1, figsize=(14, 12))
im_d = ax_detail.imshow(bathy_plot, extent=bathy_extent, cmap="Blues_r", aspect="auto")
reef_gdf.plot(ax=ax_detail, facecolor=[
    confidence_color(s) for s in reef_gdf["confidence_score"]
], edgecolor="black", linewidth=1.0, alpha=0.5)
fig2.colorbar(im_d, ax=ax_detail, shrink=0.7, label="Depth (m)")
ax_detail.plot(PEDRA_LON, PEDRA_LAT, marker="*", color="cyan", markersize=22,
               markeredgecolor="black", markeredgewidth=1.5, zorder=10)
ax_detail.annotate("Pedra Sta Eulália", xy=(PEDRA_LON, PEDRA_LAT),
                   xytext=(PEDRA_LON + 0.015, PEDRA_LAT + 0.015),
                   fontsize=11, fontweight="bold", color="white",
                   arrowprops=dict(arrowstyle="->", color="white", lw=2),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="navy", alpha=0.7))
for idx, row in top3.iterrows():
    centroid = row.geometry.centroid
    label = f"d={abs(row['depth_min_m']):.0f}m, c={row['confidence_score']}"
    ax_detail.annotate(label, xy=(centroid.x, centroid.y),
                       xytext=(centroid.x + 0.01, centroid.y + 0.01),
                       fontsize=9, color="white", fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color="white", lw=1),
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
ax_detail.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.8)
ax_detail.set_title("Bathymetry + Reef Candidates — Pedra Sta Eulália (12m Detail)",
                     fontsize=14, fontweight="bold")
ax_detail.set_xlabel("Longitude")
ax_detail.set_ylabel("Latitude")

fig2.savefig(OUT_DETAIL, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_DETAIL}")

plt.close("all")
print("Done.")
