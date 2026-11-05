import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src.enhancer import fetch_vsi_patch
from skimage.restoration import denoise_nl_means, estimate_sigma

lat = 37.05811
lon = -8.20978
dates = ["2022-09-11", "2024-09-10"]
_PROJECT_ROOT = Path(__file__).parent.parent
base_output_dir = _PROJECT_ROOT / "outputs"
base_output_dir.mkdir(parents=True, exist_ok=True)

for date_str in dates:
    output_dir = base_output_dir / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching VSI patch for {date_str}...")
    try:
        # Fetching patch with 2000m buffer
        b02_ref = fetch_vsi_patch(lat, lon, date_str, buffer_m=2000.0)
    except Exception as e:
        print(f"Failed to fetch patch for {date_str}: {e}")
        continue

    vmin, vmax = np.percentile(b02_ref[b02_ref > 0], 1), np.percentile(b02_ref[b02_ref > 0], 95)
    orig_vis = np.clip((b02_ref - vmin) / (vmax - vmin), 0, 1)

    print(f"Applying refinement for {date_str}...")
    p95 = np.percentile(b02_ref[b02_ref > 0], 95)
    b02_glint = np.clip(b02_ref - 0.8 * p95 * 0.05, 0, 1.0)

    sigma_est = np.mean(estimate_sigma(b02_glint))
    b02_den = denoise_nl_means(b02_glint, h=0.8 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)

    b02_16 = np.clip(b02_den * 65535, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(4,4))
    b02_clahe = clahe.apply(b02_16)
    b02_clahe_float = b02_clahe.astype(np.float32) / 65535.0

    b02_final = (b02_den * 0.5) + (b02_clahe_float * 0.5)

    ref_vmin, ref_vmax = np.percentile(b02_final[b02_final > 0], 1), np.percentile(b02_final[b02_final > 0], 95)
    ref_vis = np.clip((b02_final - ref_vmin) / (ref_vmax - ref_vmin), 0, 1)

    print(f"Saving images for {date_str}...")
    clean_date = date_str.replace("-", "")
    plt.imsave(os.path.join(output_dir, f"original_{clean_date}.png"), orig_vis, cmap='gray')
    plt.imsave(os.path.join(output_dir, f"refined_{clean_date}.png"), ref_vis, cmap='gray')
    plt.imsave(os.path.join(output_dir, f"refined_color_{clean_date}.png"), ref_vis, cmap='viridis')

print("All done!")
