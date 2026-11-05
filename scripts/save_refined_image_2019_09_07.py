import numpy as np
import matplotlib.pyplot as plt
import os
from src.enhancer import fetch_vsi_patch

lat = 37.05811
lon = -8.20978
date_str = "2019-09-07"
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent
output_dir = _PROJECT_ROOT / "outputs" / date_str
output_dir.mkdir(parents=True, exist_ok=True)

b02_ref = fetch_vsi_patch(lat, lon, date_str, buffer_m=2000.0)
vmin, vmax = np.percentile(b02_ref[b02_ref > 0], 1), np.percentile(b02_ref[b02_ref > 0], 95)
orig_vis = np.clip((b02_ref - vmin) / (vmax - vmin), 0, 1)

plt.imsave(os.path.join(output_dir, "test_variance_20190907.png"), orig_vis, cmap='gray')
print("Done!")
