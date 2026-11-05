import numpy as np
from src.enhancer import fetch_vsi_patch

lat, lon = 37.05811, -8.20978

def analyze(date_str):
    try:
        b02_ref = fetch_vsi_patch(lat, lon, date_str, buffer_m=2000.0)
    except Exception as e:
        return {"error": str(e)}

    # Remove 0s (nodata)
    valid = b02_ref[b02_ref > 0]
    if len(valid) == 0:
        return {"error": "Empty data"}

    # Basic stats
    mean_val = np.mean(valid)
    std_val = np.std(valid)
    p1 = np.percentile(valid, 1)
    p99 = np.percentile(valid, 99)
    contrast_ratio = p99 / p1 if p1 > 0 else 0

    # Entropy
    hist, _ = np.histogram(valid, bins=256, range=(0, 1))
    p = hist / np.sum(hist)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))

    # Frequency domain (FFT) to see structural content vs noise
    f_transform = np.fft.fft2(b02_ref)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
    
    # Calculate energy in high vs low frequencies
    h, w = b02_ref.shape
    cy, cx = h // 2, w // 2
    r = 20 # radius for low frequencies
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    
    low_freq_energy = np.sum(magnitude_spectrum[mask])
    high_freq_energy = np.sum(magnitude_spectrum[~mask])
    freq_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0

    return {
        "mean": mean_val,
        "std": std_val,
        "p1": p1,
        "p99": p99,
        "contrast_ratio": contrast_ratio,
        "entropy": entropy,
        "freq_ratio": freq_ratio
    }

dates = ["2025-09-25", "2023-09-01", "2024-09-30", "2022-09-26"]
for d in dates:
    res = analyze(d)
    print(f"--- Date: {d} ---")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}")
        else:
            print(f"  {k}: {v}")

