"""
benchmark_cpu.py — Measures CPU performance of the reef pipeline components.

Metrics recorded:
  1. ReefUNetCPU inference time (16 images, 128×128)
  2. Training: time per epoch on the local patch dataset
  3. Peak RSS memory during a training epoch
  4. ICESat-2 validation stub time (API latency check)

Results are appended to BENCHMARKS.md.

Usage:
    python benchmark_cpu.py                        # full benchmark
    python benchmark_cpu.py --skip-training        # inference + memory only
    python benchmark_cpu.py --epochs 3             # quick training probe
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maxrss_to_mb(ru_maxrss: int, plat: str) -> float:
    """Convert a ``ru_maxrss`` value to MB given the platform string.

    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` units are platform-dependent:
    macOS/Darwin reports the value in **bytes**, Linux reports it in
    **kilobytes**.  Treating bytes as kilobytes (or vice versa) produces wildly
    wrong figures (e.g. a 600 MB process logged as 656160 MB).

    Args:
        ru_maxrss: raw ``ru_maxrss`` value from ``resource.getrusage``.
        plat:      ``sys.platform`` string (e.g. ``"darwin"``, ``"linux"``).

    Returns:
        Peak RSS in megabytes.
    """
    if plat.startswith("darwin"):
        # macOS reports bytes.
        return ru_maxrss / 1e6
    # Linux (and other POSIX) report kilobytes.
    return ru_maxrss / 1024


def _peak_rss_mb() -> float:
    """Current process peak RSS in MB (cross-platform)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return _maxrss_to_mb(rss, sys.platform)
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss / 1e6


def _cpu_info() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ---------------------------------------------------------------------------
# Benchmark 1: Inference speed
# ---------------------------------------------------------------------------

def bench_inference(n_images: int = 16, img_size: int = 128) -> dict:
    import torch
    from models.unet_cpu import ReefUNetCPU

    model = ReefUNetCPU()
    model.eval()
    x = torch.randn(n_images, 1, img_size, img_size)

    # Warmup
    with torch.no_grad():
        model(x)

    # Timed run
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(3):
            model(x)
    elapsed = (time.perf_counter() - t0) / 3
    rss_after = _peak_rss_mb()

    return {
        "inference_s":     round(elapsed, 3),
        "inference_ms_per_img": round(elapsed / n_images * 1000, 1),
        "n_images":        n_images,
        "img_size":        img_size,
        "n_params":        model.n_params,
        "encoder":         model.encoder_name,
        "peak_rss_mb":     round(max(rss_before, rss_after), 0),
        "pass":            elapsed < 30.0,  # target: <30 s for 16 images
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Training speed
# ---------------------------------------------------------------------------

def bench_training(epochs: int = 3) -> dict:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from models.unet_cpu import ReefUNetCPU
    from models.losses import PhysicsInformedLoss
    from src.ml_unet_dataset import ReefPatchDataset, get_training_augmentation

    images_dir = _PROJECT_ROOT / "reef_output_sota/ml_dataset/images"
    masks_dir  = _PROJECT_ROOT / "reef_output_sota/ml_dataset/masks"

    if not images_dir.exists():
        return {"status": "skipped", "reason": "no training data found"}

    full_ds = ReefPatchDataset(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        transform=get_training_augmentation(),
    )
    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, _ = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=False)

    model     = ReefUNetCPU()
    criterion = PhysicsInformedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch_times = []
    rss_peak    = 0.0

    for _ in range(epochs):
        model.train()
        t0 = time.perf_counter()
        for imgs, masks in loader:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks, depth_norm=imgs)
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)
        rss_peak = max(rss_peak, _peak_rss_mb())

    avg_epoch = sum(epoch_times) / len(epoch_times)
    return {
        "epochs_run":       epochs,
        "avg_epoch_s":      round(avg_epoch, 1),
        "min_epoch_s":      round(min(epoch_times), 1),
        "max_epoch_s":      round(max(epoch_times), 1),
        "peak_rss_mb":      round(rss_peak, 0),
        "n_train_patches":  n_train,
        "pass":             avg_epoch < 300.0,  # target: <5 min/epoch
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Validation latency (OpenAltimetry API ping)
# ---------------------------------------------------------------------------

def bench_validation_latency() -> dict:
    import requests
    url = "https://openaltimetry.earthdatacloud.nasa.gov/data/api/icesat2/atl08"
    params = {
        "minx": -8.3, "miny": 37.0, "maxx": -8.2, "maxy": 37.1,
        "date": "2024-09-01", "trackId": 1,
    }
    t0 = time.perf_counter()
    try:
        r = requests.get(url, params=params, timeout=10)
        elapsed = time.perf_counter() - t0
        return {"api_latency_s": round(elapsed, 2), "status_code": r.status_code, "reachable": True}
    except Exception as exc:
        return {"api_latency_s": None, "reachable": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _append_benchmarks_md(results: dict) -> None:
    md_path = _PROJECT_ROOT / "BENCHMARKS.md"
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    cpu     = _cpu_info()

    lines = [
        f"\n## {ts}",
        f"**CPU:** {cpu}  |  **Python:** {platform.python_version()}",
        "",
        "| Metric | Value | Target | Pass |",
        "|--------|-------|--------|------|",
    ]

    inf = results.get("inference", {})
    if inf:
        p = "✅" if inf.get("pass") else "❌"
        lines.append(f"| Inference 16×128² | {inf['inference_s']:.2f} s | <30 s | {p} |")
        lines.append(f"| Inference ms/image | {inf['inference_ms_per_img']:.1f} ms | — | — |")
        lines.append(f"| Peak RSS (inference) | {inf['peak_rss_mb']:.0f} MB | <4000 MB | {'✅' if inf['peak_rss_mb']<4000 else '❌'} |")
        lines.append(f"| Encoder | {inf.get('encoder','')} | mobilenet | — |")
        lines.append(f"| Params | {inf.get('n_params',0):,} | <5M | {'✅' if inf.get('n_params',0)<5e6 else '❌'} |")

    tr = results.get("training", {})
    if tr and tr.get("status") != "skipped":
        p = "✅" if tr.get("pass") else "❌"
        lines.append(f"| Avg epoch time | {tr['avg_epoch_s']:.1f} s | <300 s (5 min) | {p} |")
        lines.append(f"| Peak RSS (training) | {tr['peak_rss_mb']:.0f} MB | <4000 MB | {'✅' if tr['peak_rss_mb']<4000 else '❌'} |")

    val = results.get("validation", {})
    if val.get("reachable"):
        lines.append(f"| OpenAltimetry latency | {val['api_latency_s']:.2f} s | <10 s | {'✅' if val['api_latency_s']<10 else '❌'} |")
    else:
        lines.append("| OpenAltimetry latency | unreachable | — | ⚠️ |")

    lines.append("")
    with open(md_path, "a") as fh:
        fh.write("\n".join(lines))
    print(f"\nResults appended to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_training: bool = False, epochs: int = 3) -> None:
    print(f"Reef Pipeline CPU Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"CPU: {_cpu_info()}\n")

    results: dict = {}

    print("── 1/3  Inference benchmark ──────────────────────────────")
    inf = bench_inference()
    results["inference"] = inf
    p = "PASS ✅" if inf.get("pass") else "FAIL ❌"
    print(f"  16 images × 128² → {inf['inference_s']:.2f} s  ({inf['inference_ms_per_img']:.1f} ms/img)  [{p}]")
    print(f"  Encoder: {inf['encoder']}  Params: {inf['n_params']:,}")
    print(f"  Peak RSS: {inf['peak_rss_mb']:.0f} MB")

    if not skip_training:
        print("\n── 2/3  Training benchmark ───────────────────────────────")
        tr = bench_training(epochs=epochs)
        results["training"] = tr
        if tr.get("status") == "skipped":
            print(f"  Skipped: {tr['reason']}")
        else:
            p = "PASS ✅" if tr.get("pass") else "FAIL ❌"
            print(f"  {epochs} epochs × {tr['n_train_patches']} patches → avg {tr['avg_epoch_s']:.1f} s/epoch  [{p}]")
            print(f"  Peak RSS: {tr['peak_rss_mb']:.0f} MB")
    else:
        print("\n── 2/3  Training benchmark — SKIPPED (--skip-training)")

    print("\n── 3/3  Validation API latency ───────────────────────────")
    val = bench_validation_latency()
    results["validation"] = val
    if val["reachable"]:
        print(f"  OpenAltimetry: {val['api_latency_s']:.2f} s  HTTP {val['status_code']}")
    else:
        print(f"  OpenAltimetry: unreachable ({val.get('error','')})")

    _append_benchmarks_md(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU benchmark for reef pipeline")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs for the timing probe (default: 3)")
    args = parser.parse_args()
    main(skip_training=args.skip_training, epochs=args.epochs)
