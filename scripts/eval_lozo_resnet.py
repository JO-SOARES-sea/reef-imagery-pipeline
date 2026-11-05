#!/usr/bin/env python3
"""
eval_lozo_resnet.py — Leave-One-Zone-Out evaluation of the ResNet-18 patch
classifier (4-band 128px patches over Olex 0–20 m marks).

Each zone is scored by a model trained ONLY on the other three zones, with a
depth-only logistic baseline trained on the SAME split, so the paired gain
(AUC_model − AUC_depth) answers: does the spectral patch add anything over
"reefs are shallower/deeper"?

Usage:
    python scripts/eval_lozo_resnet.py --epochs 5 --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reef_classifier import ReefClassifier  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "outputs" / "olex_0_20m"
ZONES = ("valado", "greta", "santa_eulalia", "quarteira")
OUT = PROJECT_ROOT / "outputs" / "eval"

_FNAME = re.compile(r"patch_(reef|sand)_(-?\d+\.\d+)_(-?\d+\.\d+)\.npy")


def _depth_map() -> dict[tuple[float, float], float]:
    reef = json.load(open(BASE / "olex_reefs_0_20m.geojson"))
    bathy = json.load(open(BASE / "olex_bathymetry_0_20m.geojson"))
    m: dict[tuple[float, float], float] = {}
    for f in reef["features"]:
        p = f["properties"]
        m[(round(p["lon"], 5), round(p["lat"], 5))] = float(p["olex_depth_m"])
    for f in bathy["features"]:
        p = f["properties"]
        m[(round(p["lon"], 5), round(p["lat"], 5))] = float(p["depth_m"])
    return m


def _load_zone(zone: str, depths: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, d = [], [], []
    for split in ("train", "val"):
        for cls, label in (("reef", 1), ("sand", 0)):
            dname = BASE / f"patches_{zone}" / split / cls
            if not dname.exists():
                continue
            for f in dname.glob("*.npy"):
                m = _FNAME.match(f.name)
                if not m:
                    continue
                lon, lat = float(m.group(2)), float(m.group(3))
                depth = depths.get((round(lon, 5), round(lat, 5)), np.nan)
                if not np.isfinite(depth):
                    continue
                X.append(np.load(f)); y.append(label); d.append(depth)
    return np.stack(X).astype(np.float32), np.array(y), np.array(d, dtype=np.float32)


def _bootstrap_gain(y, pm, pd, n_boot, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    gains = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        gains[i] = roc_auc_score(y[idx], pm[idx]) - roc_auc_score(y[idx], pd[idx])
    lo, hi = np.percentile(gains, (2.5, 97.5))
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}")

    depths = _depth_map()
    data = {z: _load_zone(z, depths) for z in ZONES}
    print({z: f"{len(v[0])} pts (gt={int(v[1].sum())})" for z, v in data.items()})

    print("\n=== ResNet-18 Leave-One-Zone-Out (Olex patches, 4-band) ===\n")
    print(f"{'zona':<15}{'n_gt':>5}{'n_bg':>6}{'AUC':>8}{'AUC_dep':>9}{'gain':>8}{'IC95 gain':>22}")
    print("-" * 80)
    rows = {}
    for held in ZONES:
        train_zones = [z for z in ZONES if z != held]
        Xtr = np.concatenate([data[z][0] for z in train_zones])
        ytr = np.concatenate([data[z][1] for z in train_zones])
        dtr = np.concatenate([data[z][2] for z in train_zones])
        Xte, yte, dte = data[held]

        # Depth-only baseline (same split)
        clf_d = LogisticRegression(max_iter=1000).fit(dtr.reshape(-1, 1), ytr)
        pd_ = clf_d.predict_proba(dte.reshape(-1, 1))[:, 1]

        # ResNet, class-balanced loss (pos_weight = n_neg/n_pos gives balanced
        # class weights without per-sample weight tensors, which MPS mishandles)
        model = ReefClassifier(pretrained=True, dropout=0.3).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=dev)
        Xt = torch.from_numpy(Xtr)
        for _ in range(args.epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), args.batch_size):
                idx = perm[i:i + args.batch_size]
                xb = Xt[idx].to(dev)
                yb = torch.from_numpy(ytr[idx.numpy()]).float().to(dev)
                p = model(xb).squeeze(-1).clamp(min=1e-7, max=1.0 - 1e-7)
                # Balanced BCE: positives weighted by n_neg/n_pos (elementwise
                # ops only — MPS-safe, no per-sample weight tensors).
                loss = -(pos_weight * yb * p.log() + (1.0 - yb) * (1.0 - p).log()).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        pm_ = []
        with torch.no_grad():
            for i in range(0, len(Xte), args.batch_size):
                pm_.append(model(torch.from_numpy(Xte[i:i + args.batch_size]).to(dev)).squeeze(-1).cpu().numpy())
        pm_ = np.concatenate(pm_)

        auc = roc_auc_score(yte, pm_)
        auc_d = roc_auc_score(yte, pd_)
        lo, hi = _bootstrap_gain(yte, pm_, pd_, args.n_boot, args.seed)
        n_gt, n_bg = int(yte.sum()), int((1 - yte).sum())
        rows[held] = {"n_gt": n_gt, "n_bg": n_bg, "auc": auc, "auc_depth_only": auc_d,
                      "gain": auc - auc_d, "gain_ci95": [lo, hi]}
        print(f"{held:<15}{n_gt:>5}{n_bg:>6}{auc:>8.3f}{auc_d:>9.3f}{auc - auc_d:>+8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "lozo_resnet_olex.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nGuardado em {OUT / 'lozo_resnet_olex.json'}")


if __name__ == "__main__":
    main()
