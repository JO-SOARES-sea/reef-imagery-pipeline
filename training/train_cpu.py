"""
train_cpu.py — CPU-optimised training loop for ReefUNetCPU.

Usage:
    python training/train_cpu.py
    python training/train_cpu.py --config training/config_cpu.yaml
    python training/train_cpu.py --config training/config_cpu.yaml --epochs 10

Outputs:
    models/unet_reef_cpu.pth    — best checkpoint (lowest val_loss)
    training/metrics_cpu.csv    — per-epoch metrics (train_loss, val_loss, val_iou)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split

# Project root on sys.path so src.* imports work whether called from repo root or training/
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.unet_cpu import ReefUNetCPU
from models.losses import PhysicsInformedLoss
from src.ml_unet_dataset import ReefPatchDataset, get_training_augmentation, get_validation_augmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _cfg(cfg: dict, *keys, default=None):
    """Safe nested key access."""
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self._best = float("inf")
        self._counter = 0
        self.stop = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self._best:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.stop = True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: ReefUNetCPU,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    total = 0.0
    for imgs, masks in loader:
        optimizer.zero_grad()
        logits = model(imgs)
        # depth_norm: re-derive from input image (already normalised [0,1] by dataset)
        loss = criterion(logits, masks, depth_norm=imgs)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(
    model: ReefUNetCPU,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_iou = 0.0
    for imgs, masks in loader:
        logits = model(imgs)
        total_loss += criterion(logits, masks, depth_norm=imgs).item()
        total_iou  += calculate_iou(logits, masks)
    n = len(loader)
    return total_loss / n, total_iou / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | Path = "training/config_cpu.yaml", overrides: dict | None = None) -> None:
    cfg = load_config(config_path)
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v

    torch.set_num_threads(min(torch.get_num_threads(), 8))  # cap at 8 — diminishing returns beyond this on CPU
    log.info("CPU threads: %d", torch.get_num_threads())

    # Dataset
    images_dir = _PROJECT_ROOT / _cfg(cfg, "data", "images_dir", default="reef_output_sota/ml_dataset/images")
    masks_dir  = _PROJECT_ROOT / _cfg(cfg, "data", "masks_dir",  default="reef_output_sota/ml_dataset/masks")
    val_split  = float(_cfg(cfg, "data", "val_split", default=0.2))

    full_ds = ReefPatchDataset(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        transform=get_training_augmentation(),
    )
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    # val set uses no augmentation
    val_ds.dataset.transform = get_validation_augmentation()  # type: ignore[attr-defined]

    batch   = int(_cfg(cfg, "training", "batch_size", default=4))
    workers = int(_cfg(cfg, "training", "num_workers", default=2))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    log.info("Dataset — train: %d  val: %d  batch: %d", n_train, n_val, batch)

    # Model
    model = ReefUNetCPU(encoder_name=_cfg(cfg, "model", "encoder_name"))
    log.info("Model — %s  params: %d", model.encoder_name, model.n_params)

    # Loss / optimiser / scheduler
    criterion = PhysicsInformedLoss(
        lambda_depth =float(_cfg(cfg, "loss", "lambda_depth",  default=0.3)),
        lambda_smooth=float(_cfg(cfg, "loss", "lambda_smooth", default=0.1)),
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(_cfg(cfg, "training", "lr", default=1e-3)),
        weight_decay=float(_cfg(cfg, "training", "weight_decay", default=1e-4)),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(_cfg(cfg, "scheduler", "patience", default=3)),
        factor  =float(_cfg(cfg, "scheduler", "factor",   default=0.5)),
        min_lr  =float(_cfg(cfg, "scheduler", "min_lr",   default=1e-5)),
    )
    stopper = EarlyStopping(patience=int(_cfg(cfg, "early_stopping", "patience", default=5)))

    # Output paths
    model_path  = _PROJECT_ROOT / _cfg(cfg, "output", "model_path",  default="models/unet_reef_cpu.pth")
    metrics_csv = _PROJECT_ROOT / _cfg(cfg, "output", "metrics_csv", default="training/metrics_cpu.csv")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    # Training
    epochs    = int(_cfg(cfg, "training", "epochs", default=30))
    best_loss = float("inf")

    with open(metrics_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "epoch_time_s", "lr"])

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_iou = val_epoch(model, val_loader, criterion)
            elapsed = time.perf_counter() - t0

            scheduler.step(val_loss)
            stopper(val_loss)
            lr_now = optimizer.param_groups[0]["lr"]

            marker = ""
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_path)
                marker = " ★ saved"

            log.info(
                "Epoch %02d/%d  train=%.4f  val=%.4f  iou=%.4f  %.1fs  lr=%.2e%s",
                epoch, epochs, train_loss, val_loss, val_iou, elapsed, lr_now, marker,
            )
            writer.writerow([epoch, round(train_loss, 6), round(val_loss, 6),
                             round(val_iou, 4), round(elapsed, 1), lr_now])
            fh.flush()

            if stopper.stop:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, stopper.patience)
                break

    log.info("Training complete — best val_loss=%.4f  model saved to %s", best_loss, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReefUNetCPU")
    parser.add_argument("--config", default="training/config_cpu.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    overrides: dict = {}
    if args.epochs:
        overrides = {"training": {"epochs": args.epochs}}

    main(args.config, overrides or None)
