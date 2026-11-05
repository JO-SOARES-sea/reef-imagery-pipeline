"""
reef_classifier.py — ResNet-18 patch classifier for reef vs sand.

Architecture:
  - ResNet-18 backbone (pretrained on ImageNet)
  - First conv layer adapted 3 → 4 input channels: RGB weights kept, 4th channel
    (NIR) initialised from the mean of the 3 pretrained channels.
  - Final FC replaced with a single linear → sigmoid for binary classification.

Input:  (B, 4, 128, 128) float32 tensor, values in [0, 1]
Output: (B,) float32 probability of reef presence

Usage
-----
    from src.reef_classifier import ReefClassifier, train, evaluate

    model = ReefClassifier(pretrained=True)
    history = train(model, train_dir, val_dir, output_dir, n_epochs=30)
    metrics = evaluate(model, val_dir)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms.functional as TF
    from torch.utils.data import DataLoader, Dataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    log.warning("PyTorch not available — ReefClassifier inference disabled")


def autodetect_device() -> str:
    """Best available PyTorch accelerator: CUDA → Apple MPS → CPU.

    Apple Silicon Macs have no CUDA, so the previous ``cuda else cpu`` logic
    silently trained on CPU. MPS (Metal Performance Shaders) is available on
    this hardware (torch.backends.mps.is_available()).
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────────────


class ReefPatchDataset(Dataset):  # type: ignore[misc]
    """
    Loads (4, 128, 128) float32 .npy patches from train/ or val/ directory.

    Expected layout::

        root/
            reef/   patch_00000.npy ...
            sand/   patch_00000.npy ...

    Label: 1 = reef, 0 = sand.
    """

    def __init__(self, root: str | Path, augment: bool = False) -> None:
        self.root = Path(root)
        self.augment = augment
        reef_files = sorted((self.root / "reef").glob("*.npy"))
        sand_files = sorted((self.root / "sand").glob("*.npy"))
        self.samples = [(p, 1) for p in reef_files] + [(p, 0) for p in sand_files]
        log.info(
            "Dataset loaded: root=%s  reef=%d  sand=%d",
            root,
            len(reef_files),
            len(sand_files),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)  # (4, H, W)
        x = torch.from_numpy(arr)

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                x = TF.hflip(x)
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                x = TF.vflip(x)
            # Random 90-degree rotation
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])

        return x, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────────────────────


class ReefClassifier(nn.Module):  # type: ignore[misc]
    """
    ResNet-18 adapted for 4-channel Sentinel-2 input → binary reef probability.

    Parameters
    ----------
    pretrained : Use ImageNet-pretrained ResNet-18 weights (recommended).
    dropout    : Dropout probability in the classification head (0 = disabled).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for ReefClassifier")

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Adapt first conv: 3 → 4 channels
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight  # RGB weights kept
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)  # NIR = mean
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        backbone.conv1 = new_conv

        # Replace classifier head
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.backbone(x).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)  # (B,) probability


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────────────────────


def train(
    model: ReefClassifier,
    train_dir: str | Path,
    val_dir: str | Path,
    output_dir: str | Path,
    *,
    n_epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: str | None = None,
    num_workers: int | None = None,
) -> list[dict]:
    """
    Train the model and save the best checkpoint.

    Early stopping: halts if validation AUC does not improve for *patience* epochs.

    Returns list of per-epoch dicts (training_log.json written to output_dir).
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required for training")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = autodetect_device()
    dev = torch.device(device)
    if num_workers is None:
        num_workers = 0 if dev.type == "cpu" else 2
    use_pin_memory = dev.type == "cuda"
    log.info("Training on device: %s (num_workers=%d)", dev, num_workers)

    model = model.to(dev)

    train_ds = ReefPatchDataset(train_dir, augment=True)
    val_ds = ReefPatchDataset(val_dir, augment=False)

    # Class-balanced sampler for imbalanced datasets
    labels = [lbl for _, lbl in train_ds.samples]
    n_reef = sum(labels)
    n_sand = len(labels) - n_reef
    weights = [1.0 / n_reef if lbl == 1 else 1.0 / n_sand for lbl in labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory
    )

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 100)

    best_val_auc = 0.0
    epochs_no_improve = 0
    history: list[dict] = []
    best_path = output_dir / "reef_cnn_best.pth"

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            prob = model(x)
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += ((prob > 0.5).float() == y).sum().item()
        scheduler.step()

        n_train = len(train_ds)
        avg_train_loss = train_loss / n_train
        train_acc = train_correct / n_train

        # ── Validation pass ────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                prob = model(x)
                val_loss += criterion(prob, y).item() * len(y)
                all_probs.append(prob.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        val_loss = val_loss / len(val_ds)
        val_acc = float(((all_probs > 0.5) == all_labels).mean())
        val_auc = _roc_auc(all_labels, all_probs)

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d  train_loss=%.4f  acc=%.3f  val_loss=%.4f  val_acc=%.3f  auc=%.3f  %.1fs",
            epoch,
            n_epochs,
            avg_train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auc,
            elapsed,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_auc": round(val_auc, 4),
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            log.info("  → New best (AUC=%.4f) saved to %s", val_auc, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    # Write training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training complete. Best val AUC=%.4f. Log: %s", best_val_auc, log_path)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    model: ReefClassifier,
    val_dir: str | Path,
    *,
    threshold: float = 0.5,
    device: str | None = None,
    num_workers: int | None = None,
) -> dict:
    """
    Evaluate model on the validation set and return classification metrics.

    Returns
    -------
    dict with: precision, recall, f1, auc, accuracy, n_reef, n_sand, threshold
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")

    if device is None:
        device = autodetect_device()
    dev = torch.device(device)
    if num_workers is None:
        num_workers = 0 if dev.type == "cpu" else 2
    model = model.to(dev)
    model.eval()

    val_ds = ReefPatchDataset(val_dir, augment=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=num_workers)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            prob = model(x.to(dev))
            all_probs.append(prob.cpu().numpy())
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    pred = (all_probs >= threshold).astype(int)

    tp = int(((pred == 1) & (all_labels == 1)).sum())
    fp = int(((pred == 1) & (all_labels == 0)).sum())
    fn = int(((pred == 0) & (all_labels == 1)).sum())
    tn = int(((pred == 0) & (all_labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(all_labels)
    auc = _roc_auc(all_labels, all_probs)

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_reef": int((all_labels == 1).sum()),
        "n_sand": int((all_labels == 0).sum()),
        "threshold": threshold,
    }
    log.info(
        "Evaluation: precision=%.3f  recall=%.3f  F1=%.3f  AUC=%.3f  acc=%.3f",
        precision,
        recall,
        f1,
        auc,
        accuracy,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn — O(n log n) via rank sum."""
    pos_mask = labels == 1
    n_pos = pos_mask.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(probs)
    ranks = np.argsort(order) + 1  # 1-based ranks
    rank_sum = ranks[pos_mask].sum()
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_model(checkpoint_path: str | Path, device: str | None = None) -> ReefClassifier:
    """Load a saved ReefClassifier from a checkpoint file."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")
    if device is None:
        device = autodetect_device()
    model = ReefClassifier(pretrained=False)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(torch.device(device))


def predict_single(
    model: ReefClassifier,
    patch: np.ndarray,
    device: str | None = None,
) -> float:
    """
    Predict reef probability for a single (4, H, W) float32 patch array.

    Returns probability in [0, 1].
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required")
    if device is None:
        device = autodetect_device()
    model = model.to(torch.device(device)).eval()
    x = torch.from_numpy(patch).unsqueeze(0).float().to(device)  # (1, 4, H, W)
    with torch.no_grad():
        prob = model(x).item()
    return float(prob)
