#!/usr/bin/env python3
"""
DGT Server U-Net Training Script — v3
Runs on AMD EPYC-Milan (96 cores, 503 GB RAM) — CPU-only, sem GPU.

Arquitetura: ReefUNet canónica (features=[64,128,256,512], sigmoid in forward).
Modelo: ~31 M params — maior que v2 (7.7 M) mas melhor capacidade de representação.

Novo em v3 vs v2:
- ARCH: ReefUNet canónica features=[64,128,256,512] (era [32,64,128,256]).
  Output: sigmoid probabilities — loss usa BCELoss (era BCEWithLogitsLoss com logits).
- CLASS IMBALANCE: WeightedRandomSampler — patches com recife amostrados 30× mais.
  Resolve o colapso para "prever zero" que ocorreu com 94.4% de patches vazios.
- pos_weight=20.0 no loss — pixels de recife valem 20× mais no BCE.
- Rebalanceamento: de ~6% para ~50% de patches com recife por batch.

Mantido de v2:
- BUG: dois _CachedReefDataset separados (fix do bug de transform partilhado).
- PERF: torch.set_num_threads explícito (16 threads no EPYC-Milan).
- STABILITY: gradient clipping (max_norm=1.0).
- SCHEDULE: CosineAnnealingLR.
- RESUME: full checkpoint every epoch.
- SPEED: todos os patches em RAM (341 × 128×128 ≈ 22 MB).
"""
from __future__ import annotations
import os, sys, json, time, random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

# ── Path setup ────────────────────────────────────────────────────────────────
# Repo root: auto-detect from this file's location so the script runs wherever
# the repo is cloned (e.g. /home/jovyan/work/reef_imagery_pipeline). Override
# with REEF_PIPELINE_ROOT if needed.
_env_root = os.environ.get("REEF_PIPELINE_ROOT")
WORK = Path(_env_root) if _env_root else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
os.chdir(str(WORK))

from src.ml_unet_model import ReefUNet, get_loss_function, calculate_iou
from src.constants import SDB_OPTICAL_LIMIT_M as _DEPTH_LIMIT

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR   = WORK / 'data' / 'master_ml_dataset'
MODELS_DIR    = WORK / 'models'
LOG_FILE      = WORK / 'training_log.json'
CHECKPOINT    = MODELS_DIR / 'checkpoint_latest.pth'
BEST_WEIGHTS  = MODELS_DIR / 'unet_reef_best.pth'

EPOCHS        = 100      # more epochs — CosineAnnealingLR benefits from longer runs
BATCH_SIZE    = 8        # was 32; 341*0.8/8 ≈ 34 batches/epoch (was 8)
LR            = 5e-4     # slightly lower than 1e-3 for warm-start stability
WEIGHT_DECAY  = 1e-4
MAX_GRAD_NORM = 1.0      # gradient clipping — prevents loss spikes
VAL_FRAC      = 0.20     # 80/20 split (was 70/15/15)
SEED          = 42

# Threads: PyTorch tensor ops. DataLoader workers are separate.
# On DGT (96-core EPYC-Milan), 16 threads for ops + 16 workers for augmentation.
NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', min(os.cpu_count() or 1, 16)))
NUM_WORKERS = int(os.environ.get('DL_NUM_WORKERS', min(os.cpu_count() or 1, 16)))
torch.set_num_threads(NUM_THREADS)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.makedirs(MODELS_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ── In-memory cached dataset (341 patches ≈ 22 MB — trivial for 125 GB RAM) ──

class _CachedReefDataset(Dataset):
    """
    Preloads all depth patches and masks into RAM.
    Each call to __getitem__ applies the supplied augmentation transform.

    This eliminates per-batch rasterio I/O which is the main bottleneck
    on spinning-disk or NFS-backed JupyterHub storage.
    """

    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False):
        import glob, rasterio
        img_paths  = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
        mask_paths = [os.path.join(masks_dir, os.path.basename(p)) for p in img_paths]

        limit = _DEPTH_LIMIT  # e.g. 40.0 m

        self._imgs:  list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        for ip, mp in zip(img_paths, mask_paths):
            with rasterio.open(ip) as s:
                raw = s.read(1).astype(np.float32)
            with rasterio.open(mp) as s:
                mask = s.read(1).astype(np.float32)
            # Normalise depth to [0,1]
            raw = np.nan_to_num(raw, nan=0.0)
            norm = np.clip((raw - (-limit)) / limit, 0.0, 1.0)
            self._imgs.append(norm)
            self._masks.append(mask)

        self._augment = augment
        print(f"  Loaded {len(self._imgs)} patches into RAM"
              f"  ({'train+aug' if augment else 'val/no-aug'})")

    def __len__(self) -> int:
        return len(self._imgs)

    def __getitem__(self, idx: int):
        img  = self._imgs[idx].copy()   # (H, W) float32
        mask = self._masks[idx].copy()  # (H, W) float32

        if self._augment:
            img, mask = _augment(img, mask)

        # (1, H, W)
        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))


def _augment(img: np.ndarray, mask: np.ndarray):
    """Spatial-only augmentation — safe for depth maps (no colour jitter)."""
    # Random horizontal/vertical flip
    if random.random() < 0.5:
        img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
    if random.random() < 0.5:
        img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
    # Random 90° rotation
    k = random.randint(0, 3)
    if k:
        img  = np.rot90(img,  k).copy()
        mask = np.rot90(mask, k).copy()
    # Small random depth noise (simulates SDB uncertainty ±2%)
    if random.random() < 0.3:
        img = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    return img, mask


# ── Dataset split ─────────────────────────────────────────────────────────────

def make_loaders():
    # Two separate instances → each has its own augment flag (v2 bug fix preserved)
    train_full = _CachedReefDataset(str(DATASET_DIR / 'images'),
                                    str(DATASET_DIR / 'masks'), augment=True)
    val_full   = _CachedReefDataset(str(DATASET_DIR / 'images'),
                                    str(DATASET_DIR / 'masks'), augment=False)
    n = len(train_full)
    indices = list(range(n))
    random.shuffle(indices)
    n_val   = max(1, int(n * VAL_FRAC))
    n_train = n - n_val
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Class imbalance fix: oversample reef patches 30× via WeightedRandomSampler.
    # Dataset has ≈94% empty patches (322/341) → model collapses to "predict nothing".
    # WeightedRandomSampler makes each batch ≈50% reef patches instead of ≈6%.
    has_reef_list = [bool((train_full._masks[i] > 0.5).any()) for i in train_idx]
    sample_weights = [30.0 if h else 1.0 for h in has_reef_list]
    
    reef_in_train = sum(1 for h in has_reef_list if h)
    empty_in_train = len(has_reef_list) - reef_in_train
    effective_reef_pct = (reef_in_train * 30) / (reef_in_train * 30 + empty_in_train) * 100
    print(f"  Split: {n_train} train / {n_val} val  (batch={BATCH_SIZE})")
    print(f"  Reef patches in train: {reef_in_train}/{n_train} "
          f"({100*reef_in_train/n_train:.1f}%) → oversampled to ≈{effective_reef_pct:.0f}%")

    # num_samples = 2× epoch length — more reef exposure per epoch
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=n_train * 2,
                                    replacement=True)

    train_loader = DataLoader(Subset(train_full, train_idx),
                              batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=False,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(Subset(val_full, val_idx),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False,
                              persistent_workers=(NUM_WORKERS > 0))
    return train_loader, val_loader


# ── Training utilities ─────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total = 0.0
    for imgs, masks in loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def val_epoch(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = total_iou = 0.0
    for imgs, masks in loader:
        out = model(imgs)
        total_loss += criterion(out, masks).item()
        total_iou  += calculate_iou(out, masks)
    n = max(len(loader), 1)
    return total_loss / n, total_iou / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("REEF U-NET TRAINING v3  —  DGT AMD EPYC-Milan  (CPU-only)")
    print(f"Cores: {os.cpu_count()}  Threads: {NUM_THREADS}  "
          f"Workers: {NUM_WORKERS}  PyTorch: {torch.__version__}")
    print("Arch: ReefUNet features=[64,128,256,512]  ~31M params")
    print("Loss: BCE(pos_weight=20) + SoftDice  |  Sampler: reef×30")
    print("=" * 60)

    train_loader, val_loader = make_loaders()

    device    = torch.device('cpu')  # NUNCA .cuda() — sem GPU no DGT HPC
    model     = ReefUNet().to(device)
    criterion = get_loss_function(pos_weight=20.0)  # reef pixels valem 20× mais no BCE
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Resume / warm-start ───────────────────────────────────────────────────
    start_epoch = 1
    best_iou    = 0.0
    history: list[dict] = []

    checkpoint_loaded = False
    if CHECKPOINT.exists():
        print(f"\nResuming full checkpoint: {CHECKPOINT}")
        try:
            ckpt = torch.load(str(CHECKPOINT), map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_iou    = ckpt['best_iou']
            history     = ckpt.get('history', [])
            print(f"  → epoch {start_epoch}  best_iou={best_iou:.4f}")
            checkpoint_loaded = True
        except (RuntimeError, EOFError, Exception) as e:
            print(f"  → checkpoint load failed ({type(e).__name__}: {e!s:.80}). Removing and starting fresh.")
            try:
                CHECKPOINT.unlink()  # delete corrupted checkpoint
            except OSError:
                pass

    if not checkpoint_loaded:
        if BEST_WEIGHTS.exists():
            print(f"\nWarm-start from best weights: {BEST_WEIGHTS}")
            try:
                state = torch.load(str(BEST_WEIGHTS), map_location=device)
                model.load_state_dict(state)
                print("  → weights loaded (optimizer reset, fresh LR schedule)")
            except (RuntimeError, EOFError, Exception) as e:
                print(f"  → weight load failed ({type(e).__name__}: {e!s:.80}); starting from scratch")
        else:
            print("\nStarting from scratch (no checkpoint or weights found)")

    # ── Training loop ─────────────────────────────────────────────────────────
    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        t_ep = time.time()
        tr_loss          = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_iou  = val_epoch(model, val_loader, criterion)
        scheduler.step()

        saved = ""
        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save(model.state_dict(), str(BEST_WEIGHTS))
            saved = "  ★ best"

        # Full checkpoint every epoch (enables resume after any interruption)
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou':             best_iou,
            'history':              history,
        }, str(CHECKPOINT))

        # Timestamp checkpoint every 5 epochs — preserva histórico de treino
        if epoch % 5 == 0:
            ts_path = MODELS_DIR / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_iou': best_iou}, str(ts_path))

        lr_now = optimizer.param_groups[0]['lr']
        row = dict(epoch=epoch, train_loss=round(tr_loss, 4),
                   val_loss=round(vl_loss, 4), val_iou=round(vl_iou, 4),
                   best_iou=round(best_iou, 4),
                   epoch_s=round(time.time() - t_ep, 1),
                   elapsed=round(time.time() - t0, 1))
        history.append(row)
        with open(LOG_FILE, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"Ep {epoch:>3}/{EPOCHS}  "
              f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
              f"iou={vl_iou:.4f}  lr={lr_now:.2e}  "
              f"{time.time()-t_ep:.0f}s{saved}", flush=True)

    print(f"\n{'='*60}")
    print(f"DONE  best_iou={best_iou:.4f}  "
          f"total={(time.time()-t0)/60:.1f} min")
    print(f"Model: {BEST_WEIGHTS}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
