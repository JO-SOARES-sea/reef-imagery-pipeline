import os
import sys
from pathlib import Path

# Fix pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from src.ml_unet_model import ReefUNet, get_loss_function

_DEPTH_LIMIT = 40.0
class _CachedReefDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False):
        import glob, rasterio
        img_paths  = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
        mask_paths = [os.path.join(masks_dir, os.path.basename(p)) for p in img_paths]

        limit = _DEPTH_LIMIT

        self._imgs:  list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        for ip, mp in zip(img_paths, mask_paths):
            with rasterio.open(ip) as s:
                raw = s.read(1).astype(np.float32)
            with rasterio.open(mp) as s:
                mask = s.read(1).astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0)
            norm = np.clip((raw - (-limit)) / limit, 0.0, 1.0)
            self._imgs.append(norm)
            self._masks.append(mask)
        self._augment = augment

    def __len__(self) -> int: return len(self._imgs)
    def __getitem__(self, idx: int):
        img  = self._imgs[idx].copy()
        mask = self._masks[idx].copy()
        if self._augment:
            if random.random() < 0.5: img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5: img, mask = img[::-1, :].copy(), mask[::-1, :].copy()
            k = random.randint(0, 3)
            if k: img, mask = np.rot90(img,  k).copy(), np.rot90(mask, k).copy()
            if random.random() < 0.3: img = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    DATASET_DIR = Path('data/master_ml_dataset')
    images_dir = str(DATASET_DIR / 'images')
    masks_dir = str(DATASET_DIR / 'masks')
    
    full_dataset_aug = _CachedReefDataset(images_dir, masks_dir, augment=True)
    full_dataset_val = _CachedReefDataset(images_dir, masks_dir, augment=False)
    
    n = len(full_dataset_val)
    has_reef_list = [bool((full_dataset_val._masks[i] > 0.5).any()) for i in range(n)]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), has_reef_list)):
        print(f"\n--- FOLD {fold+1} ---")
        
        train_reef = sum(has_reef_list[i] for i in train_idx)
        val_reef = sum(has_reef_list[i] for i in val_idx)
        print(f"Train size: {len(train_idx)} ({train_reef} reef), Val size: {len(val_idx)} ({val_reef} reef)")
        
        sample_weights = [30.0 if has_reef_list[i] else 1.0 for i in train_idx]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_idx)*2, replacement=True)
        
        train_loader = DataLoader(Subset(full_dataset_aug, train_idx), batch_size=32, sampler=sampler)
        val_loader = DataLoader(Subset(full_dataset_val, val_idx), batch_size=32, shuffle=False)
        
        model = ReefUNet().to(device)
        criterion = get_loss_function(pos_weight=20.0)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        
        t0 = time.time()
        for epoch in range(1, 101):
            model.train()
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            if epoch == 1: print(f"  Epoch 1 took {time.time()-t0:.2f}s")
        print(f"  Fold {fold+1} 100 epochs took {time.time()-t0:.1f}s")
        
        model.eval()
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for i in val_idx:
                img_np = full_dataset_val._imgs[i]
                mask_np = full_dataset_val._masks[i]
                x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
                out = model(x)
                pred = (out[0,0].cpu().numpy() > 0.5).astype(int)
                gt = (mask_np > 0.5).astype(int)
                
                tp += np.sum((pred == 1) & (gt == 1))
                fp += np.sum((pred == 1) & (gt == 0))
                tn += np.sum((pred == 0) & (gt == 0))
                fn += np.sum((pred == 0) & (gt == 1))
        
        eps = 1e-6
        pr = tp / (tp + fp + eps)
        re = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        print(f"Fold {fold+1} metrics -> IoU: {iou:.4f}, Pr: {pr:.4f}, Re: {re:.4f}")
        results.append((iou, pr, re))
    
    ious = [r[0] for r in results]
    prs = [r[1] for r in results]
    res = [r[2] for r in results]
    print("\n--- FINAL K-FOLD RESULTS ---")
    print(f"IoU:       {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Precision: {np.mean(prs):.4f} ± {np.std(prs):.4f}")
    print(f"Recall:    {np.mean(res):.4f} ± {np.std(res):.4f}")

if __name__ == '__main__':
    main()
