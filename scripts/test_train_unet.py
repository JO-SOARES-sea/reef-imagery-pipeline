import os
import sys
import shutil
from glob import glob
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

sys.path.append(os.path.abspath('.'))

from src.ml_unet_dataset import ReefPatchDataset, get_training_augmentation, get_validation_augmentation
from src.ml_unet_model import ReefUNet, get_loss_function

def main():
    print("1. Aggregating Dataset...")
    MASTER_DATASET = 'data/master_ml_dataset'
    os.makedirs(f'{MASTER_DATASET}/images', exist_ok=True)
    os.makedirs(f'{MASTER_DATASET}/masks', exist_ok=True)

    patch_dirs = glob('reef_output_*/ml_dataset')
    idx = 0
    for pdir in patch_dirs:
        images = glob(os.path.join(pdir, 'images', '*.tif'))
        for img_path in images:
            mask_path = img_path.replace('images', 'masks')
            if os.path.exists(mask_path):
                shutil.copy(img_path, f'{MASTER_DATASET}/images/patch_{idx:04d}.tif')
                shutil.copy(mask_path, f'{MASTER_DATASET}/masks/patch_{idx:04d}.tif')
                idx += 1
    print(f"Total aggregated patches: {idx}")

    if idx == 0:
        print("No patches found. Exiting.")
        return

    print("2. Setting up DataLoaders...")
    full_dataset = ReefPatchDataset(
        images_dir=f'{MASTER_DATASET}/images',
        masks_dir=f'{MASTER_DATASET}/masks',
        transform=get_training_augmentation()
    )

    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset.dataset.transform = get_validation_augmentation()

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print("3. Initializing Model & Testing Forward Pass (1 Epoch Mock)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = ReefUNet().to(device)
        criterion = get_loss_function()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Test single batch forward/backward
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            print(f"  -> Forward/Backward pass SUCCESS! Loss: {loss.item():.4f}")
            break # Just test one batch
            
        print("Everything is working correctly!")
    except Exception as e:
        print(f"Error during model initialization or training: {e}")

if __name__ == '__main__':
    main()
