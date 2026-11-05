import time, torch
from src.ml_unet_model import ReefUNet
from scripts.dgt_train_unet import make_loaders

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')
model = ReefUNet().to(device)
train_loader, _ = make_loaders()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss() # Doesn't matter for speed test

t0 = time.time()
model.train()
for i, (imgs, masks) in enumerate(train_loader):
    imgs, masks = imgs.to(device), masks.to(device)
    optimizer.zero_grad()
    out = model(imgs)
    loss = criterion(out, masks)
    loss.backward()
    optimizer.step()
t1 = time.time()
print(f'One epoch took: {t1-t0:.2f} seconds')
