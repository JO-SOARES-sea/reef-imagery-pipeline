#!/usr/bin/env python3
"""
train_siamese_ranking.py — Siamese CNN Pairwise Image Ranking
===========================================================

This script trains a Siamese Neural Network (based on ResNet18) to perform
pairwise comparisons of Sentinel-2 coastal images. It learns to predict 
"benthic usefulness" by ranking images based on human preference.

Features:
- Dynamic ROI Cropping to focus on the reef and exclude titles/colorbars.
- ResNet18 shared backbone.
- MarginRankingLoss for pairwise supervision.

Usage:
    python scripts/train_siamese_ranking.py --data-csv data/pairwise_labels.csv

Author: Antigravity
"""

import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. Dataset & Preprocessing
# ---------------------------------------------------------

class BenthicPairwiseDataset(Dataset):
    """
    Dataset for pairwise image comparison.
    Expects a CSV with columns: ['image_a_path', 'image_b_path', 'label']
    label: 1 if A > B (A is better), -1 if B > A.
    """
    def __init__(self, csv_file, transform=None):
        if not os.path.exists(csv_file):
            # Create a dummy dataframe for demonstration if not found
            log.warning(f"CSV {csv_file} not found. Using a dummy dataset for demonstration.")
            self.data = pd.DataFrame({
                'image_a_path': ['dummy_a.jpg'],
                'image_b_path': ['dummy_b.jpg'],
                'label': [1]
            })
            self.is_dummy = True
        else:
            self.data = pd.read_csv(csv_file)
            self.is_dummy = False
            
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        if self.is_dummy:
            # Generate random noise tensors for dummy mode
            img_a = torch.randn(3, 224, 224)
            img_b = torch.randn(3, 224, 224)
            return img_a, img_b, label
            
        img_a_path = row['image_a_path']
        img_b_path = row['image_b_path']

        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b, label


def get_transforms():
    """
    Returns the torchvision transforms for preprocessing.
    Includes ROI cropping to ignore titles at the top and borders.
    """
    return transforms.Compose([
        # Crop the bottom 80% of the image to ignore titles/headers
        transforms.Lambda(lambda img: img.crop((0, int(img.height * 0.2), img.width, img.height))),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])


# ---------------------------------------------------------
# 2. Model Architecture
# ---------------------------------------------------------

class SiameseBenthicRanker(nn.Module):
    """
    Siamese Network using ResNet18 to output a scalar "Benthic Score".
    """
    def __init__(self, pretrained=True):
        super(SiameseBenthicRanker, self).__init__()
        # Use a lightweight ResNet18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final classification layer with a scoring head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Outputs a single scalar score
        )

    def forward_one(self, x):
        """Pass one image through the network to get its score."""
        return self.backbone(x)

    def forward(self, img_a, img_b):
        """Pass both images through the network."""
        score_a = self.forward_one(img_a)
        score_b = self.forward_one(img_b)
        return score_a, score_b


# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------

def train_siamese_model(csv_path, epochs=5, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    # Prepare Dataset & DataLoader
    transform = get_transforms()
    dataset = BenthicPairwiseDataset(csv_file=csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = SiameseBenthicRanker(pretrained=True).to(device)
    
    # MarginRankingLoss: max(0, -y * (x1 - x2) + margin)
    # y=1 means we want x1 > x2 by at least 'margin'
    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log.info("Starting training loop...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_pairs = 0
        total_pairs = 0
        
        for i, (img_a, img_b, labels) in enumerate(dataloader):
            img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            score_a, score_b = model(img_a, img_b)
            
            # Compute loss
            # score_a and score_b are shape (batch, 1), need to view as 1D
            loss = criterion(score_a.view(-1), score_b.view(-1), labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            # If label == 1, we want score_a > score_b
            # If label == -1, we want score_b > score_a
            preds = (score_a.view(-1) - score_b.view(-1)).sign()
            correct_pairs += (preds == labels).sum().item()
            total_pairs += labels.size(0)
            
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
        
        log.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Pairwise Acc: {accuracy:.4f}")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    save_path = "models/siamese_benthic_ranker.pth"
    torch.save(model.state_dict(), save_path)
    log.info(f"Training complete. Model saved to {save_path}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Train Siamese pairwise ranking model")
    parser.add_argument("--data-csv", default="data/pairwise_labels.csv", 
                        help="CSV file with columns: image_a_path, image_b_path, label")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    # Run the training pipeline
    train_siamese_model(csv_path=args.data_csv, epochs=args.epochs)

if __name__ == "__main__":
    main()
