#!/usr/bin/env python3
"""
rank_image_pair.py — Infer Winner Using Trained Siamese Model
=============================================================

Loads a trained Siamese Benthic Ranker model and predicts which 
of two Sentinel-2 images provides better benthic visibility.

Usage:
    python scripts/rank_image_pair.py --img-a path/to/imgA.png --img-b path/to/imgB.png --model models/siamese_benthic_ranker.pth

Author: Antigravity
"""

import argparse
import logging
import torch
from PIL import Image
from pathlib import Path

# Import the model and transforms from our training script
from train_siamese_ranking import SiameseBenthicRanker, get_transforms

log = logging.getLogger(__name__)

def predict_winner(img_a_path, img_b_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not Path(model_path).exists():
        log.error(f"Model weights not found at {model_path}. Please train the model first.")
        return

    log.info(f"Loading model from {model_path} onto {device}...")
    model = SiameseBenthicRanker(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transforms()

    log.info("Processing images...")
    try:
        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')
    except Exception as e:
        log.error(f"Failed to open images: {e}")
        return

    # Add batch dimension
    tensor_a = transform(img_a).unsqueeze(0).to(device)
    tensor_b = transform(img_b).unsqueeze(0).to(device)

    log.info("Running inference...")
    with torch.no_grad():
        score_a, score_b = model(tensor_a, tensor_b)
        
        score_a = score_a.item()
        score_b = score_b.item()
        
    diff = score_a - score_b
    
    # Simple probability conversion using sigmoid
    prob_a_better = torch.sigmoid(torch.tensor(diff)).item()
    
    log.info("=" * 40)
    log.info("RESULTS")
    log.info("=" * 40)
    log.info(f"Image A Score: {score_a:.4f} ({img_a_path})")
    log.info(f"Image B Score: {score_b:.4f} ({img_b_path})")
    
    if score_a > score_b:
        log.info("WINNER: Image A")
        log.info(f"Confidence (Prob A > B): {prob_a_better:.2%}")
    else:
        log.info("WINNER: Image B")
        log.info(f"Confidence (Prob B > A): {(1 - prob_a_better):.2%}")
    log.info("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Rank two images using Siamese network")
    parser.add_argument("--img-a", required=True, help="Path to Image A")
    parser.add_argument("--img-b", required=True, help="Path to Image B")
    parser.add_argument("--model", default="models/siamese_benthic_ranker.pth", help="Path to trained model weights")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    predict_winner(args.img_a, args.img_b, args.model)


if __name__ == "__main__":
    main()
