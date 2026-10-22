"""
losses.py — Physics-informed loss functions for reef segmentation on CPU.

This module owns the canonical losses used by the reef segmentation pipeline:

    BCEDiceLoss          — BCE (with pos_weight) + Soft Dice, logits-in.
                           Designed for the master_ml_dataset (1.19% positive
                           pixels) with pos_weight ≈ 80.
    ReefSegmentationLoss — Convenience wrapper that adds an optional
                           depth-consistency term + TV smoothness term.
    DepthConsistencyLoss — Penalises predictions inconsistent with depth
                           physics (shallow → reef, deep → no-reef).
    SmoothnessLoss       — Total-variation smoothness penalty.
    PhysicsInformedLoss  — Backwards-compatible composite (BCE + Dice +
                           depth + smooth) — kept for older checkpoints.
    pos_weight_from_dataset — Computes pos_weight from a directory of
                               binary uint8 masks (used by the splitter).

All losses accept LOGITS (not post-sigmoid probabilities) and work on CPU.
For extremely imbalanced binary segmentation (≈ 1% positive pixels) the
composite BCE(pos_weight≈80) + Dice is the recommended default; Dice
handles geometric alignment, BCE(pos_weight) handles class imbalance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pos-weight helper
# ─────────────────────────────────────────────────────────────────────────────


def pos_weight_from_dataset(mask_dir: str | Path,
                            recursive: bool = False,
                            sample: int | None = None) -> float:
    """
    Compute ``pos_weight = n_negative / n_positive`` for a binary segmentation
    dataset on disk.

    Reads every ``*.tif`` mask under ``mask_dir`` as a single-band uint8 array
    and counts pixels equal to 1 (positive) vs 0 (negative). Designed for the
    ``data/master_ml_dataset/masks/`` layout (flat directory, 341 files).

    Args:
        mask_dir:  path to a directory of binary uint8 masks.
        recursive: if True, recurse into subdirectories.
        sample:    if not None, stop after reading this many masks (useful for
                   large datasets where exact count is not required).

    Returns:
        float >= 1.0 representing ``n_negative / n_positive``. Returns the
        safe default ``80.0`` (the master_ml_dataset ratio) if the directory
        contains no positive pixels at all.
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask_dir does not exist: {mask_dir}")

    # Local import so this helper is importable even if rasterio is missing.
    import rasterio

    pattern = "**/*.tif" if recursive else "*.tif"
    paths = sorted(mask_dir.glob(pattern))
    if sample is not None:
        paths = paths[:sample]

    n_pos = 0
    n_neg = 0
    for p in paths:
        with rasterio.open(p) as src:
            a = src.read(1)
        valid = (a == 0) | (a == 1)
        n_pos += int((a == 1).sum())
        n_neg += int((a == 0)[valid].sum())

    if n_pos == 0:
        log.warning(
            "pos_weight_from_dataset(%s): no positive pixels found — returning default 80.0",
            mask_dir,
        )
        return 80.0

    pw = n_neg / n_pos
    log.info(
        "pos_weight_from_dataset(%s): %d masks scanned, "
        "n_pos=%d n_neg=%d → pos_weight=%.3f",
        mask_dir, len(paths), n_pos, n_neg, pw,
    )
    return float(pw)


# ─────────────────────────────────────────────────────────────────────────────
# Core building blocks
# ─────────────────────────────────────────────────────────────────────────────


class BCEDiceLoss(nn.Module):
    """
    BCE (with ``pos_weight``) + Soft Dice — logits in, scalar out.

    Designed for highly imbalanced binary segmentation (≈ 1% positive pixels).
    The Dice term handles geometric alignment on small patches; the
    ``pos_weight`` term handles class imbalance. Both are needed because Dice
    alone diverges when no positive pixels are predicted, and plain BCE alone
    collapses to all-zero predictions when the prior is overwhelmingly
    negative.

    Args:
        pos_weight:  weight applied to positive (reef) pixels in BCE. Set to
                     ``(1 - p) / p`` where ``p`` is the positive-pixel ratio
                     in the training set. For the master_ml_dataset
                     (p≈0.0119) the recommended value is 80.
        dice_weight: weight of the Dice term in the sum (default 1.0).
        bce_weight:  weight of the BCE term in the sum (default 1.0).
        from_dataset: optional path to a masks directory — if supplied,
                      ``pos_weight`` is computed automatically from the on-disk
                      masks (overrides the ``pos_weight`` argument). Useful in
                      the training script: ``BCEDiceLoss(from_dataset=mask_dir)``.

    Forward signature:
        loss = criterion(logits, targets)

        logits:  (B, 1, H, W) raw logits (no sigmoid applied yet)
        targets: (B, 1, H, W) float/binary ground-truth in {0.0, 1.0}
    """

    def __init__(
        self,
        pos_weight: float = 80.0,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        from_dataset: str | Path | None = None,
    ) -> None:
        super().__init__()
        if from_dataset is not None:
            pw = pos_weight_from_dataset(from_dataset)
        else:
            pw = float(pos_weight)
        if pw < 1.0:
            raise ValueError(
                f"pos_weight must be >= 1.0 (got {pw}); values < 1 would make "
                "positive pixels cost *less* than negatives."
            )

        # nn.BCEWithLogitsLoss is numerically more stable than F.binary_cross_entropy
        # with sigmoid applied separately. Register pos_weight as a buffer so it
        # follows the module to the correct device.
        self.register_buffer("_pos_weight", torch.tensor([pw], dtype=torch.float32))
        self._bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        self._dice = _SoftDiceLoss()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.pos_weight_value = pw

    @property
    def pos_weight(self) -> float:
        return float(self._pos_weight.item())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self._bce(logits, targets.float())
        dice = self._dice(logits, targets.float())
        return self.bce_weight * bce + self.dice_weight * dice

    def extra_repr(self) -> str:
        return (
            f"pos_weight={self.pos_weight:.2f}, "
            f"bce_weight={self.bce_weight}, dice_weight={self.dice_weight}"
        )


class _SoftDiceLoss(nn.Module):
    """Soft Dice loss on LOGITS — numerically stable, no sigmoid-then-loss."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Physics / smoothness terms
# ─────────────────────────────────────────────────────────────────────────────


class DepthConsistencyLoss(nn.Module):
    """
    Penalises predictions inconsistent with the depth physics.

    Shallow pixels (normalised depth near 1.0) should have HIGH reef probability.
    Deep pixels (normalised depth near 0.0) should have LOW reef probability.

    The penalty is the L2 distance between sigmoid(logits) and the
    depth-derived prior p_reef = depth_norm (linear heuristic).
    """

    def forward(self, logits: torch.Tensor, depth_norm: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # depth_norm ∈ [0,1]: 1 = at surface (shallow), 0 = at optical limit (deep)
        prior = depth_norm.clamp(0.0, 1.0)
        return F.mse_loss(probs, prior)


class SmoothnessLoss(nn.Module):
    """Total-variation smoothness penalty on the predicted logit map."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        diff_h = (logits[:, :, 1:, :] - logits[:, :, :-1, :]).abs()
        diff_w = (logits[:, :, :, 1:] - logits[:, :, :, :-1]).abs()
        return diff_h.mean() + diff_w.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Composite wrappers
# ─────────────────────────────────────────────────────────────────────────────


class ReefSegmentationLoss(nn.Module):
    """
    Canonical composite loss for the reef segmentation pipeline:

        BCE(pos_weight) + Dice + λ_depth · DepthConsistency + λ_smooth · Smoothness

    Use this as the default in the UNet trainer. All sub-terms share the same
    logits-in / masks-in / depth_norm-in contract.

    Args:
        pos_weight:    class-imbalance weight (default 80 for the master dataset).
        lambda_depth:  weight for the depth-consistency term (default 0.3).
        lambda_smooth: weight for the TV smoothness term (default 0.1).
        from_dataset:  if not None, compute pos_weight from the on-disk masks
                       directory (overrides ``pos_weight``).

    Forward signature:
        loss = criterion(logits, masks, depth_norm=None)
    """

    def __init__(
        self,
        pos_weight: float = 80.0,
        lambda_depth: float = 0.3,
        lambda_smooth: float = 0.1,
        from_dataset: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._bce_dice = BCEDiceLoss(pos_weight=pos_weight, from_dataset=from_dataset)
        self._depth = DepthConsistencyLoss()
        self._smooth = SmoothnessLoss()
        self.lambda_depth = float(lambda_depth)
        self.lambda_smooth = float(lambda_smooth)

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        depth_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._bce_dice(logits, masks)

        if depth_norm is not None and self.lambda_depth > 0:
            loss = loss + self.lambda_depth * self._depth(logits, depth_norm)

        if self.lambda_smooth > 0:
            loss = loss + self.lambda_smooth * self._smooth(logits)

        return loss

    @property
    def pos_weight(self) -> float:
        return self._bce_dice.pos_weight


class PhysicsInformedLoss(nn.Module):
    """
    Backwards-compatible composite loss (kept for older checkpoints that
    expect the original PhysicsInformedLoss signature).

    For new training runs prefer ``ReefSegmentationLoss`` — same math, but
    it routes through ``BCEDiceLoss`` so ``pos_weight`` is honored.

    Args:
        lambda_depth:  Weight for the depth-consistency term (default 0.3).
        lambda_smooth: Weight for the smoothness/TV term (default 0.1).
        pos_weight:    Class-imbalance weight (default 80).

    Forward signature:
        loss = criterion(logits, masks, depth_norm=None)
    """

    def __init__(
        self,
        lambda_depth: float = 0.3,
        lambda_smooth: float = 0.1,
        pos_weight: float = 80.0,
    ) -> None:
        super().__init__()
        self.lambda_depth = float(lambda_depth)
        self.lambda_smooth = float(lambda_smooth)
        self._bce_dice = BCEDiceLoss(pos_weight=pos_weight)
        self._depth = DepthConsistencyLoss()
        self._smooth = SmoothnessLoss()

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        depth_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._bce_dice(logits, masks)

        if depth_norm is not None and self.lambda_depth > 0:
            loss = loss + self.lambda_depth * self._depth(logits, depth_norm)

        if self.lambda_smooth > 0:
            loss = loss + self.lambda_smooth * self._smooth(logits)

        return loss

    @property
    def pos_weight(self) -> float:
        return self._bce_dice.pos_weight
