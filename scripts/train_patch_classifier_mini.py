#!/usr/bin/env python3
"""
train_patch_classifier_mini.py — parallel trainer for the reef patch classifier.

This is a *side-by-side* counterpart to ``scripts/train_reef_classifier.py``.
It reuses the model, dataset, training loop, and evaluation routine from
``src/reef_classifier.py`` unchanged, but parameterises the patch and model
output paths so that several experiments can coexist without ever touching the
baseline artefacts in ``outputs/reef_habitat/``.

Safety
------
* The baseline ``outputs/reef_habitat/model/reef_cnn_best.pth`` is NEVER
  written to or overwritten by this script.
* ``src/reef_classifier.py`` is imported, never modified.
* All artefacts produced by this script live under ``--model-root``, which
  defaults to ``outputs/reef_habitat_mini/model``.

Inputs (default patch layout)
-----------------------------
::

    --patches-root/
        train/
            reef/  patch_*.npy
            sand/  patch_*.npy
        val/
            reef/  patch_*.npy
            sand/  patch_*.npy

Outputs (under --model-root)
----------------------------
* ``reef_cnn_best.pth``     — best checkpoint by validation AUC
* ``training_log.json``     — per-epoch metrics (written by ``src.reef_classifier.train``)
* ``evaluation.json``       — final validation metrics + provenance

Example
-------
::

    python scripts/train_patch_classifier_mini.py \\
        --patches-root outputs/reef_habitat_mini/patches \\
        --model-root   outputs/reef_habitat_mini/model \\
        --epochs 30 --batch-size 16

Multiple experiments can run side by side by varying ``--model-root``::

    python scripts/train_patch_classifier_mini.py --model-root outputs/reef_habitat_mini/model_sceneA
    python scripts/train_patch_classifier_mini.py --model-root outputs/reef_habitat_mini/model_sceneB

Notes
-----
``--no-augment`` is accepted for reproducibility bookkeeping but the underlying
``src.reef_classifier.train()`` hardcodes augmentation for the training split.
The flag value is recorded in ``evaluation.json``; a warning is logged when set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

print("SAFE MODE CHECK: I will only create new mini-files and will not modify existing files.")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _resolve_device(device_arg: str) -> str:
    """
    Resolve the ``--device`` argument.

    ``"auto"`` returns ``"cuda"`` if available, otherwise ``"cpu"``.
    Any other value is passed through verbatim (e.g. ``"mps"`` on Apple Silicon).
    """
    import torch

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _check_dataset(patches_root: Path) -> bool:
    """Return True iff every required ``{train,val}/{reef,sand}`` directory contains .npy files."""
    for split in ("train", "val"):
        for cls in ("reef", "sand"):
            d = patches_root / split / cls
            if not d.exists() or not any(d.glob("*.npy")):
                log.error("Missing patches in %s — run build_reef_dataset_mini.py first", d)
                return False
    return True


def _count_patches(patches_root: Path) -> dict[str, int]:
    """Count .npy files per ``{split}_{class}`` bucket for logging."""
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        for cls in ("reef", "sand"):
            d = patches_root / split / cls
            counts[f"{split}_{cls}"] = len(list(d.glob("*.npy"))) if d.exists() else 0
    return counts


def run_training(args: argparse.Namespace) -> None:
    """Train (or only evaluate) the reef classifier and write artefacts under ``--model-root``."""
    from src.reef_classifier import ReefClassifier, evaluate, load_model, train

    import torch

    patches_root: Path = Path(args.patches_root)
    model_root: Path = Path(args.model_root)
    train_dir: Path = patches_root / "train"
    val_dir: Path = patches_root / "val"
    best_path: Path = model_root / "reef_cnn_best.pth"
    eval_path: Path = model_root / "evaluation.json"
    log_path: Path = model_root / "training_log.json"

    model_root.mkdir(parents=True, exist_ok=True)

    if not _check_dataset(patches_root):
        sys.exit(1)

    counts = _count_patches(patches_root)
    log.info(
        "Dataset: train reef=%d sand=%d  |  val reef=%d sand=%d",
        counts["train_reef"], counts["train_sand"],
        counts["val_reef"], counts["val_sand"],
    )

    device = _resolve_device(args.device)
    log.info("Device requested: %s  (resolved to %s)", args.device, device)

    if args.no_augment:
        log.warning(
            "--no-augment set: src.reef_classifier.train() currently hardcodes "
            "augment=True for the training split, so this flag is recorded for "
            "reproducibility only and does NOT change the augmentation behaviour."
        )

    if args.eval_only:
        if not best_path.exists():
            log.error(
                "No checkpoint at %s — train first (drop --eval-only) or "
                "point --model-root at a directory produced by a previous run.",
                best_path,
            )
            sys.exit(1)
        log.info("Eval-only mode: loading checkpoint %s", best_path)
        model = load_model(best_path, device=device)
    else:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        log.info("Seed: %d", args.seed)

        model = ReefClassifier(pretrained=True, dropout=args.dropout)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "Model: ResNet-18 4-channel  (%d params / %.1f MB)",
            n_params, n_params * 4 / 1e6,
        )

        history = train(
            model,
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=model_root,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
        )

        if history:
            best_epoch = max(history, key=lambda h: h["val_auc"])
            print("\n── Training summary ──────────────────────────────────────")
            print(f"  Best epoch  : {best_epoch['epoch']}")
            print(f"  Best val AUC: {best_epoch['val_auc']:.4f}")
            print(f"  Val accuracy: {best_epoch['val_acc']:.4f}")
            print(f"  Checkpoint  : {best_path}")
            print("──────────────────────────────────────────────────────────")

        model = load_model(best_path, device=device)

    log.info("Evaluating on validation set…")
    metrics = evaluate(model, val_dir, threshold=args.threshold, device=device)

    provenance = {
        "seed": args.seed,
        "device": device,
        "patches_root": str(patches_root),
        "model_root": str(model_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "dropout": args.dropout,
        "threshold": args.threshold,
        "no_augment_requested": bool(args.no_augment),
        "eval_only": bool(args.eval_only),
        "training_log": str(log_path),
        "checkpoint": str(best_path),
    }
    metrics_with_provenance = {**metrics, "provenance": provenance}

    with open(eval_path, "w") as f:
        json.dump(metrics_with_provenance, f, indent=2)

    print("\n── Evaluation results ────────────────────────────────────────")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"  Saved to  : {eval_path}")
    print(f"  Checkpoint: {best_path}")
    print(f"  Model root: {model_root}")
    print("  Baseline  : outputs/reef_habitat/model/ (untouched)")
    print("──────────────────────────────────────────────────────────────")


def main() -> None:
    """Parse CLI arguments and dispatch to :func:`run_training`."""
    parser = argparse.ArgumentParser(
        description=(
            "Train ReefClassifier (ResNet-18, 4-band S2) on an arbitrary "
            "patches/ directory. Reuses src.reef_classifier.py unchanged."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patches-root", type=str, default="outputs/reef_habitat_mini/patches",
        help="Root containing {train,val}/{reef,sand}/*.npy",
    )
    parser.add_argument(
        "--model-root", type=str, default="outputs/reef_habitat_mini/model",
        help="Directory for reef_cnn_best.pth, training_log.json, evaluation.json",
    )
    parser.add_argument("--epochs",       type=int,   default=30,   help="Max training epochs")
    parser.add_argument("--batch-size",   type=int,   default=16,   help="Mini-batch size")
    parser.add_argument("--lr",           type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--patience",     type=int,   default=8,    help="Early stopping patience (epochs)")
    parser.add_argument("--dropout",      type=float, default=0.3,  help="Dropout in classifier head")
    parser.add_argument("--threshold",    type=float, default=0.5,  help="Binary prediction threshold")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device: 'auto' | 'cuda' | 'cpu' | 'mps' | etc.",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help=(
            "Record intent to disable augmentation. NOTE: src.reef_classifier.train() "
            "currently always augments; flag is logged and saved to evaluation.json "
            "for reproducibility checks only."
        ),
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, load existing --model-root/reef_cnn_best.pth and evaluate only",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (ignored under --eval-only)")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
