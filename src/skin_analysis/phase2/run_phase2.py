"""CLI entry point for Phase 2 deep learning pipeline.

Usage
-----
    # Full pipeline (train + evaluate + Grad-CAM)
    python3 -m src.skin_analysis.phase2.run_phase2 \\
        --data-dir "data/raw/Multi-Class Skin Condition Image Dataset (MSC-6)" \\
        --output-dir outputs/phase2_deep_learning \\
        --run-gradcam

    # With explicit device override
    python3 -m src.skin_analysis.phase2.run_phase2 --device cuda

    # Run ablation study
    python3 -m src.skin_analysis.phase2.run_phase2 --run-ablation
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

from .augment import (
    SkinConditionDataset,
    build_dataloaders,
    build_val_transforms,
    visualize_augmented_samples,
)
from .config import Phase2Config
from .evaluate_phase2 import evaluate_model, predict, save_confusion_matrix, save_classification_report
from .gradcam import generate_gradcam_overlays
from .model import EfficientNetB3CBAM
from .train import detect_device, train_phase2


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_banner(cfg: Phase2Config, device: torch.device) -> None:
    """Print a formatted banner with configuration details."""
    print("=" * 70)
    print("  Phase 2: EfficientNet-B3 + CBAM — Skin Condition Classification")
    print("=" * 70)
    print(f"  Backbone     : {cfg.backbone}")
    print(f"  Image size   : {cfg.image_size}×{cfg.image_size}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Stage 1      : {cfg.epochs_stage1} epochs, lr={cfg.lr_stage1}")
    print(f"  Stage 2      : {cfg.epochs_stage2} epochs, lr_bb={cfg.lr_stage2_backbone}, lr_head={cfg.lr_stage2_head}")
    print(f"  MixUp α      : {cfg.mixup_alpha}")
    print(f"  RandAugment  : M={cfg.randaug_magnitude}, N={cfg.randaug_num_ops}")
    print(f"  Device       : {device}")
    print(f"  Data dir     : {cfg.data_dir}")
    print(f"  Output dir   : {cfg.output_dir}")
    print("=" * 70)


def run_phase2_pipeline(cfg: Phase2Config, device_override: str | None = None,
                        run_gradcam: bool = True, run_ablation: bool = False) -> None:
    """Execute the full Phase 2 pipeline: train → evaluate → Grad-CAM."""
    _set_seed(cfg.random_state)
    device = detect_device(device_override)
    _print_banner(cfg, device)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n╔══ Loading datasets ══╗")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(cfg)
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    # Class distribution
    labels = train_dataset.get_labels()
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    for idx, name in enumerate(cfg.class_names):
        count = class_counts.get(idx, 0)
        print(f"    {name:>12}: {count:>5}")

    # Save augmentation visualisation
    print("\n  Generating augmented samples visualisation…")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    visualize_augmented_samples(
        train_dataset, cfg.display_names,
        cfg.output_dir / "augmented_samples.png",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n╔══ Building model ══╗")
    model = EfficientNetB3CBAM(
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        cbam_reduction=cfg.cbam_reduction,
        pretrained=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    # ── Training ──────────────────────────────────────────────────────────
    t0 = time.time()
    history = train_phase2(model, train_loader, val_loader, cfg, device)
    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed / 60:.1f} minutes")

    # ── Evaluation ────────────────────────────────────────────────────────
    model = model.to(device)
    report = evaluate_model(model, val_loader, history, cfg, device)

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    if run_gradcam:
        print("\n╔══ Grad-CAM Visualisation ══╗")
        generate_gradcam_overlays(model, val_dataset, cfg, device)

    # ── Ablation study ────────────────────────────────────────────────────
    if run_ablation:
        _run_ablation_study(cfg, train_loader, val_loader, device)

    print("\n" + "=" * 70)
    print("  Phase 2 pipeline complete!")
    print(f"  All outputs saved to: {cfg.output_dir}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
#  Ablation study
# ═══════════════════════════════════════════════════════════════════════════


def _run_ablation_study(
    cfg: Phase2Config,
    train_loader,
    val_loader,
    device: torch.device,
) -> None:
    """Train and evaluate 3 ablation variants.

    Variant A: Frozen backbone, no CBAM (head only)
    Variant B: Full fine-tuning, no CBAM
    Variant C: Full model (EfficientNet-B3 + CBAM) — this is the main model
    """
    from .train import _validate

    print("\n╔══ Ablation Study ══╗")
    results = []

    # Variant A: head only, no CBAM
    print("\n  ── Variant A: Frozen backbone, no CBAM ──")
    model_a = EfficientNetB3CBAM(
        num_classes=cfg.num_classes, hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout, pretrained=True,
    )
    # Replace CBAM with identity
    model_a.cbam = torch.nn.Identity()
    model_a.freeze_backbone()
    model_a = model_a.to(device)

    optimizer_a = torch.optim.Adam(
        list(model_a.classifier.parameters()), lr=cfg.lr_stage1,
    )
    from .train import _train_one_epoch
    for epoch in range(1, min(cfg.epochs_stage1, 10) + 1):
        _train_one_epoch(model_a, train_loader, optimizer_a, device,
                         use_mixup=True, label_smoothing=cfg.label_smoothing,
                         num_classes=cfg.num_classes)
    _, acc_a, f1_a = _validate(model_a, val_loader, device, cfg.num_classes)
    results.append(("A: Frozen, no CBAM", acc_a, f1_a))
    print(f"    Accuracy: {acc_a:.4f}  F1: {f1_a:.4f}")

    # Variant B: fine-tuned, no CBAM
    print("\n  ── Variant B: Fine-tuned, no CBAM ──")
    model_b = EfficientNetB3CBAM(
        num_classes=cfg.num_classes, hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout, pretrained=True,
    )
    model_b.cbam = torch.nn.Identity()
    model_b = model_b.to(device)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=cfg.lr_stage2_head)
    for epoch in range(1, min(cfg.epochs_stage2, 10) + 1):
        _train_one_epoch(model_b, train_loader, optimizer_b, device,
                         use_mixup=True, label_smoothing=cfg.label_smoothing,
                         num_classes=cfg.num_classes)
    _, acc_b, f1_b = _validate(model_b, val_loader, device, cfg.num_classes)
    results.append(("B: Fine-tuned, no CBAM", acc_b, f1_b))
    print(f"    Accuracy: {acc_b:.4f}  F1: {f1_b:.4f}")

    # Variant C: main model (already trained) — we just load it
    print("\n  ── Variant C: EfficientNet-B3 + CBAM (main model) ──")
    model_c = EfficientNetB3CBAM(
        num_classes=cfg.num_classes, hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout, pretrained=False,
    )
    checkpoint = cfg.output_dir / "best_model_phase2.pth"
    if checkpoint.exists():
        model_c.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model_c = model_c.to(device)
        _, acc_c, f1_c = _validate(model_c, val_loader, device, cfg.num_classes)
    else:
        acc_c, f1_c = 0.0, 0.0
        print("    ⚠ No checkpoint found, skipping")
    results.append(("C: EfficientNet-B3 + CBAM", acc_c, f1_c))
    print(f"    Accuracy: {acc_c:.4f}  F1: {f1_c:.4f}")

    # Save results
    csv_path = cfg.output_dir / "ablation_study.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "Accuracy", "F1 (weighted)"])
        for name, acc, f1 in results:
            writer.writerow([name, f"{acc:.4f}", f"{f1:.4f}"])
    print(f"\n  Ablation results saved to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Parse CLI arguments and run the Phase 2 pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 2: EfficientNet-B3 + CBAM skin-condition classification",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="data/raw/Multi-Class Skin Condition Image Dataset (MSC-6)",
        help="Path to the MSC-6 dataset root",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/phase2_deep_learning",
        help="Directory for output artefacts",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (mps/cuda/cpu)")
    parser.add_argument("--run-gradcam", action="store_true", help="Generate Grad-CAM overlays")
    parser.add_argument("--run-ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs-s1", type=int, default=None, help="Override Stage 1 epochs")
    parser.add_argument("--epochs-s2", type=int, default=None, help="Override Stage 2 epochs")

    args = parser.parse_args()

    cfg = Phase2Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs_s1:
        cfg.epochs_stage1 = args.epochs_s1
    if args.epochs_s2:
        cfg.epochs_stage2 = args.epochs_s2

    run_phase2_pipeline(cfg, device_override=args.device,
                        run_gradcam=args.run_gradcam, run_ablation=args.run_ablation)


if __name__ == "__main__":
    main()
