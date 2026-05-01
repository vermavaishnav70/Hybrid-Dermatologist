"""Two-stage training loop for the Hybrid Fusion Model (Model C).

Training strategy
-----------------
* **Stage 1 — Fusion head only (CNN backbone frozen)**:
    Only the attention gate, stream projections, and classifier are optimised.
    The CNN backbone retains its Phase 2 fine-tuned features.
    Adam with lr=1e-3, CosineAnnealingWarmRestarts scheduler.
    CrossEntropyLoss with class weights and label smoothing=0.1.

* **Stage 2 — End-to-end fine-tuning**:
    Unfreezes the last 3 EfficientNet blocks + CBAM.
    Discriminative learning rates: backbone at 2e-5, fusion at 1e-4.
    Linear warmup for 3 epochs -> cosine decay.
    Early stopping on validation F1 with patience=7.

Optimizer choice
~~~~~~~~~~~~~~~~
Adam is preferred over SGD because:
    1. Per-parameter adaptive learning rates handle the mixed-magnitude
       gradients from frozen vs unfrozen blocks and from the two very
       different feature streams (CNN ~0-10, ML ~0-1).
    2. The cosine warmup schedule prevents early large updates from
       destabilising the pretrained Phase 2 weights.

Why F1 macro over accuracy for model selection:
    Class distribution: eczema(~1800) vs dark_spots(~250) = ~7x imbalance.
    Accuracy is dominated by majority class.  F1 macro weights all classes
    equally regardless of support.  Standard for imbalanced classification.
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from .config import Phase3Config
from .model_c import HybridFusionModel


# ═══════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Why deterministic mode: ensures identical results across runs,
    which is critical for ablation studies where we compare variants.
    torch.backends.cudnn.deterministic = True trades speed for
    reproducibility in convolution operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════
#  Device detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_device(override: str | None = None) -> torch.device:
    """Auto-detect the best available device: cuda -> mps -> cpu."""
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════
#  Cosine warmup scheduler
# ═══════════════════════════════════════════════════════════════════════════


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_epochs then cosine decay to zero.

    Warmup: LR(t) = LR_max * (t / T_warmup)
    Cosine: LR(t) = LR_max * 0.5 * (1 + cos(pi * (t - T_w) / (T_total - T_w)))

    Why warmup: prevents the randomly-initialised fusion layers from
    generating large gradients that corrupt the pretrained backbone.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
#  Single-epoch routines
# ═══════════════════════════════════════════════════════════════════════════


def _train_one_epoch(
    model: HybridFusionModel,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, ml_features, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        ml_features = ml_features.to(device)
        labels = labels.to(device)

        logits = model(images, ml_features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients in the fusion layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def _validate(
    model: HybridFusionModel,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Run validation. Returns (loss, accuracy, f1_weighted, f1_macro)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, ml_features, labels in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        ml_features = ml_features.to(device)
        labels = labels.to(device)

        logits = model(images, ml_features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(
        len(all_labels), 1
    )
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1_w, f1_m


# ═══════════════════════════════════════════════════════════════════════════
#  Full two-stage training
# ═══════════════════════════════════════════════════════════════════════════


def train_hybrid(
    model: HybridFusionModel,
    train_loader,
    val_loader,
    cfg: Phase3Config,
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    """Run the full two-stage hybrid training. Returns per-epoch history.

    Returns a list of dicts with keys:
        stage, epoch, train_loss, train_acc, val_loss, val_acc,
        val_f1_weighted, val_f1_macro, lr
    """
    if device is None:
        device = detect_device()
    model = model.to(device)

    # ── Loss function with class weights ──────────────────────────────────
    # Why BOTH sampler AND weighted loss: double defence against imbalance.
    # Sampler ensures balanced mini-batches, weighted loss further penalises
    # misclassification of minority classes within each batch.
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.label_smoothing,
    )

    history: list[dict] = []
    best_f1 = 0.0
    best_state = None

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_hybrid.pth"

    # ── Check for existing checkpoint ─────────────────────────────────────
    start_epoch = 1
    start_stage = 1
    if checkpoint_path.exists():
        print(f"\n  ↻ Found checkpoint at {checkpoint_path}. Resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_f1 = ckpt.get("best_f1", 0.0)
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        start_stage = ckpt.get("stage", 1)
        print(
            f"    Resuming from Stage {start_stage}, "
            f"Epoch {start_epoch} (Best F1: {best_f1:.4f})"
        )

    # ── Stage 1: Fusion head only ─────────────────────────────────────────
    if start_stage == 1:
        print("\n╔══ Stage 1: Training fusion layers (CNN backbone frozen) ══╗")
        model.freeze_backbone()

        # Only train fusion-related parameters
        fusion_params = (
            list(model.cnn_projection.parameters())
            + list(model.ml_projection.parameters())
            + list(model.attention_gate.parameters())
            + list(model.classifier.parameters())
        )
        optimizer = torch.optim.Adam(fusion_params, lr=cfg.lr_stage1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5
        )

        # Restore optimizer/scheduler state if resuming within Stage 1
        if checkpoint_path.exists() and start_epoch > 1:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "optimizer_state_dict" in ckpt and ckpt.get("stage") == 1:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"    Restored optimizer state from checkpoint")
            if "scheduler_state_dict" in ckpt and ckpt.get("stage") == 1:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print(f"    Restored scheduler state from checkpoint")

        for epoch in range(start_epoch, cfg.epochs_stage1 + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\n  Epoch {epoch}/{cfg.epochs_stage1}  (lr={current_lr:.6f})")

            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, val_f1_w, val_f1_m = _validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            row = {
                "stage": 1,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_weighted": val_f1_w,
                "val_f1_macro": val_f1_m,
                "lr": current_lr,
            }
            history.append(row)
            print(
                f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"val_f1_w={val_f1_w:.4f}  val_f1_m={val_f1_m:.4f}"
            )

            if val_f1_w > best_f1:
                best_f1 = val_f1_w
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, output_dir / "best_model_hybrid.pth")
                print(f"  ✓ New best val_f1={best_f1:.4f}")

            # Save checkpoint every epoch (crash-safe: optimizer + scheduler included)
            torch.save(
                {
                    "stage": 1,
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "history": history,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path,
            )
            print(f"  💾 Checkpoint saved (Stage 1, Epoch {epoch})")

        start_epoch = 1
        start_stage = 2

    # ── Stage 2: End-to-end fine-tuning ───────────────────────────────────
    if start_stage == 2:
        print(f"\n╔══ Stage 2: Fine-tuning last {cfg.unfreeze_blocks} blocks ══╗")
        model.unfreeze_last_n_blocks(n=cfg.unfreeze_blocks)

        param_groups = model.get_trainable_param_groups(
            lr_backbone=cfg.lr_stage2_backbone,
            lr_head=cfg.lr_stage2_head,
        )
        optimizer = torch.optim.Adam(param_groups)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.epochs_stage2,
        )

        # Restore optimizer/scheduler state if resuming within Stage 2
        if checkpoint_path.exists() and start_epoch > 1:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "optimizer_state_dict" in ckpt and ckpt.get("stage") == 2:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"    Restored optimizer state from checkpoint")
            if "scheduler_state_dict" in ckpt and ckpt.get("stage") == 2:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print(f"    Restored scheduler state from checkpoint")
            else:
                # Catch up scheduler if no scheduler state saved
                for _ in range(start_epoch - 1):
                    scheduler.step()
        elif start_epoch > 1:
            # Catch up scheduler if resuming without checkpoint file
            for _ in range(start_epoch - 1):
                scheduler.step()

        patience_counter = 0
        # Restore patience counter from checkpoint if available
        if checkpoint_path.exists():
            ckpt_check = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            patience_counter = ckpt_check.get("patience_counter", 0)

        for epoch in range(start_epoch, cfg.epochs_stage2 + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"\n  Epoch {epoch}/{cfg.epochs_stage2}  "
                f"(lr_backbone={current_lr:.7f})"
            )

            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, val_f1_w, val_f1_m = _validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            row = {
                "stage": 2,
                "epoch": cfg.epochs_stage1 + epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_weighted": val_f1_w,
                "val_f1_macro": val_f1_m,
                "lr": current_lr,
            }
            history.append(row)
            print(
                f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"val_f1_w={val_f1_w:.4f}  val_f1_m={val_f1_m:.4f}"
            )

            if val_f1_w > best_f1:
                best_f1 = val_f1_w
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, output_dir / "best_model_hybrid.pth")
                patience_counter = 0
                print(f"  ✓ New best val_f1={best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(
                        f"  ✗ Early stopping triggered "
                        f"(patience={cfg.early_stopping_patience})"
                    )
                    break

            # Save checkpoint every epoch (crash-safe: optimizer + scheduler included)
            torch.save(
                {
                    "stage": 2,
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "history": history,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "patience_counter": patience_counter,
                },
                checkpoint_path,
            )
            print(f"  💾 Checkpoint saved (Stage 2, Epoch {epoch})")

    # ── Restore best model ────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  ✓ Best model restored (val_f1={best_f1:.4f})")

    # ── Save training history ─────────────────────────────────────────────
    history_path = output_dir / "training_history.csv"
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"  Training history saved to {history_path}")

    # ── Clean up checkpoint after successful completion ────────────────────
    # Keep the checkpoint so users can verify the final state
    print(f"  💾 Final checkpoint available at: {checkpoint_path}")
    print(f"  💾 Best model weights at: {output_dir / 'best_model_hybrid.pth'}")

    return history


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Train the Hybrid Fusion Model from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3: Hybrid Attention-Weighted Fusion Model"
    )
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs-s1", type=int, default=None)
    parser.add_argument("--epochs-s2", type=int, default=None)
    args = parser.parse_args()

    seed_everything(42)

    cfg = Phase3Config()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs_s1:
        cfg.epochs_stage1 = args.epochs_s1
    if args.epochs_s2:
        cfg.epochs_stage2 = args.epochs_s2

    device = detect_device(args.device)

    print("=" * 70)
    print("  Phase 3: Hybrid Attention-Weighted Fusion Model")
    print("=" * 70)
    print(f"  CNN features  : {cfg.cnn_feature_dim} dims (EfficientNet-B3 + CBAM)")
    print(f"  ML features   : {cfg.ml_feature_dim} dims (color + LBP + GLCM)")
    print(f"  Fusion dim    : {cfg.fusion_hidden_dim}")
    print(f"  Stage 1       : {cfg.epochs_stage1} epochs, lr={cfg.lr_stage1}")
    print(f"  Stage 2       : {cfg.epochs_stage2} epochs, lr={cfg.lr_stage2_head}")
    print(f"  Device        : {device}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    from .dataset import build_hybrid_dataloaders, compute_class_weights

    print("\n╔══ Loading datasets ══╗")
    train_loader, val_loader, train_dataset, val_dataset = build_hybrid_dataloaders(cfg)
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    class_weights = compute_class_weights(train_dataset)
    print(f"  Class weights: {class_weights.tolist()}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n╔══ Building Hybrid Fusion Model ══╗")
    model = HybridFusionModel.from_phase2_checkpoint(
        checkpoint_path=cfg.phase2_checkpoint,
        num_classes=cfg.num_classes,
        ml_feature_dim=cfg.ml_feature_dim,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        dropout=cfg.dropout,
        cbam_reduction=cfg.cbam_reduction,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    # ── Training ──────────────────────────────────────────────────────────
    t0 = time.time()
    history = train_hybrid(
        model, train_loader, val_loader, cfg,
        class_weights=class_weights, device=device,
    )
    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed / 60:.1f} minutes")

    print("\n" + "=" * 70)
    print("  Phase 3 training complete!")
    print(f"  Outputs saved to: {cfg.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
