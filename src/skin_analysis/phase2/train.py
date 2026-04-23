"""Two-stage training loop for EfficientNet-B3 + CBAM.

Training strategy
-----------------
* **Stage 1 — Head only (frozen backbone)**:
    - Only the CBAM + classifier parameters are optimised.
    - Adam with lr=1e-3, CosineAnnealingWarmRestarts scheduler.
    - MixUp soft-label CE loss (when MixUp is active) or standard CE with
      label smoothing=0.1.
    - BatchNorm in the backbone stays in eval mode to preserve ImageNet
      running statistics.

* **Stage 2 — Fine-tune last N blocks**:
    - Unfreezes the last 3 EfficientNet blocks.
    - Discriminative learning rates: backbone unfrozen blocks at 2e-5,
      CBAM + head at 1e-4.
    - Linear warmup for 3 epochs → cosine decay.
    - Early stopping on validation F1 with patience=7.

Optimizer choice
~~~~~~~~~~~~~~~~
Adam is preferred over SGD because:
    1. Per-parameter adaptive learning rates handle the mixed-magnitude
       gradients from frozen vs. unfrozen blocks.
    2. The cosine warmup schedule (LR_t = LR_max × t / T_warmup for
       t < T_warmup, then cosine decay) prevents early large updates from
       destabilising pretrained weights.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from .config import Phase2Config
from .model import EfficientNetB3CBAM


# ═══════════════════════════════════════════════════════════════════════════
#  Loss helpers
# ═══════════════════════════════════════════════════════════════════════════


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss supporting soft (MixUp) labels.

    When targets are one-hot soft distributions we compute:
        loss = -sum(targets * log_softmax(logits)) / batch_size
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  Device detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_device(override: str | None = None) -> torch.device:
    """Auto-detect the best available device: mps → cuda → cpu.
    If torch_xla is installed, it returns the TPU device.
    """
    if override is not None:
        return torch.device(override)
    if HAS_XLA:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════
#  Cosine warmup scheduler
# ═══════════════════════════════════════════════════════════════════════════


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for ``warmup_epochs`` then cosine decay to zero.

    Warmup formula: LR(t) = LR_max × (t / T_warmup)
    Cosine formula: LR(t) = LR_max × 0.5 × (1 + cos(π × (t - T_warmup) / (T_total - T_warmup)))
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
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
#  Single-epoch routines
# ═══════════════════════════════════════════════════════════════════════════


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixup: bool,
    label_smoothing: float,
    num_classes: int,
) -> tuple[float, float]:
    """Run one training epoch and return (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    if HAS_XLA:
        xla_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    else:
        xla_loader = loader

    for batch in tqdm(xla_loader, total=len(loader), desc="  train", leave=False):
        if use_mixup:
            images, soft_targets, _ = batch
            images = images.to(device)
            soft_targets = soft_targets.to(device)
            logits = model(images)
            loss = soft_cross_entropy(logits, soft_targets)
            # Accuracy: compare argmax of prediction vs argmax of soft target
            preds = logits.argmax(dim=1)
            labels = soft_targets.argmax(dim=1)
        else:
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            preds = logits.argmax(dim=1)

        optimizer.zero_grad()
        loss.backward()
        
        if HAS_XLA:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float]:
    """Run validation and return (loss, accuracy, weighted_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    if HAS_XLA:
        xla_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    else:
        xla_loader = loader

    for images, labels, _ in tqdm(xla_loader, total=len(loader), desc="  val  ", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, accuracy, f1


# ═══════════════════════════════════════════════════════════════════════════
#  Full two-stage training
# ═══════════════════════════════════════════════════════════════════════════


def train_phase2(
    model: EfficientNetB3CBAM,
    train_loader,
    val_loader,
    cfg: Phase2Config,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    """Run the full two-stage training and return per-epoch history.

    Returns a list of dicts with keys:
        stage, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr
    """
    if device is None:
        device = detect_device()
    model = model.to(device)
    history: list[dict[str, float]] = []
    best_f1 = 0.0
    best_state = None
    start_epoch = 1
    start_stage = 1

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_phase2.pth"

    # ── Check for existing checkpoint ─────────────────────────────────────
    if checkpoint_path.exists():
        print(f"\n  ↻ Found checkpoint at {checkpoint_path}. Resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_f1 = ckpt.get("best_f1", 0.0)
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        start_stage = ckpt.get("stage", 1)
        print(f"    Resuming from Stage {start_stage}, Epoch {start_epoch} (Best F1: {best_f1:.4f})")

    # ── Stage 1: Head only ────────────────────────────────────────────────
    if start_stage == 1:
        print("\n╔══ Stage 1: Training CBAM + classifier head (backbone frozen) ══╗")
        model.freeze_backbone()
        head_params = list(model.cbam.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.Adam(head_params, lr=cfg.lr_stage1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

        # Handle resumption within Stage 1
        if start_epoch > 1:
            # Catch up scheduler and optimizer if we had them saved (simplified here for stage 1)
            pass

        for epoch in range(start_epoch, cfg.epochs_stage1 + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\n  Epoch {epoch}/{cfg.epochs_stage1}  (lr={current_lr:.6f})")

            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, device,
                use_mixup=True, label_smoothing=cfg.label_smoothing,
                num_classes=cfg.num_classes,
            )
            val_loss, val_acc, val_f1 = _validate(model, val_loader, device, cfg.num_classes)
            scheduler.step()

            row = {
                "stage": 1, "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "lr": current_lr,
            }
            history.append(row)
            print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, output_dir / "best_model_phase2.pth")
            
            # Save checkpoint
            torch.save({
                "stage": 1, "epoch": epoch, "best_f1": best_f1, "history": history,
                "model_state_dict": model.state_dict(),
            }, checkpoint_path)

        # Transition to Stage 2 start
        start_epoch = 1
        start_stage = 2

    # ── Stage 2: Fine-tune last N blocks ──────────────────────────────────
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

        # If resuming in stage 2, catch up scheduler
        if start_epoch > 1:
            for _ in range(start_epoch - 1):
                scheduler.step()

        patience_counter = 0
        for epoch in range(start_epoch, cfg.epochs_stage2 + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\n  Epoch {epoch}/{cfg.epochs_stage2}  (lr_backbone={current_lr:.7f})")

            train_loss, train_acc = _train_one_epoch(
                model, train_loader, optimizer, device,
                use_mixup=True, label_smoothing=cfg.label_smoothing,
                num_classes=cfg.num_classes,
            )
            val_loss, val_acc, val_f1 = _validate(model, val_loader, device, cfg.num_classes)
            scheduler.step()

            row = {
                "stage": 2, "epoch": cfg.epochs_stage1 + epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "lr": current_lr,
            }
            history.append(row)
            print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, output_dir / "best_model_phase2.pth")
                patience_counter = 0
                print(f"  ✓ New best val_f1={best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"  ✗ Early stopping triggered (patience={cfg.early_stopping_patience})")
                    break
            
            # Save checkpoint
            torch.save({
                "stage": 2, "epoch": epoch, "best_f1": best_f1, "history": history,
                "model_state_dict": model.state_dict(),
            }, checkpoint_path)

    # ── Save best model ───────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        checkpoint_path = output_dir / "best_model_phase2.pth"
        torch.save(best_state, checkpoint_path)
        print(f"\n  Best model saved to {checkpoint_path} (val_f1={best_f1:.4f})")

    # ── Save training history ─────────────────────────────────────────────
    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"  Training history saved to {history_path}")

    return history
