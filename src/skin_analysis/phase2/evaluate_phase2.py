"""Evaluation suite for Phase 2, including ablation study and Phase 1 comparison.

Generates all artefacts required for the project report:
    * Confusion matrix (CSV + heatmap PNG)
    * Classification report (per-class precision / recall / F1)
    * Baseline comparison table (Phase 2 vs Phase 1 RF)
    * Learning curves with overfitting analysis
    * Ablation study (3 variants)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from .config import Phase2Config
from .model import EfficientNetB3CBAM


# ═══════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def predict(
    model: EfficientNetB3CBAM,
    loader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """Run inference and return (predictions, labels, confidences)."""
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    for batch in tqdm(loader, desc="  predict", leave=False):
        images, labels, _ = batch
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels)
        all_confs.extend(confs.cpu().tolist())
    return all_preds, all_labels, all_confs


# ═══════════════════════════════════════════════════════════════════════════
#  Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: Sequence[str],
    output_dir: Path,
    prefix: str = "phase2",
) -> None:
    """Save confusion matrix as CSV + styled heatmap PNG."""
    cm = confusion_matrix(y_true, y_pred)
    display_names = list(class_names)

    # CSV
    csv_path = output_dir / f"confusion_matrix_{prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + display_names)
        for i, row in enumerate(cm):
            writer.writerow([display_names[i]] + row.tolist())

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {prefix.replace('_', ' ').title()}", fontsize=14, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    png_path = output_dir / f"confusion_matrix_{prefix}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {csv_path} and {png_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Classification report
# ═══════════════════════════════════════════════════════════════════════════


def save_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: Sequence[str],
    output_dir: Path,
    prefix: str = "phase2",
) -> dict:
    """Save sklearn classification report as CSV and return the metrics dict."""
    report = classification_report(
        y_true, y_pred,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    csv_path = output_dir / f"classification_report_{prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls_name in list(class_names) + ["macro avg", "weighted avg"]:
            if cls_name in report:
                r = report[cls_name]
                writer.writerow([cls_name, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                                 f"{r['f1-score']:.4f}", int(r.get("support", 0))])
        if "accuracy" in report:
            writer.writerow(["accuracy", "", "", f"{report['accuracy']:.4f}", ""])

    print(f"  Classification report saved to {csv_path}")
    return report


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline comparison
# ═══════════════════════════════════════════════════════════════════════════


def save_baseline_comparison(
    y_true: list[int],
    y_pred: list[int],
    output_dir: Path,
    phase1_accuracy: float = 0.809,
    phase1_f1: float = 0.806,
) -> None:
    """Save Phase 2 vs Phase 1 RF comparison table."""
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    csv_path = output_dir / "baseline_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Phase 1 (RF)", "Phase 2 (EfficientNet-B3 + CBAM)", "Improvement"])
        rows = [
            ("Accuracy", phase1_accuracy, acc),
            ("F1 (weighted)", phase1_f1, f1_w),
            ("F1 (macro)", None, f1_m),
            ("Precision (weighted)", None, prec),
            ("Recall (weighted)", None, rec),
        ]
        for name, p1, p2 in rows:
            p1_str = f"{p1:.4f}" if p1 is not None else "—"
            p2_str = f"{p2:.4f}"
            if p1 is not None:
                diff = p2 - p1
                imp_str = f"{diff:+.4f} ({diff / p1 * 100:+.1f}%)"
            else:
                imp_str = "—"
            writer.writerow([name, p1_str, p2_str, imp_str])

    print(f"  Baseline comparison saved to {csv_path}")
    print(f"  Phase 2 accuracy: {acc:.4f}  (Phase 1 RF: {phase1_accuracy:.4f})")
    print(f"  Phase 2 F1 (weighted): {f1_w:.4f}  (Phase 1 RF: {phase1_f1:.4f})")


# ═══════════════════════════════════════════════════════════════════════════
#  Learning curves
# ═══════════════════════════════════════════════════════════════════════════


def plot_learning_curves(
    history: list[dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot train/val loss and accuracy vs epoch with stage boundary."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] for r in history]
    val_acc = [r["val_acc"] for r in history]
    val_f1 = [r["val_f1"] for r in history]

    # Detect stage boundary
    stage_boundary = None
    for i, r in enumerate(history):
        if r["stage"] == 2 and (i == 0 or history[i - 1]["stage"] == 1):
            stage_boundary = r["epoch"]
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#e74c3c", markersize=3)
    axes[0].plot(epochs, val_loss, "s-", label="Val Loss", color="#3498db", markersize=3)
    if stage_boundary:
        axes[0].axvline(x=stage_boundary, color="gray", linestyle="--", alpha=0.7, label="Stage 2 start")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Epoch", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_acc, "o-", label="Train Acc", color="#e74c3c", markersize=3)
    axes[1].plot(epochs, val_acc, "s-", label="Val Acc", color="#3498db", markersize=3)
    if stage_boundary:
        axes[1].axvline(x=stage_boundary, color="gray", linestyle="--", alpha=0.7, label="Stage 2 start")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Epoch", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, val_f1, "D-", label="Val F1 (weighted)", color="#2ecc71", markersize=3)
    if stage_boundary:
        axes[2].axvline(x=stage_boundary, color="gray", linestyle="--", alpha=0.7, label="Stage 2 start")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Validation F1 vs Epoch", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Learning Curves — EfficientNet-B3 + CBAM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Learning curves saved to {path}")

    # Overfitting analysis
    if len(val_loss) > 5:
        min_val_idx = int(np.argmin(val_loss))
        if min_val_idx < len(val_loss) - 1:
            overfit_epoch = epochs[min_val_idx]
            print(f"  ⚠ Minimum val loss at epoch {overfit_epoch} — "
                  f"potential overfitting after this point.")


# ═══════════════════════════════════════════════════════════════════════════
#  Full evaluation
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: EfficientNetB3CBAM,
    val_loader,
    history: list[dict[str, float]],
    cfg: Phase2Config,
    device: torch.device,
) -> dict:
    """Run the full evaluation suite and return the metrics dict."""
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n╔══ Evaluation Suite ══╗")

    preds, labels, confs = predict(model, val_loader, device)

    save_confusion_matrix(labels, preds, cfg.display_names, output_dir)
    report = save_classification_report(labels, preds, cfg.display_names, output_dir)
    save_baseline_comparison(labels, preds, output_dir)
    plot_learning_curves(history, output_dir)

    return report
