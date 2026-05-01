"""Unified evaluation pipeline for all three model phases.

Generates all artefacts required for the ablation table:
  * Per-model confusion matrix (CSV + heatmap PNG)
  * Per-model classification report (per-class precision / recall / F1)
  * Combined comparison table across all models
  * Per-class F1 bar chart for visual comparison

This module evaluates ALL models on the SAME held-out test/val split
to ensure fair comparison.  The test set is touched ONCE — final
evaluation only, after all tuning is complete.
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


# ═══════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def predict_hybrid(model, loader, device):
    """Run inference with the hybrid model (expects images + ml_features).

    Returns (predictions, labels, confidences, probabilities).
    """
    model.eval()
    all_preds, all_labels, all_confs, all_probs = [], [], [], []
    for images, ml_features, labels in tqdm(loader, desc="  predict", leave=False):
        images = images.to(device)
        ml_features = ml_features.to(device)
        logits = model(images, ml_features)
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels)
        all_confs.extend(confs.cpu().tolist())
        all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    return all_preds, all_labels, all_confs, all_probs


@torch.no_grad()
def predict_phase2(model, loader, device):
    """Run inference with Phase 2 model (expects images only).

    Returns (predictions, labels, confidences, probabilities).
    """
    model.eval()
    all_preds, all_labels, all_confs, all_probs = [], [], [], []
    for batch in tqdm(loader, desc="  predict", leave=False):
        images = batch[0].to(device)
        labels = batch[1]
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels)
        all_confs.extend(confs.cpu().tolist())
        all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    return all_preds, all_labels, all_confs, all_probs


# ═══════════════════════════════════════════════════════════════════════════
#  Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: Sequence[str],
    output_dir: Path,
    prefix: str = "model",
) -> None:
    """Save confusion matrix as CSV + styled heatmap PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
    ax.set_title(
        f"Confusion Matrix — {prefix.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

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
    prefix: str = "model",
) -> dict:
    """Save sklearn classification report as CSV and return metrics dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
                writer.writerow([
                    cls_name,
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                    f"{r['f1-score']:.4f}",
                    int(r.get("support", 0)),
                ])
        if "accuracy" in report:
            writer.writerow(["accuracy", "", "", f"{report['accuracy']:.4f}", ""])

    print(f"  Classification report saved to {csv_path}")
    return report


# ═══════════════════════════════════════════════════════════════════════════
#  Comparison table
# ═══════════════════════════════════════════════════════════════════════════


def save_ablation_table(
    results: list[dict],
    output_dir: Path,
    class_names: Sequence[str],
) -> None:
    """Save the full ablation comparison table.

    Each entry in results should have:
        model_name, accuracy, f1_macro, f1_weighted, per_class_f1 (dict),
        ece (optional), train_time (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_table.csv"

    headers = [
        "Model", "Accuracy", "F1 Macro", "F1 Weighted",
    ]
    for name in class_names:
        headers.append(f"F1 {name}")
    headers.extend(["ECE", "Train Time (min)"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            row = [
                r["model_name"],
                f"{r['accuracy']:.4f}",
                f"{r['f1_macro']:.4f}",
                f"{r['f1_weighted']:.4f}",
            ]
            per_class = r.get("per_class_f1", {})
            for name in class_names:
                f1 = per_class.get(name, 0.0)
                row.append(f"{f1:.4f}")
            row.append(f"{r.get('ece', 0.0):.4f}" if r.get("ece") is not None else "—")
            row.append(
                f"{r.get('train_time', 0.0):.1f}" if r.get("train_time") is not None else "—"
            )
            writer.writerow(row)

    print(f"  Ablation table saved to {csv_path}")

    # ── Per-class F1 bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.8 / max(len(results), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for i, r in enumerate(results):
        per_class = r.get("per_class_f1", {})
        f1_values = [per_class.get(name, 0.0) for name in class_names]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_values, width, label=r["model_name"], color=colors[i])

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Comparison Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class F1 chart saved to {output_dir / 'per_class_f1_comparison.png'}")
