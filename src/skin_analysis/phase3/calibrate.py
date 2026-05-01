"""Temperature scaling calibration and Expected Calibration Error (ECE).

Calibration ensures that when a model says "90% confident", the prediction
is correct ~90% of the time.  Medical AI requires well-calibrated confidence
because clinicians need trustworthy probability estimates for decision-making.

Temperature scaling (Guo et al., 2017) is a single-parameter post-hoc
calibration method:
    q_i = softmax(z_i / T)
where T > 1 softens the distribution (reduces overconfidence) and T < 1
sharpens it.  T is optimised on the validation set by minimising NLL.

ECE (Expected Calibration Error) measures the gap between predicted
confidence and actual accuracy across B bins:
    ECE = sum_{b=1}^{B} (n_b / N) * |acc(b) - conf(b)|
A perfectly calibrated model has ECE = 0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class TemperatureScaler(nn.Module):
    """Learned temperature scaling for post-hoc calibration.

    Why temperature scaling over Platt scaling:
      - Single parameter T -> no overfitting risk on small val sets
      - Preserves the ranking of predictions (only rescales softmax)
      - Proven effective for deep neural networks (Guo et al., 2017)
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialise T=1.0 (no scaling) — will be optimised on val set
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature."""
        return logits / self.temperature


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error with binned statistics.

    Args:
        probs: (N, C) predicted probabilities
        labels: (N,) true labels
        n_bins: number of bins for the calibration histogram

    Returns:
        ece: scalar ECE value
        bin_accs: (n_bins,) accuracy per bin
        bin_confs: (n_bins,) mean confidence per bin
        bin_counts: (n_bins,) sample count per bin
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (confidences > bin_boundaries[b]) & (confidences <= bin_boundaries[b + 1])
        if mask.sum() > 0:
            bin_accs[b] = accuracies[mask].mean()
            bin_confs[b] = confidences[mask].mean()
            bin_counts[b] = mask.sum()

    ece = np.sum(bin_counts / max(len(labels), 1) * np.abs(bin_accs - bin_confs))
    return ece, bin_accs, bin_confs, bin_counts


def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    is_hybrid: bool = False,
    max_iter: int = 100,
    lr: float = 0.01,
) -> TemperatureScaler:
    """Fit temperature scaling on the validation set.

    The temperature is optimised by minimising negative log-likelihood
    (NLL) on the validation set.  This is equivalent to maximising
    the log-probability of the correct class.

    Args:
        model: trained model (in eval mode)
        val_loader: validation DataLoader
        device: computation device
        is_hybrid: if True, expects (images, ml_features, labels) batches
        max_iter: LBFGS optimisation iterations
        lr: learning rate for LBFGS
    """
    model.eval()
    scaler = TemperatureScaler().to(device)

    # Collect all logits and labels from val set
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  collecting logits", leave=False):
            if is_hybrid:
                images, ml_features, labels = batch
                images = images.to(device)
                ml_features = ml_features.to(device)
                logits = model(images, ml_features)
            else:
                images = batch[0].to(device)
                labels = batch[1]
                logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(
                labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
            )

    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    # Optimise temperature with LBFGS
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(all_logits)
        loss = criterion(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"  Optimal temperature: T = {scaler.temperature.item():.4f}")
    return scaler


def calibrate_and_report(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    model_name: str = "model",
    is_hybrid: bool = False,
    n_bins: int = 15,
) -> dict:
    """Full calibration pipeline: fit T on val, compute ECE, plot reliability.

    Returns dict with keys: ece_before, ece_after, temperature
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # ── Collect predictions ───────────────────────────────────────────────
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"  {model_name} inference", leave=False):
            if is_hybrid:
                images, ml_features, labels = batch
                images = images.to(device)
                ml_features = ml_features.to(device)
                logits = model(images, ml_features)
            else:
                images = batch[0].to(device)
                labels = batch[1]
                logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(
                labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
            )

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # ── ECE before calibration ────────────────────────────────────────────
    probs_before = F.softmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    ece_before, accs_before, confs_before, counts_before = compute_ece(
        probs_before, labels_np, n_bins=n_bins
    )
    print(f"  {model_name} ECE (before calibration): {ece_before:.4f}")

    # ── Fit temperature scaling ───────────────────────────────────────────
    scaler = TemperatureScaler()
    scaler_device = torch.device("cpu")  # fit on CPU for stability
    all_logits_dev = all_logits.to(scaler_device)
    all_labels_dev = all_labels.to(scaler_device)
    scaler = scaler.to(scaler_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(all_logits_dev), all_labels_dev)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = scaler.temperature.item()

    # ── ECE after calibration ─────────────────────────────────────────────
    with torch.no_grad():
        probs_after = F.softmax(scaler(all_logits_dev), dim=1).numpy()
    ece_after, accs_after, confs_after, counts_after = compute_ece(
        probs_after, labels_np, n_bins=n_bins
    )
    print(f"  {model_name} ECE (after calibration):  {ece_after:.4f}")
    print(f"  {model_name} Temperature:              {temperature:.4f}")

    # ── Reliability diagram ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, title, accs, confs, counts, ece_val in [
        (axes[0], "Before Calibration", accs_before, confs_before, counts_before, ece_before),
        (axes[1], "After Calibration", accs_after, confs_after, counts_after, ece_after),
    ]:
        bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
        mask = counts > 0

        ax.bar(bin_centers[mask], accs[mask], width=1 / n_bins, alpha=0.6,
               edgecolor="black", color="#3498db", label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.scatter(confs[mask], accs[mask], color="#e74c3c", zorder=5, s=30)

        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{title}\nECE = {ece_val:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Reliability Diagram — {model_name}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"reliability_diagram_{model_name}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Reliability diagram saved to {output_dir}")

    return {
        "ece_before": ece_before,
        "ece_after": ece_after,
        "temperature": temperature,
    }
