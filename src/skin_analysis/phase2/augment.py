"""Data augmentation, dataset, and dataloader utilities for Phase 2.

Augmentation strategy
---------------------
* **RandAugment** (Cubuk et al., 2020): Applies *num_ops* randomly chosen
  transformations at *magnitude* strength.  This provides diverse colour jitter,
  rotation, and cutout variations without hand-tuning individual transform
  probabilities.

* **MixUp** (Zhang et al., 2018): Convex interpolation of image pairs at the
  batch level.  With α=0.2 the Beta distribution concentrates near λ≈1,
  producing mild but effective label smoothing that reduces overconfident
  predictions on the majority class (eczema).

* **WeightedRandomSampler**: Assigns per-sample weights inversely proportional
  to class frequency so that each mini-batch sees roughly balanced classes
  despite the 13× eczema/normal imbalance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .config import Phase2Config


# ═══════════════════════════════════════════════════════════════════════════
#  Transform pipelines
# ═══════════════════════════════════════════════════════════════════════════


def build_train_transforms(cfg: Phase2Config) -> transforms.Compose:
    """Training-time augmentation pipeline with RandAugment."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 20),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(
            num_ops=cfg.randaug_num_ops,
            magnitude=cfg.randaug_magnitude,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def build_val_transforms(cfg: Phase2Config) -> transforms.Compose:
    """Validation / test-time deterministic transform pipeline."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 20),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_label(name: str) -> str:
    """Canonicalize a folder name to a label (mirrors Phase 1 logic)."""
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    normalized = re.sub(r"^class\d+_", "", normalized)
    alias_map = {
        "dark_spot": "dark_spots",
        "darkspots": "dark_spots",
    }
    return alias_map.get(normalized, normalized)


class SkinConditionDataset(Dataset):
    """PyTorch dataset reading from class-named subfolders.

    Supports the MSC-6 layout where ``data_dir`` is a split root such as
    ``data/raw/.../train/`` containing ``class0_normal/``, ``class1_acne/``, etc.
    """

    def __init__(
        self,
        data_dir: str | Path,
        class_names: Sequence[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []

        for folder in sorted(self.data_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = _normalize_label(folder.name)
            if label not in self.class_to_idx:
                continue
            label_idx = self.class_to_idx[label]
            for img_path in sorted(folder.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label_idx))

        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label_idx = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx, str(path)

    def get_labels(self) -> list[int]:
        """Return all labels for use with WeightedRandomSampler."""
        return [label for _, label in self.samples]


# ═══════════════════════════════════════════════════════════════════════════
#  Weighted sampler
# ═══════════════════════════════════════════════════════════════════════════


def build_weighted_sampler(dataset: SkinConditionDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to address class imbalance.

    Each sample receives weight = 1 / (count of its class), so under-represented
    classes are sampled more frequently.
    """
    labels = dataset.get_labels()
    class_counts: dict[int, int] = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  MixUp collate
# ═══════════════════════════════════════════════════════════════════════════


def mixup_collate_fn(
    batch: list[tuple[torch.Tensor, int, str]],
    alpha: float = 0.2,
    num_classes: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate function that applies MixUp at the batch level.

    Returns soft one-hot labels so that the loss function can use
    ``F.cross_entropy(logits, soft_targets)`` via a manual KL/CE computation.
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    paths = [item[2] for item in batch]

    # One-hot encode labels
    targets = torch.zeros(len(labels), num_classes)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)

    # Sample lambda from Beta(alpha, alpha)
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0

    # Shuffle indices for pairing
    indices = torch.randperm(len(images))
    images = lam * images + (1 - lam) * images[indices]
    targets = lam * targets + (1 - lam) * targets[indices]

    return images, targets, paths


class MixupCollate:
    """Picklable collate function wrapper for multiprocessing DataLoaders."""
    
    def __init__(self, alpha: float, num_classes: int):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        return mixup_collate_fn(batch, alpha=self.alpha, num_classes=self.num_classes)


# ═══════════════════════════════════════════════════════════════════════════
#  DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════


def build_dataloaders(
    cfg: Phase2Config,
) -> tuple[DataLoader, DataLoader, SkinConditionDataset, SkinConditionDataset]:
    """Build training and validation DataLoaders from the MSC-6 dataset.

    Uses MSC-6 ``train/`` for training (with weighted sampling + MixUp) and
    MSC-6 ``val/`` as the held-out validation/test set.
    """
    train_dir = cfg.data_dir / "train"
    val_dir = cfg.data_dir / "val"

    train_dataset = SkinConditionDataset(
        data_dir=train_dir,
        class_names=cfg.class_names,
        transform=build_train_transforms(cfg),
    )
    val_dataset = SkinConditionDataset(
        data_dir=val_dir,
        class_names=cfg.class_names,
        transform=build_val_transforms(cfg),
    )

    sampler = build_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        collate_fn=MixupCollate(cfg.mixup_alpha, cfg.num_classes),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset, val_dataset


# ═══════════════════════════════════════════════════════════════════════════
#  Augmentation visualisation (for the report notebook)
# ═══════════════════════════════════════════════════════════════════════════


def visualize_augmented_samples(
    dataset: SkinConditionDataset,
    class_names: Sequence[str],
    output_path: str | Path,
    samples_per_class: int = 3,
) -> None:
    """Save a grid of augmented training samples for visual inspection.

    Each row shows one class, each column a different augmented version of
    a randomly selected image from that class.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, samples_per_class,
        figsize=(4 * samples_per_class, 4 * n_classes),
    )

    # Group samples by class
    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    mean = torch.tensor(dataset.transform.transforms[-1].mean).view(3, 1, 1)
    std = torch.tensor(dataset.transform.transforms[-1].std).view(3, 1, 1)

    for row, cls_idx in enumerate(range(n_classes)):
        indices = class_indices.get(cls_idx, [])
        rng = np.random.RandomState(42)
        chosen = rng.choice(indices, size=min(samples_per_class, len(indices)), replace=False)
        for col, sample_idx in enumerate(chosen):
            img_tensor, _, _ = dataset[sample_idx]
            # Denormalize for display
            img = img_tensor * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[row, col] if n_classes > 1 else axes[col]
            ax.imshow(img)
            ax.set_title(class_names[cls_idx], fontsize=10)
            ax.axis("off")

    plt.suptitle("Augmented Training Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
