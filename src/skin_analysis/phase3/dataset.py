"""Hybrid Dataset for Phase 3: returns (image_tensor, ml_features, label).

This dataset simultaneously loads images for the CNN stream and extracts
Phase 1 handcrafted features for the ML stream.  Features are cached to
disk after first extraction to avoid recomputation.

Clinical rationale for dual-stream data loading:
  - The CNN stream needs ImageNet-normalised tensors at 300x300
  - The ML stream needs raw uint8 BGR images for feature extraction
  - Both must correspond to the same original image for fusion to work
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from ..phase1.features import extract_features
from .config import Phase3Config


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


class HybridSkinDataset(Dataset):
    """PyTorch dataset returning (image_tensor, ml_features, label).

    For each image, this dataset:
    1. Loads the image as a PIL Image -> applies torchvision transforms -> tensor
    2. Loads the same image as OpenCV BGR uint8 -> extracts Phase 1 features
    3. Caches the extracted features to avoid recomputation

    The feature cache is stored as a single .npy file per image, keyed by
    a hash of the image path for fast lookup.
    """

    def __init__(
        self,
        data_dir: str | Path,
        class_names: Sequence[str],
        transform: transforms.Compose | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        for folder in sorted(self.data_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = _normalize_label(folder.name)
            if label not in self.class_to_idx:
                continue
            label_idx = self.class_to_idx[label]
            for img_path in sorted(folder.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    # Pre-calculate cache path to avoid MD5 overhead during training
                    cache_path = self._get_cache_path(img_path)
                    self.samples.append({
                        "path": img_path,
                        "label": label_idx,
                        "cache_path": cache_path
                    })

        if not self.samples:
            raise ValueError(f"No images found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cache_path(self, img_path: Path) -> Path | None:
        """Generate a cache file path for the given image."""
        if self.cache_dir is None:
            return None
        path_hash = hashlib.md5(str(img_path).encode()).hexdigest()
        return self.cache_dir / f"{path_hash}.npy"

    def _extract_ml_features(self, img_path: Path) -> np.ndarray:
        """Extract Phase 1 handcrafted features, using cache if available.

        Feature vector (154 dims):
          - HSV color histogram: 32 bins x 3 channels = 96 dims
            Clinical: captures redness (rosacea), pigmentation shifts (dark spots)
          - LBP histogram: P=24, R=3, uniform = 26 dims
            Clinical: captures micro-texture (eczema dryness, acne bumps)
          - GLCM features: 4 properties x 2 distances x 4 angles = 32 dims
            Clinical: captures macro-texture spatial correlation (scarring)
        """
        cache_path = self._get_cache_path(img_path)

        # Try loading from cache
        if cache_path and cache_path.exists():
            return np.load(cache_path)

        # Extract features from raw image
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            # Return zero features for unreadable images
            return np.zeros(154, dtype=np.float32)

        # Resize to 224x224 for consistent feature extraction (Phase 1 standard)
        image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        features = extract_features(image_bgr)

        # Cache features
        if cache_path:
            np.save(cache_path, features)

        return features

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[index]
        path, label_idx, cache_path = sample["path"], sample["label"], sample["cache_path"]

        # ── CNN stream: PIL Image -> transform -> tensor ──────────────────
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # ── ML stream: load features (cached) ─────────────────────────────
        # If cache exists, loading is instantaneous. 
        # If not, extraction will be slow (20s+).
        if cache_path and cache_path.exists():
            ml_features = np.load(cache_path)
        else:
            # Fallback for missing cache (triggers slowness warning)
            # print(f"⚠️  Cache miss for {path.name}. Extracting on-the-fly (SLOW)...")
            ml_features = self._extract_ml_features(path)
            
        ml_features = torch.from_numpy(ml_features).float()

        return image, ml_features, label_idx

    def get_labels(self) -> list[int]:
        """Return all labels for use with WeightedRandomSampler."""
        return [s["label"] for s in self.samples]


# ═══════════════════════════════════════════════════════════════════════════
#  Transform pipelines (same as Phase 2 for consistency)
# ═══════════════════════════════════════════════════════════════════════════


def build_train_transforms(cfg: Phase3Config) -> transforms.Compose:
    """Training-time augmentation pipeline.

    Note: augmentations are applied to the CNN stream only.
    ML features are extracted from the original image to maintain
    clinical interpretability of the handcrafted features.
    """
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 20),
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        # NO RandomVerticalFlip — unnatural face orientation
        transforms.RandAugment(
            num_ops=cfg.randaug_num_ops,
            magnitude=cfg.randaug_magnitude,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def build_val_transforms(cfg: Phase3Config) -> transforms.Compose:
    """Validation / test-time deterministic transform pipeline."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 20),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


# ═══════════════════════════════════════════════════════════════════════════
#  Weighted sampler
# ═══════════════════════════════════════════════════════════════════════════


def build_weighted_sampler(dataset: HybridSkinDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class imbalance.

    Each sample receives weight = 1 / count_of_its_class, so minority
    classes (dark_spots) are sampled more frequently than majority
    classes (eczema).

    Why BOTH sampler AND weighted loss: the sampler ensures balanced
    mini-batches during training, while the weighted loss further
    penalises misclassification of minority classes.  Together they
    provide stronger imbalance handling than either alone.
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


def compute_class_weights(dataset: HybridSkinDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Returns a tensor of shape (num_classes,) where weight[c] = N / (K * n_c)
    with N = total samples, K = num classes, n_c = samples in class c.
    """
    labels = dataset.get_labels()
    num_classes = len(dataset.class_to_idx)
    class_counts = [0] * num_classes
    for label in labels:
        class_counts[label] += 1

    total = sum(class_counts)
    weights = [total / (num_classes * max(c, 1)) for c in class_counts]
    return torch.tensor(weights, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════


def build_hybrid_dataloaders(
    cfg: Phase3Config,
) -> tuple[DataLoader, DataLoader, HybridSkinDataset, HybridSkinDataset]:
    """Build training and validation DataLoaders for the hybrid model.

    Returns (train_loader, val_loader, train_dataset, val_dataset).
    The DataLoaders use WeightedRandomSampler for balanced training.
    """
    train_dir = cfg.data_dir / "train"
    val_dir = cfg.data_dir / "val"

    train_dataset = HybridSkinDataset(
        data_dir=train_dir,
        class_names=cfg.class_names,
        transform=build_train_transforms(cfg),
        cache_dir=cfg.feature_cache_dir / "train",
    )
    val_dataset = HybridSkinDataset(
        data_dir=val_dir,
        class_names=cfg.class_names,
        transform=build_val_transforms(cfg),
        cache_dir=cfg.feature_cache_dir / "val",
    )

    sampler = build_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
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
