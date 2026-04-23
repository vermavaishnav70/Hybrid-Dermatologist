"""Central configuration for the Phase 2 deep learning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Phase2Config:
    """All hyperparameters and paths for the Phase 2 EfficientNet-B3 + CBAM pipeline.

    EfficientNet-B3 is chosen over B0 because its 12M parameters (vs B0's 5M)
    provide better capacity for fine-grained 6-class medical image classification,
    as documented by Tan & Le (2019) in the compound scaling paper.
    """

    # ── Architecture ──────────────────────────────────────────────────────
    backbone: str = "efficientnet_b3"
    num_classes: int = 6
    hidden_dim: int = 512
    dropout: float = 0.4
    cbam_reduction: int = 16
    unfreeze_blocks: int = 3

    # ── Image preprocessing ───────────────────────────────────────────────
    image_size: int = 300  # EfficientNet-B3 native resolution
    imagenet_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    # ── Augmentation ──────────────────────────────────────────────────────
    mixup_alpha: float = 0.2      # Zhang et al. (2018) MixUp interpolation
    randaug_magnitude: int = 9    # RandAugment strength
    randaug_num_ops: int = 2      # RandAugment operations per image

    # ── Training — Stage 1: head only ─────────────────────────────────────
    epochs_stage1: int = 15
    lr_stage1: float = 1e-3
    label_smoothing: float = 0.1

    # ── Training — Stage 2: fine-tune last N blocks ───────────────────────
    epochs_stage2: int = 30
    lr_stage2_backbone: float = 2e-5
    lr_stage2_head: float = 1e-4
    warmup_epochs: int = 3
    early_stopping_patience: int = 7

    # ── DataLoader ────────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: Path = Path("data/raw/Multi-Class Skin Condition Image Dataset (MSC-6)")
    output_dir: Path = Path("outputs/phase2_deep_learning")
    phase1_output_dir: Path = Path("outputs/phase1_baseline")

    # ── Reproducibility ───────────────────────────────────────────────────
    random_state: int = 42

    # ── Class names (canonical order matching Phase 1) ────────────────────
    class_names: tuple[str, ...] = (
        "acne",
        "dark_spots",
        "eczema",
        "normal",
        "rosacea",
        "wrinkles",
    )

    display_names: tuple[str, ...] = (
        "acne",
        "dark spots",
        "eczema",
        "normal",
        "rosacea",
        "wrinkles",
    )

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.phase1_output_dir = Path(self.phase1_output_dir)
