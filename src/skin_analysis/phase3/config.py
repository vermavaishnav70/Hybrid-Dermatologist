"""Central configuration for the Phase 3 Hybrid Fusion pipeline.

Phase 3 builds on Phase 2's EfficientNet-B3 + CBAM backbone and Phase 1's
handcrafted feature extractor.  The hybrid model fuses both streams via a
learned attention gate that decides per-sample which stream to trust.

Architecture rationale
----------------------
* CNN feature dim (1536): EfficientNet-B3 backbone output after global pool.
* ML feature dim (154): color_hist(96) + LBP(26) + GLCM(32) from Phase 1.
* Fusion hidden dim (512): projects both streams to a common space before
  applying the learned attention gate alpha = sigmoid(W . [cnn, ml]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Phase3Config:
    """All hyperparameters and paths for the Phase 3 Hybrid Fusion pipeline.

    The hybrid architecture combines EfficientNet-B3 + CBAM features (Phase 2)
    with handcrafted color/texture features (Phase 1) via attention-weighted
    fusion.  This design choice is motivated by the observation that:
      - CNN features excel at capturing global spatial patterns
      - Handcrafted features capture clinically interpretable signals (redness,
        texture roughness) that the CNN may not prioritise
    The attention gate lets the model learn WHEN each stream matters.
    """

    # ── Architecture ──────────────────────────────────────────────────────
    backbone: str = "efficientnet_b3"
    num_classes: int = 6
    cnn_feature_dim: int = 1536   # EfficientNet-B3 backbone output
    ml_feature_dim: int = 154     # Phase 1: color(96) + LBP(26) + GLCM(32)
    fusion_hidden_dim: int = 512  # Shared projection dimension for fusion
    hidden_dim: int = 512         # Classifier hidden layer
    dropout: float = 0.4
    cbam_reduction: int = 16
    unfreeze_blocks: int = 3

    # ── Image preprocessing ───────────────────────────────────────────────
    image_size: int = 300         # EfficientNet-B3 native resolution
    imagenet_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    # ── Augmentation ──────────────────────────────────────────────────────
    mixup_alpha: float = 0.2
    randaug_magnitude: int = 9
    randaug_num_ops: int = 2

    # ── Training — Stage 1: fusion head only (backbone frozen) ────────────
    epochs_stage1: int = 5
    lr_stage1: float = 1e-3
    label_smoothing: float = 0.1

    # ── Training — Stage 2: fine-tune backbone + fusion ───────────────────
    epochs_stage2: int = 10
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
    output_dir: Path = Path("outputs/phase3_hybrid")
    phase2_checkpoint: Path = Path("outputs/phase2_deep_learning/best_model_phase2.pth")
    phase1_output_dir: Path = Path("outputs/phase1_baseline")
    feature_cache_dir: Path = Path("outputs/phase3_hybrid/feature_cache")

    # ── Reproducibility ───────────────────────────────────────────────────
    random_state: int = 42

    # ── Class names (canonical order matching Phase 1 & 2) ────────────────
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
        self.phase2_checkpoint = Path(self.phase2_checkpoint)
        self.phase1_output_dir = Path(self.phase1_output_dir)
        self.feature_cache_dir = Path(self.feature_cache_dir)
