"""Grad-CAM visualisation for the Phase 2 model.

Grad-CAM (Selvaraju et al., 2017) highlights the image regions that most
influence the predicted class by computing gradients of the class score with
respect to the feature maps of a target convolutional layer.

For the EfficientNet-B3 + CBAM architecture the target layer is the spatial
attention convolution inside CBAM, so the heatmaps directly show where the
attention mechanism focuses — proving that acne lesions are localised
(the failure mode identified in Phase 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from .config import Phase2Config
from .model import EfficientNetB3CBAM


def _build_inference_transform(cfg: Phase2Config) -> transforms.Compose:
    """Deterministic transform for Grad-CAM inference."""
    return transforms.Compose([
        transforms.Resize(cfg.image_size + 20),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def _denormalize(tensor: torch.Tensor, mean: tuple, std: tuple) -> np.ndarray:
    """Convert a normalised image tensor back to a [0, 1] numpy array."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = tensor.cpu() * std_t + mean_t
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def generate_gradcam_overlays(
    model: EfficientNetB3CBAM,
    dataset,
    cfg: Phase2Config,
    device: torch.device,
    images_per_class: int = 2,
) -> Path:
    """Generate Grad-CAM overlays for sample images from each class.

    Saves individual ``gradcam_{class}_{idx}.png`` files plus a combined
    summary grid ``gradcam_summary.png``.

    Returns the path to the summary image.
    """
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    target_layer = model.get_gradcam_target_layer()
    cam = GradCAM(model=model, target_layers=[target_layer])

    transform = _build_inference_transform(cfg)

    # Group dataset samples by class
    class_indices: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        # Robustly handle different dataset return formats
        item = dataset[idx]
        label = item[1] if isinstance(item, (tuple, list)) else item
        class_indices.setdefault(label, []).append(idx)

    n_classes = len(cfg.class_names)
    fig, axes = plt.subplots(
        n_classes, images_per_class,
        figsize=(5 * images_per_class, 5 * n_classes),
    )
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    rng = np.random.RandomState(cfg.random_state)

    for cls_idx, cls_name in enumerate(cfg.class_names):
        indices = class_indices.get(cls_idx, [])
        if not indices:
            continue
        chosen = rng.choice(indices, size=min(images_per_class, len(indices)), replace=False)

        for col, sample_idx in enumerate(chosen):
            item = dataset[sample_idx]
            img_tensor = item[0]

            # Transform for model (already transformed by dataset)
            input_tensor = img_tensor.unsqueeze(0).to(device)

            # Generate Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0]  # (H, W)

            # Original image for overlay
            rgb_img = _denormalize(input_tensor.squeeze(0), cfg.imagenet_mean, cfg.imagenet_std)
            overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Save individual
            individual_path = output_dir / f"gradcam_{cls_name}_{col}.png"
            plt.imsave(str(individual_path), overlay)

            # Add to grid
            ax = axes[cls_idx, col]
            ax.imshow(overlay)
            display_name = cfg.display_names[cls_idx] if cls_idx < len(cfg.display_names) else cls_name
            ax.set_title(f"{display_name}", fontsize=12, fontweight="bold")
            ax.axis("off")

    plt.suptitle(
        "Grad-CAM Attention Maps — EfficientNet-B3 + CBAM",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    summary_path = output_dir / "gradcam_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM summary saved to {summary_path}")
    return summary_path
