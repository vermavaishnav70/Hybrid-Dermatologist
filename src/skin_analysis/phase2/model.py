"""EfficientNet-B3 + CBAM architecture for skin-condition classification.

Architecture rationale
----------------------
* **EfficientNet-B3** (Tan & Le, 2019): Compound-scaled CNN with 12M parameters.
  B3 is chosen over B0 (5M) because the additional capacity improves fine-grained
  discrimination across 6 visually similar skin conditions.  Depthwise separable
  convolutions keep the FLOPs manageable despite the larger model.

* **CBAM** (Woo et al., 2018): Convolutional Block Attention Module inserted
  between the backbone feature maps and the global average pool.  The channel
  attention gate highlights *which* feature maps matter (e.g., redness channels
  for rosacea) while the spatial attention gate highlights *where* to look
  (e.g., localised acne lesions).  This directly addresses Phase 1's failure
  mode where global pooling of handcrafted features lost spatial locality.

* **Two-stage fine-tuning**: Stage 1 freezes the backbone and trains only the
  CBAM + classifier head so that the randomly initialised layers converge
  before Stage 2 unfreezes the last N backbone blocks for end-to-end tuning.
  BatchNorm layers in frozen blocks stay in eval mode to preserve ImageNet
  running statistics (He et al., 2016).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
#  CBAM — Convolutional Block Attention Module (Woo et al., 2018)
# ═══════════════════════════════════════════════════════════════════════════


class ChannelAttention(nn.Module):
    """Channel attention: *what* feature maps to emphasise.

    Applies global average pooling and global max pooling in parallel, feeds
    both through a shared two-layer MLP, and combines them with a sigmoid gate.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=(2, 3))                       # (B, C)
        max_pool = x.amax(dim=(2, 3))                        # (B, C)
        gate = torch.sigmoid(
            self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        )                                                     # (B, C)
        return x * gate.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention: *where* to focus.

    Concatenates channel-wise average and max maps, then applies a 7×7
    convolution followed by a sigmoid gate.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)                # (B, 1, H, W)
        max_map = x.amax(dim=1, keepdim=True)                # (B, 1, H, W)
        gate = torch.sigmoid(
            self.conv(torch.cat([avg_map, max_map], dim=1))
        )                                                     # (B, 1, H, W)
        return x * gate


class CBAM(nn.Module):
    """Full CBAM block: channel attention → spatial attention (sequential)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  EfficientNet-B3 + CBAM classifier
# ═══════════════════════════════════════════════════════════════════════════


class EfficientNetB3CBAM(nn.Module):
    """EfficientNet-B3 backbone → CBAM → classifier head.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 6).
    hidden_dim : int
        Width of the hidden dense layer before the softmax head.
    dropout : float
        Dropout probability applied before the hidden layer.
    cbam_reduction : int
        Channel reduction ratio inside the CBAM module.
    pretrained : bool
        Whether to load ImageNet-pretrained backbone weights.
    """

    def __init__(
        self,
        num_classes: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.4,
        cbam_reduction: int = 16,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,           # strip the original classifier head
            global_pool="",          # we add our own pooling after CBAM
        )
        backbone_channels = self.backbone.num_features  # 1536 for B3

        # ── CBAM ─────────────────────────────────────────────────────────
        self.cbam = CBAM(channels=backbone_channels, reduction=cbam_reduction)

        # ── Pooling + Classifier head ─────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(backbone_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 1536, H, W)
        features = self.cbam(features)       # (B, 1536, H, W) — attention-gated
        pooled = self.pool(features)         # (B, 1536, 1, 1)
        return self.classifier(pooled)       # (B, num_classes)

    # ── Fine-tuning helpers ───────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters and set BatchNorm to eval mode.

        During Stage 1 training we only optimise the CBAM + classifier head.
        Freezing BN layers is critical because fine-tuning BN with a small
        batch will corrupt the running mean/variance learned on ImageNet.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_last_n_blocks(self, n: int = 3) -> None:
        """Unfreeze the last *n* blocks of the EfficientNet backbone.

        Called at the start of Stage 2.  Earlier blocks retain ImageNet
        features (edges, textures) while later blocks are adapted to the
        skin-condition domain.
        """
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
            block.train()

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the layer to hook for Grad-CAM visualisation.

        We use the spatial attention convolution inside CBAM so that the
        Grad-CAM heatmap directly reflects the attention-weighted features.
        """
        return self.cbam.spatial_att.conv

    def get_trainable_param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        """Return discriminative learning-rate parameter groups for Stage 2.

        Backbone (unfrozen blocks) get a low LR to preserve pretrained features,
        while the CBAM + classifier head use a higher LR.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.cbam.parameters()) + list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]
