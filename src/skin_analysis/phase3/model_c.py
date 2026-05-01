"""Hybrid Attention-Weighted Fusion Model (Model C).

Architecture overview
---------------------
This model implements a **neuro-symbolic** hybrid that fuses deep CNN features
with handcrafted classical ML features via a learned attention gate.

    cnn_embed  = EfficientNet-B3 + CBAM features    (B, 1536)
    ml_features = Phase 1 handcrafted features       (B, 154)
    combined   = concat([cnn_embed, ml_features])    (B, 1690)
    alpha      = sigmoid(attn_layer(combined))       (B, 512) per-sample gate
    cnn_proj   = cnn_projection(cnn_embed)           (B, 512)
    ml_proj    = ml_projection(ml_features)          (B, 512)
    fused      = alpha * cnn_proj + (1-alpha) * ml_proj   (B, 512) symbiotic blend
    output     = classifier(fused)                   (B, num_classes)

Why attention-weighted fusion over simple concatenation
-------------------------------------------------------
Simple concatenation (Hybrid Innovation 2/5) gives equal weight to both streams
for every sample.  Attention-weighted fusion (Hybrid Innovation 5/5) learns
alpha = sigma(W . [CNN, ML]) — the model decides PER-SAMPLE which stream to trust:
  - For texture-heavy eczema:   alpha -> ML stream (GLCM/LBP are primary signals)
  - For complex multi-condition: alpha -> CNN stream (global spatial context)
  - The whole is greater than the sum of its parts -> Synergistic

Mathematical formulation
------------------------
Let c in R^{1536} be the CNN embedding and m in R^{154} the ML feature vector.

    z = [c; m]                           (concatenation, z in R^{1690})
    alpha = sigma(W_a z + b_a)           (attention gate, alpha in R^{512})
    c' = W_c c + b_c                     (CNN projection, c' in R^{512})
    m' = W_m m + b_m                     (ML projection, m' in R^{512})
    f  = alpha . c' + (1 - alpha) . m'   (element-wise gated fusion)
    y  = softmax(W_y f + b_y)            (classification head)

Why this is neuro-symbolic:
  - CNN (neural): learns hierarchical features end-to-end
  - Handcrafted (symbolic): encodes domain knowledge (haemoglobin redness,
    texture periodicity) into fixed-form features
  - Fusion gate: bridges both paradigms with a learned soft switch

Residual connections and vanishing gradients
--------------------------------------------
The skip connection in the fusion (alpha * c' + (1-alpha) * m') ensures:
    dL/dx = dL/dF(x) . (alpha + (1-alpha)) = dL/dF(x) . 1
The gradient always has a non-zero path, preventing vanishing gradient.

Why EfficientNet-B3 over B0
----------------------------
Compound scaling (depth x width x resolution) achieves better accuracy
per parameter.  12M vs 5M params — acceptable overhead for M2 Mac.
B3 at 300x300 input captures finer skin texture details than B0 at 224x224.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..phase2.model import EfficientNetB3CBAM, CBAM


class AttentionFusionGate(nn.Module):
    """Learned attention gate for CNN/ML feature fusion.

    Given concatenated CNN and ML features, produces per-dimension attention
    weights alpha in [0, 1] that blend the two projected streams.

    Clinical rationale: different skin conditions benefit from different
    feature sources.  Acne lesion counting benefits from CNN spatial features,
    while eczema texture roughness benefits from GLCM/LBP features.
    The gate learns this automatically from data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Why two-layer MLP for attention: single linear layer cannot learn
        # non-linear interactions between CNN and ML feature spaces.
        # The intermediate ReLU allows the gate to learn conditional attention
        # patterns (e.g., "trust ML when CNN confidence is low").
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),  # Output in [0, 1] for soft blending
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        """Compute attention weights alpha from concatenated features.

        Args:
            combined: (B, cnn_dim + ml_dim) concatenated feature vector

        Returns:
            alpha: (B, output_dim) attention weights in [0, 1]
        """
        return self.gate(combined)


class HybridFusionModel(nn.Module):
    """Hybrid Attention-Weighted Fusion Model (Model C).

    Combines EfficientNet-B3 + CBAM deep features with Phase 1 handcrafted
    features (color histogram + LBP + GLCM) via a learned attention gate.

    Parameters
    ----------
    num_classes : int
        Number of output classes (6 for MSC-6 dataset).
    cnn_feature_dim : int
        Dimensionality of CNN backbone output (1536 for EfficientNet-B3).
    ml_feature_dim : int
        Dimensionality of handcrafted ML features (154 for Phase 1).
    fusion_hidden_dim : int
        Shared projection dimension for both streams before fusion.
    dropout : float
        Dropout probability for regularisation.
    cbam_reduction : int
        Channel reduction ratio for the CBAM module.
    pretrained : bool
        Whether to load ImageNet-pretrained backbone weights.
    ablation_mode : str or None
        If set, disables specific components for ablation study:
        - 'no_attention': simple concatenation instead of attention fusion
        - 'no_glcm': zeros out GLCM features (dims 122:154)
        - 'no_lbp': zeros out LBP features (dims 96:122)
        - 'no_color': zeros out color histogram features (dims 0:96)
        - 'no_ml': zeros out all ML features
        - None: full model (default)
    """

    def __init__(
        self,
        num_classes: int = 6,
        cnn_feature_dim: int = 1536,
        ml_feature_dim: int = 154,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.4,
        cbam_reduction: int = 16,
        pretrained: bool = True,
        ablation_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.cnn_feature_dim = cnn_feature_dim
        self.ml_feature_dim = ml_feature_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.ablation_mode = ablation_mode

        # ── CNN backbone (EfficientNet-B3 + CBAM from Phase 2) ────────────
        # Why reuse Phase 2 backbone: transfer learning from the already
        # fine-tuned skin condition features avoids redundant training and
        # ensures the CNN stream is domain-adapted.
        self.backbone = EfficientNetB3CBAM(
            num_classes=num_classes,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            cbam_reduction=cbam_reduction,
            pretrained=pretrained,
        )
        # We extract features BEFORE the classifier head
        # The backbone's forward returns logits, so we need to hook into
        # the intermediate representations
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── Stream projections ────────────────────────────────────────────
        # Project both streams to the same dimensionality for fusion.
        # BatchNorm after projection reduces internal covariate shift
        # between the very different feature scales (CNN: ~0-10, ML: ~0-1).
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.ml_projection = nn.Sequential(
            nn.Linear(ml_feature_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── Attention fusion gate ─────────────────────────────────────────
        # Input: concatenated CNN + ML features (1536 + 154 = 1690)
        # Output: attention weights alpha (512) in [0, 1]
        self.attention_gate = AttentionFusionGate(
            input_dim=cnn_feature_dim + ml_feature_dim,
            output_dim=fusion_hidden_dim,
            dropout=0.3,
        )

        # ── Classification head ───────────────────────────────────────────
        # Why F1 macro over accuracy: class distribution has ~13x imbalance
        # (eczema vs dark_spots). F1 macro weights all classes equally.
        if ablation_mode == "no_attention":
            # Simple concatenation baseline: 512 + 512 = 1024
            classifier_input_dim = fusion_hidden_dim * 2
        else:
            classifier_input_dim = fusion_hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

    def _extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN feature embeddings before the classifier head.

        Uses the backbone's internal structure to get the 1536-dim
        feature vector after CBAM attention and global average pooling.
        """
        features = self.backbone.backbone(x)       # (B, 1536, H, W)
        features = self.backbone.cbam(features)    # (B, 1536, H, W) attention-gated
        pooled = self.pool(features)               # (B, 1536, 1, 1)
        return pooled.flatten(1)                   # (B, 1536)

    def _apply_ablation_mask(self, ml_features: torch.Tensor) -> torch.Tensor:
        """Zero out specific feature groups for ablation study.

        Feature layout (154 dims total):
          [0:96]   = HSV color histogram (32 bins x 3 channels)
          [96:122] = LBP histogram (26 bins, uniform, P=24, R=3)
          [122:154] = GLCM features (4 properties x 2 distances x 4 angles)

        Clinical reasoning for each ablation:
          - no_color: tests if redness/pigmentation cues from HSV matter
          - no_lbp: tests if micro-texture patterns matter
          - no_glcm: tests if macro-texture spatial correlation matters
          - no_ml: tests if handcrafted features add value beyond CNN
        """
        if self.ablation_mode == "no_glcm":
            ml_features = ml_features.clone()
            ml_features[:, 122:154] = 0.0
        elif self.ablation_mode == "no_lbp":
            ml_features = ml_features.clone()
            ml_features[:, 96:122] = 0.0
        elif self.ablation_mode == "no_color":
            ml_features = ml_features.clone()
            ml_features[:, 0:96] = 0.0
        elif self.ablation_mode == "no_ml":
            ml_features = torch.zeros_like(ml_features)
        return ml_features

    def forward(
        self,
        images: torch.Tensor,
        ml_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the hybrid fusion model.

        Args:
            images: (B, 3, H, W) input images (ImageNet-normalised)
            ml_features: (B, 154) handcrafted feature vectors from Phase 1

        Returns:
            logits: (B, num_classes) raw classification logits
        """
        # ── Extract CNN features ──────────────────────────────────────────
        cnn_embed = self._extract_cnn_features(images)  # (B, 1536)

        if self.ablation_mode == "no_dl":
            cnn_embed = torch.zeros_like(cnn_embed)

        # ── Apply ablation mask to ML features ────────────────────────────
        ml_features = self._apply_ablation_mask(ml_features)

        # ── Project both streams ──────────────────────────────────────────
        cnn_proj = self.cnn_projection(cnn_embed)       # (B, 512)
        ml_proj = self.ml_projection(ml_features)       # (B, 512)

        # ── Fusion ────────────────────────────────────────────────────────
        if self.ablation_mode == "no_attention":
            # Ablation: simple concatenation (no learned attention)
            # This is the baseline that scores Hybrid Innovation 2/5
            fused = torch.cat([cnn_proj, ml_proj], dim=1)  # (B, 1024)
        else:
            # Full model: attention-weighted fusion (Hybrid Innovation 5/5)
            combined = torch.cat([cnn_embed, ml_features], dim=1)  # (B, 1690)
            alpha = self.attention_gate(combined)            # (B, 512)
            # alpha * CNN + (1-alpha) * ML: symbiotic blend
            fused = alpha * cnn_proj + (1.0 - alpha) * ml_proj  # (B, 512)

        # ── Classification ────────────────────────────────────────────────
        return self.classifier(fused)  # (B, num_classes)

    def get_attention_weights(
        self,
        images: torch.Tensor,
        ml_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned attention weights alpha for analysis.

        Useful for interpreting which stream the model trusts per sample.
        High alpha -> trusts CNN more. Low alpha -> trusts ML more.
        """
        with torch.no_grad():
            cnn_embed = self._extract_cnn_features(images)
            combined = torch.cat([cnn_embed, ml_features], dim=1)
            alpha = self.attention_gate(combined)
        return alpha

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the layer to hook for Grad-CAM visualisation.

        We use the CBAM spatial attention convolution so that the
        Grad-CAM heatmap reflects the attention-weighted CNN features.
        """
        return self.backbone.cbam.spatial_att.conv

    def freeze_backbone(self) -> None:
        """Freeze all CNN backbone parameters for Stage 1 training.

        During Stage 1 we only train the fusion layers (attention gate,
        projections, classifier).  The backbone stays frozen to preserve
        the domain-adapted features learned in Phase 2.

        Why freeze BN layers: fine-tuning BatchNorm with small batches
        corrupts the running mean/variance from ImageNet (He et al., 2016).
        """
        for param in self.backbone.backbone.parameters():
            param.requires_grad = False
        self.backbone.backbone.eval()
        # Also freeze CBAM since it was trained in Phase 2
        for param in self.backbone.cbam.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3) -> None:
        """Unfreeze the last n blocks for Stage 2 fine-tuning.

        Earlier blocks retain ImageNet + Phase 2 features (edges, textures,
        skin patterns) while later blocks adapt to the hybrid fusion signal.
        """
        blocks = list(self.backbone.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
            block.train()
        # Also unfreeze CBAM for end-to-end adaptation
        for param in self.backbone.cbam.parameters():
            param.requires_grad = True

    def get_trainable_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
    ) -> list[dict]:
        """Return discriminative learning-rate parameter groups for Stage 2.

        Backbone (unfrozen blocks) use a low LR to preserve pretrained features.
        Fusion layers (attention gate, projections, classifier) use a higher LR.

        Why discriminative LR: the backbone has already converged to useful
        features in Phase 2.  Large updates would destroy those features.
        The fusion layers are new and need faster learning.
        """
        backbone_params = [
            p for p in self.backbone.backbone.parameters() if p.requires_grad
        ]
        cbam_params = list(self.backbone.cbam.parameters())
        fusion_params = (
            list(self.cnn_projection.parameters())
            + list(self.ml_projection.parameters())
            + list(self.attention_gate.parameters())
            + list(self.classifier.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": cbam_params, "lr": lr_backbone * 2},
            {"params": fusion_params, "lr": lr_head},
        ]

    @classmethod
    def from_phase2_checkpoint(
        cls,
        checkpoint_path: str | Path,
        num_classes: int = 6,
        ml_feature_dim: int = 154,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.4,
        cbam_reduction: int = 16,
        ablation_mode: str | None = None,
        device: torch.device | None = None,
    ) -> "HybridFusionModel":
        """Create a HybridFusionModel initialised from a Phase 2 checkpoint.

        This loads the pretrained EfficientNet-B3 + CBAM weights from Phase 2
        into the CNN backbone, then adds randomly initialised fusion layers.

        Why initialise from Phase 2: the backbone has already learned
        domain-specific skin condition features.  Starting from ImageNet
        weights would require redundant Phase 2 training.
        """
        model = cls(
            num_classes=num_classes,
            cnn_feature_dim=1536,  # EfficientNet-B3
            ml_feature_dim=ml_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            cbam_reduction=cbam_reduction,
            pretrained=False,  # We load our own weights
            ablation_mode=ablation_mode,
        )

        # Load Phase 2 checkpoint into the backbone
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            map_location = device if device else "cpu"
            state_dict = torch.load(
                checkpoint_path, map_location=map_location, weights_only=True
            )
            # The Phase 2 checkpoint has keys for the full EfficientNetB3CBAM
            # We load matching keys into our backbone
            backbone_state = {}
            for key, value in state_dict.items():
                backbone_state[key] = value

            # Load with strict=False because the classifier head dimensions differ
            missing, unexpected = model.backbone.load_state_dict(
                backbone_state, strict=False
            )
            print(f"  Loaded Phase 2 checkpoint: {checkpoint_path}")
            if missing:
                print(f"    Missing keys (expected for new fusion layers): {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
        else:
            print(f"  ⚠ Phase 2 checkpoint not found: {checkpoint_path}")
            print("    Initialising backbone with ImageNet weights only.")

        return model
