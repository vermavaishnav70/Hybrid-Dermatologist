"""Phase 3: Hybrid Attention-Weighted Fusion Model (Model C).

This phase combines the classical ML handcrafted features from Phase 1
(color histogram, LBP, GLCM) with the deep CNN features from Phase 2
(EfficientNet-B3 + CBAM) using a learned attention-weighted fusion mechanism.

The key innovation is that the fusion weight alpha is learned per-sample:
    alpha = sigmoid(W . [CNN_embed, ML_features])
    fused = alpha * proj(CNN) + (1 - alpha) * proj(ML)

This allows the model to dynamically decide which stream to trust:
- For texture-heavy conditions (eczema): alpha -> ML stream (GLCM/LBP dominant)
- For complex multi-condition images: alpha -> CNN stream (global context)

Result: the whole is greater than the sum of its parts (Synergistic Innovation).
"""
