# Hybrid Skin Analysis & Care Recommendation System

> **Final-Year Project — Hybrid Dermatologist**
> A three-phase ablation study comparing Classical ML, Deep Learning, and Hybrid Fusion
> for automated skin condition classification and personalised skincare recommendation.

---

## Results Summary

| Model | Phase | Val Accuracy | F1 Weighted | F1 Macro |
|-------|-------|:---:|:---:|:---:|
| SVM (RBF Kernel) | Phase 1 | 0.785 | 0.788 | 0.771 |
| Random Forest | Phase 1 | 0.809 | 0.806 | 0.784 |
| EfficientNet-B3 + CBAM | Phase 2 | 0.853 | 0.852 | 0.842 |
| **Hybrid Fusion (Ours)** | **Phase 3** | **0.854** | **0.854** | **0.842** |

### Ablation Study (Phase 3)

| Variant | F1 Weighted | Δ vs Full Model |
|---------|:-----------:|:---------------:|
| Hybrid Fusion (full model) | **0.8538** | baseline |
| Concat instead of Attention | 0.8550 | ≈ tie (+0.1%) |
| CNN only (no ML features) | 0.8419 | −1.4% |
| ML only (no CNN) | 0.1365 | **−84.0% collapse** |

> The 84% collapse proves the CNN stream is the primary performance driver.
> The 1.4% drop when ML features are removed proves they add measurable value.

---

## Architecture

```
Input Image
     │
     ├─── CNN Stream ──────────────────────────────────────────────────────────┐
     │    EfficientNet-B3 (pretrained ImageNet)                                │
     │    + CBAM Attention (channel + spatial)                                 │
     │    → cnn_embed: (B, 1536)                                               │
     │    → cnn_proj:  (B, 512)  via Linear + BN + GELU                       │
     │                                                                         │
     └─── ML Stream ───────────────────────────────────────────────────────────┤
          Phase 1 Handcrafted Features (154 dims):                             │
          • HSV Color Histogram (96 dims) — redness / pigmentation             │
          • LBP Texture (26 dims)         — dryness / bumps                    │
          • GLCM Texture (32 dims)        — scarring / macro-texture           │
          → ml_proj: (B, 512) via Linear + BN + GELU                          │
                                                                               │
                    Attention Gate: α = sigmoid(W · [cnn_embed; ml_features])  │
                    fused = α · cnn_proj + (1−α) · ml_proj                    │
                                                                               │
                    Classifier: 512 → 256 → 6 classes                         │
                         ↓
              [acne | dark_spots | eczema | normal | rosacea | wrinkles]
```

**Key Innovation:** The attention gate learns *per sample* how much to trust each stream. Texture-heavy conditions (eczema) → α shifts toward ML. Complex spatial conditions (acne, rosacea) → α shifts toward CNN.

---

## Dataset

**Multi-Class Skin Condition Image Dataset (MSC-6)**
- 6 classes: `acne`, `dark_spots`, `eczema`, `normal`, `rosacea`, `wrinkles`
- ~9,400 images total | 7,879 train / 944 val / 1,179 test
- Imbalance handling: `WeightedRandomSampler` + inverse-frequency loss weights

Expected layout:
```
data/raw/Multi-Class Skin Condition Image Dataset (MSC-6)/
  train/
    acne/
    dark_spots/
    eczema/
    normal/
    rosacea/
    wrinkles/
  val/
  test/
```

---

## Setup & Commands

### 1. Environment
```bash
bash setup.sh               # Creates .venv, installs all dependencies
source .venv/bin/activate
```

### 2. Pre-compute ML Features (do this ONCE before training)
```bash
make precompute             # Parallel LBP/GLCM extraction → 50x training speedup
```

### 3. Training
```bash
make train-a                # Phase 1: SVM + Random Forest
make train-b                # Phase 2: EfficientNet-B3 + CBAM (two-stage)
make train-c                # Phase 3: Hybrid Fusion (auto-resumes from checkpoint)
make train                  # All three phases sequentially
```

### 4. Evaluation & Ablation
```bash
make eval                   # Confusion matrix, classification report, cross-phase comparison
make ablation               # Component-removal ablation study
make gradcam                # Grad-CAM heatmaps for all 6 classes
make calibrate              # Temperature scaling + ECE reliability diagram
```

### 5. Demo
```bash
make demo                   # Launch Streamlit web app on localhost:8501
```

### 6. Docker
```bash
make docker                 # Build and run in container
```

---

## Project Structure

```
Hybrid-Dermatologist/
├── app/
│   └── streamlit_app.py        # Production Streamlit UI (skin crop, recommendations)
├── data/
│   └── raw/                    # MSC-6 dataset (not committed to git)
├── docker/
│   └── Dockerfile
├── notebooks/
│   ├── phase1/                 # 01–04: EDA, features, training, evaluation
│   ├── phase2/                 # phase2.ipynb: EfficientNet-B3 deep learning
│   └── phase3/
│       ├── 05_hybrid_model_training.ipynb      # Model C training (crash-safe)
│       ├── 06_final_evaluation_and_ablation.ipynb  # Ablation study
│       └── 07_interpretability_and_gradcam.ipynb   # Grad-CAM + attention
├── outputs/
│   ├── phase1_baseline/        # SVM/RF models, metrics, confusion matrices
│   ├── phase2_deep_learning/   # EfficientNet weights, Grad-CAM, metrics
│   └── phase3_hybrid/          # Hybrid model, ablation CSV, all visualizations
├── scripts/
│   ├── precompute_features.py  # Parallel ML feature extraction
│   └── gpt4v_baseline.py       # GPT-4V zero-shot baseline
├── src/skin_analysis/
│   ├── phase1/                 # data.py, features.py, models.py, pipeline.py
│   ├── phase2/                 # model.py (EfficientNet+CBAM), train.py, gradcam.py
│   └── phase3/
│       ├── config.py           # All hyperparameters (single source of truth)
│       ├── model_c.py          # HybridFusionModel with attention gate
│       ├── dataset.py          # HybridSkinDataset with feature caching
│       ├── train_c.py          # Two-stage training with crash-safe checkpointing
│       ├── evaluate_all.py     # Cross-phase evaluation
│       ├── ablation.py         # Component-removal ablation runner
│       ├── calibrate.py        # Temperature scaling + ECE
│       └── gradcam_hybrid.py   # Grad-CAM for the hybrid model
├── Makefile                    # One-command pipeline
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Outputs Generated

All artifacts are stored in `outputs/phase3_hybrid/`:

| File | Description |
|------|-------------|
| `best_model_hybrid.pth` | Best model weights (by val F1) |
| `training_history.csv` | Per-epoch loss, accuracy, F1 (15 epochs) |
| `confusion_matrix_hybrid.png` | Per-class classification performance |
| `ablation_component_removal.csv` | Full ablation results table |
| `ablation_bar_chart.png` | Visual ablation comparison |
| `ablation_diagnostics.txt` | Written diagnostic interpretation |
| `cross_phase_comparison.png` | Phase 1 vs 2 vs 3 comparison chart |
| `gradcam_summary_hybrid.png` | 6-class Grad-CAM heatmap grid |
| `gradcam_hybrid_<class>_*.png` | Individual Grad-CAM images per class |
| `attention_weight_distribution.png` | CNN vs ML trust (α) per condition |
| `feature_importance_per_class.png` | ML feature gradient importance |
| `per_class_ml_impact.png` | Per-class F1 drop when ML stream removed |

---

## Streamlit App Features

- **Upload** any face/skin image (JPG, PNG, WEBP)
- **Auto skin-crop** using OpenCV HSV detection before analysis
- **Dual validation**: One-Class SVM + Heuristic colour check to reject non-skin images
- **Hybrid inference**: Real-time classification into 6 skin conditions
- **Grad-CAM overlay**: Visual explanation of where the model is looking
- **Skincare recommendations**: Ingredient suggestions per condition
- **Climate & skin-type tips**: Personalised advice for tropical, dry, and cold climates
- **Clinical disclaimer**: Prominently displayed on every result

---

## Reproducibility

All experiments use `seed_everything(42)` which seeds:
- Python `random`, `numpy`, `torch`, and `torch.cuda`/`torch.mps`

**Checkpointing:** Training saves the full state (model + optimizer + scheduler + history) every epoch. If training is interrupted, re-running `make train-c` automatically resumes from the last saved epoch.

---

## Per-Class Performance (Hybrid Fusion, Final)

| Class | F1 Score | Notes |
|-------|:--------:|-------|
| Acne | 0.797 | Visually similar to early-stage rosacea |
| Dark Spots | 0.764 | Hardest class — subtle vs normal skin |
| Eczema | 0.856 | Texture signals from LBP/GLCM help significantly |
| Normal | 0.913 | Easiest — clear visual distinction |
| Rosacea | 0.818 | Requires both colour and spatial features |
| Wrinkles | 0.907 | CNN detects fine line patterns well |

---

## Citation / References

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
- Ojala, T., et al. (2002). Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. *IEEE TPAMI*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
- Dataset: Multi-Class Skin Condition Image Dataset (MSC-6), Kaggle.

---

*This project was developed as a final-year research project. It is for educational purposes only and does not constitute medical advice.*
