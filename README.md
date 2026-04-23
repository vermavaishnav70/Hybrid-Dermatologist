# Hybrid-Dermatologist

Phase 1 and 2 for a skin condition classification project.
Phase 1 uses handcrafted computer vision features and classical machine learning.
Phase 2 uses deep learning with an EfficientNet-B3 backbone and CBAM attention.
## Scope

This repository implements two phases:

**Phase 1 (Classical ML Baseline):**
- HSV color histogram features for redness-sensitive signals such as rosacea
- Texture features using LBP and GLCM for texture-heavy classes such as eczema
- Two baseline classifiers: SVM with RBF kernel and Random Forest

**Phase 2 (Deep Learning):**
- **EfficientNet-B3** backbone pre-trained on ImageNet.
- **CBAM** (Convolutional Block Attention Module) to localize lesions.
- Two-stage training: frozen backbone warm-up followed by fine-tuning.
- Advanced Augmentation: **MixUp** and **RandAugment**.
- Imbalance handling via `WeightedRandomSampler`.
- **Grad-CAM** heatmaps to visualize the attention mechanism.

Both phases are evaluated using Accuracy, Precision, Recall, F1 score, and Confusion Matrices.

## Expected Dataset Layout

Store images in class-named folders:

```text
dataset/
  acne/
    img_001.jpg
    img_002.jpg
  dark_spots/
  eczema/
  normal/
  rosacea/
  wrinkles/
```

The loader also accepts dataset roots that contain split folders such as `train/`, `val/`, `valid/`, or `test/`. In that case it automatically gathers class folders from those splits and normalizes names like `class0_normal` to `normal`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### Phase 1 (Baseline)

```bash
python3 -m src.skin_analysis.phase1.main \
  --data-dir /path/to/dataset \
  --output-dir outputs/phase1_baseline
```

### Phase 2 (Deep Learning)

```bash
# Full pipeline: Train, Evaluate, Grad-CAM
python3 -m src.skin_analysis.phase2.run_phase2 \
  --data-dir "/path/to/dataset" \
  --output-dir outputs/phase2_deep_learning \
  --run-gradcam

# Run ablation study (3 variants)
python3 -m src.skin_analysis.phase2.run_phase2 --run-ablation
```

## Predict One Image

```bash
python3 -m src.skin_analysis.phase1.predict \
  --image /path/to/image.jpg \
  --model outputs/phase1_baseline_msc6/trained_pipeline_random_forest.joblib \
  --label-map data/processed/label_mapping.json
```

By default, the predictor:

- tries to crop the largest detected face before feature extraction
- falls back to a center crop if no face is detected
- prints `uncertain` when the top probability is below `0.60`
- can save the exact crop used via `--save-crop`

## Outputs

Each run writes model artifacts and processed artifacts separately.

In the selected output directory, for example `outputs/phase1_baseline_msc6/`:

- `metrics_summary.csv`
- `classification_report_<model>.csv`
- `confusion_matrix_<model>.csv`
- `confusion_matrix_<model>.png`
- `classwise_metrics_<model>.png`
- `trained_pipeline_<model>.joblib`

In `data/processed/` by default:

- `features_dataset.csv`
- `train_test_split.csv`
- `label_mapping.json`
- `skipped_files.csv` when unreadable files are found

## Project Structure

```text
notebooks/
  phase1/
    01_eda_dataset_overview.ipynb
    02_feature_engineering.ipynb
    03_model_training.ipynb
    04_evaluation_and_interpretation.ipynb
  phase2/
    05_phase2_deep_learning.ipynb
src/skin_analysis/
  phase1/
    __init__.py
    data.py
    evaluate.py
    features.py
    main.py
    models.py
    pipeline.py
    predict.py
  phase2/
    __init__.py
    augment.py
    config.py
    evaluate_phase2.py
    gradcam.py
    model.py
    run_phase2.py
    train.py
```

## Notebook Workflow

The notebooks are modular and mirror the Phase 1 workflow:

- `01_eda_dataset_overview.ipynb`: dataset balance, image sizes, and sample inspection
- `02_feature_engineering.ipynb`: HSV histogram, LBP, GLCM, and combined feature vectors
- `03_model_training.ipynb`: `X` and `y` creation, stratified split, SVM and Random Forest training
- `04_evaluation_and_interpretation.ipynb`: confusion matrices, class-wise metrics, and report-ready interpretation

They import functions from `src/skin_analysis/` rather than duplicating the implementation.

## Why These Features

- Rosacea is often color-dominant because redness is a strong cue, so HSV histograms help capture that.
- Eczema is often texture-dominant because of dry and flaky visual patterns, so LBP and GLCM are important.

That feature reasoning is reflected directly in the implementation so it is explainable in a report or viva.
