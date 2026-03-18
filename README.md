# Hybrid-Dermatologist

Phase 1 baseline for a skin condition classification project using handcrafted computer vision features and classical machine learning.

## Scope

This repository currently implements the mandatory classical ML phase:

- HSV color histogram features for redness-sensitive signals such as rosacea
- Texture features using LBP and GLCM for texture-heavy classes such as eczema
- Two baseline classifiers:
  - SVM with RBF kernel
  - Random Forest
- Evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Confusion matrix

No deep learning is used in this phase.

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

```bash
python3 -m src.skin_analysis.main \
  --data-dir /path/to/dataset \
  --output-dir outputs/phase1_baseline
```

## Predict One Image

```bash
python3 -m src.skin_analysis.predict \
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
  01_eda_dataset_overview.ipynb
  02_feature_engineering.ipynb
  03_model_training.ipynb
  04_evaluation_and_interpretation.ipynb
src/skin_analysis/
  data.py
  evaluate.py
  features.py
  main.py
  models.py
  pipeline.py
  predict.py
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
