# ═══════════════════════════════════════════════════════════════════════════
#  Hybrid Skin Analysis System — Makefile
# ═══════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    make install      # Set up virtual environment and install dependencies
#    make train        # Train all three models sequentially
#    make eval         # Evaluate all models on the same test set
#    make ablation     # Run component-removal ablation study
#    make gradcam      # Generate Grad-CAM heatmaps for all models
#    make calibrate    # Run calibration (temperature scaling + ECE)
#    make demo         # Launch Streamlit demo
#    make docker       # Build and run Docker container
#    make all          # Full pipeline: train + eval + ablation + gradcam

.PHONY: install train train-a train-b train-c eval ablation gradcam calibrate demo docker all clean

# ── Configuration ─────────────────────────────────────────────────────────
PYTHON      = python3
DATA_DIR    = data/raw/Multi-Class\ Skin\ Condition\ Image\ Dataset\ \(MSC-6\)
OUTPUT_A    = outputs/phase1_baseline
OUTPUT_B    = outputs/phase2_deep_learning
OUTPUT_C    = outputs/phase3_hybrid

# ── Setup ─────────────────────────────────────────────────────────────────
install:
	bash setup.sh

# ── Phase 1: Classical ML (SVM + Random Forest) ──────────────────────────
train-a:
	@echo "═══ Phase 1: Classical ML ═══"
	$(PYTHON) -m src.skin_analysis.phase1.main

# ── Phase 2: Deep Learning (EfficientNet-B3 + CBAM) ──────────────────────
train-b:
	@echo "═══ Phase 2: EfficientNet-B3 + CBAM ═══"
	$(PYTHON) -m src.skin_analysis.phase2.run_phase2 \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_B) \
		--run-gradcam

# ── Phase 3: Hybrid Fusion (Model C) ─────────────────────────────────────
train-c:
	@echo "═══ Phase 3: Hybrid Fusion ═══"
	$(PYTHON) -m src.skin_analysis.phase3.train_c

# ── Train all models ─────────────────────────────────────────────────────
train: train-a train-b train-c

# ── Evaluation ────────────────────────────────────────────────────────────
eval:
	@echo "═══ Evaluating all models ═══"
	$(PYTHON) -m src.skin_analysis.phase3.evaluate_all

# ── Calibration (Temperature Scaling + ECE) ──────────────────────────────
calibrate:
	@echo "═══ Calibration ═══"
	$(PYTHON) -c "\
from src.skin_analysis.phase3.calibrate import calibrate_and_report; \
from src.skin_analysis.phase3.config import Phase3Config; \
from src.skin_analysis.phase3.model_c import HybridFusionModel; \
from src.skin_analysis.phase3.dataset import build_hybrid_dataloaders; \
from src.skin_analysis.phase3.train_c import detect_device; \
import torch; \
cfg = Phase3Config(); \
device = detect_device(); \
_, val_loader, _, _ = build_hybrid_dataloaders(cfg); \
model = HybridFusionModel(pretrained=False); \
ckpt = cfg.output_dir / 'best_model_hybrid.pth'; \
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)); \
calibrate_and_report(model, val_loader, device, cfg.output_dir, 'hybrid_fusion', is_hybrid=True)"

# ── Ablation study ────────────────────────────────────────────────────────
ablation:
	@echo "═══ Ablation Study ═══"
	$(PYTHON) -m src.skin_analysis.phase3.ablation

# ── Grad-CAM ─────────────────────────────────────────────────────────────
gradcam:
	@echo "═══ Grad-CAM ═══"
	$(PYTHON) -c "\
from src.skin_analysis.phase3.gradcam_hybrid import generate_gradcam_grid; \
from src.skin_analysis.phase3.config import Phase3Config; \
from src.skin_analysis.phase3.model_c import HybridFusionModel; \
from src.skin_analysis.phase3.dataset import HybridSkinDataset, build_val_transforms; \
from src.skin_analysis.phase3.train_c import detect_device; \
import torch; \
cfg = Phase3Config(); \
device = detect_device(); \
model = HybridFusionModel(pretrained=False); \
ckpt = cfg.output_dir / 'best_model_hybrid.pth'; \
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)); \
val_ds = HybridSkinDataset(cfg.data_dir / 'val', cfg.class_names, build_val_transforms(cfg), cfg.feature_cache_dir / 'val'); \
generate_gradcam_grid(model, val_ds, cfg, device)"

# ── GPT-4V baseline (requires OPENAI_API_KEY) ────────────────────────────
gpt4v:
	@echo "═══ GPT-4V Zero-Shot Baseline ═══"
	$(PYTHON) scripts/gpt4v_baseline.py

# ── Streamlit demo ────────────────────────────────────────────────────────
demo:
	streamlit run app/streamlit_app.py

# ── Docker ────────────────────────────────────────────────────────────────
docker:
	docker build -f docker/Dockerfile -t hybrid-skin-analysis .
	docker run --rm hybrid-skin-analysis

# ── Full pipeline ─────────────────────────────────────────────────────────
all: train eval ablation gradcam calibrate
	@echo "═══ Full pipeline complete! ═══"

# ── Clean generated artefacts ─────────────────────────────────────────────
clean:
	rm -rf outputs/phase3_hybrid/feature_cache
	rm -rf outputs/phase3_hybrid/ablation_no_attention
	@echo "Cleaned Phase 3 cache artefacts"
