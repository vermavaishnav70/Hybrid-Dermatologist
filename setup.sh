#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  Hybrid Skin Analysis System — One-command setup
# ═══════════════════════════════════════════════════════════════════════════
#
#  Usage: bash setup.sh
#
#  This script:
#    1. Creates a Python virtual environment (.venv)
#    2. Installs all dependencies from requirements.txt
#    3. Creates necessary output directories
#    4. Verifies the installation

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  Hybrid Skin Analysis System — Setup"
echo "═══════════════════════════════════════════════════════════════"

# ── Virtual environment ───────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created at .venv/"
else
    echo "✓ Virtual environment already exists at .venv/"
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# ── Dependencies ──────────────────────────────────────────────────────────
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ── Output directories ────────────────────────────────────────────────────
echo "Creating output directories..."
mkdir -p outputs/phase1_baseline
mkdir -p outputs/phase2_deep_learning
mkdir -p outputs/phase3_hybrid
mkdir -p outputs/phase3_hybrid/feature_cache/train
mkdir -p outputs/phase3_hybrid/feature_cache/val
mkdir -p outputs/gpt4v_baseline

# ── Verification ──────────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torchvision; print(f'  torchvision: {torchvision.__version__}')"
python3 -c "import sklearn; print(f'  scikit-learn: {sklearn.__version__}')"
python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
python3 -c "import streamlit; print(f'  Streamlit: {streamlit.__version__}')"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete! Activate with: source .venv/bin/activate"
echo "  Then run: make train"
echo "═══════════════════════════════════════════════════════════════"
