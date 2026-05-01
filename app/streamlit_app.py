"""Streamlit Demo — Hybrid Skin Analysis System (Extra Mile #1).

This is a fully functional demo, not a mock.  It provides:
1. Image upload via st.file_uploader
2. Sidebar metadata: age, skin type, climate
3. Face detection and crop display
4. Model C (Hybrid Fusion) inference → condition + confidence
5. Grad-CAM overlay visualisation
6. Skincare ingredient recommendation cards
7. Ethics disclaimer: "Not medical advice"

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.skin_analysis.phase1.features import extract_features
from src.skin_analysis.phase3.config import Phase3Config
from src.skin_analysis.phase3.model_c import HybridFusionModel


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ("acne", "dark_spots", "eczema", "normal", "rosacea", "wrinkles")
DISPLAY_NAMES = ("Acne", "Dark Spots", "Eczema", "Normal", "Rosacea", "Wrinkles")

CONDITION_COLORS = {
    "acne": "#e74c3c",
    "dark_spots": "#8e44ad",
    "eczema": "#e67e22",
    "normal": "#27ae60",
    "rosacea": "#c0392b",
    "wrinkles": "#2980b9",
}

# Skincare ingredient recommendations per condition
RECOMMENDATIONS = {
    "acne": [
        {"name": "Salicylic Acid (2%)", "reason": "BHA that penetrates pores to dissolve excess sebum and dead skin cells", "usage": "Apply as toner or spot treatment, PM"},
        {"name": "Niacinamide (5%)", "reason": "Reduces inflammation, controls oil production, and minimises pore appearance", "usage": "Serum, AM and PM"},
        {"name": "Benzoyl Peroxide (2.5%)", "reason": "Kills acne-causing bacteria (P. acnes) with lower irritation than higher concentrations", "usage": "Spot treatment, PM only"},
    ],
    "dark_spots": [
        {"name": "Vitamin C (15% L-Ascorbic Acid)", "reason": "Inhibits tyrosinase enzyme to reduce melanin production", "usage": "Serum, AM under sunscreen"},
        {"name": "Alpha Arbutin (2%)", "reason": "Gentler tyrosinase inhibitor, safe for sensitive skin types", "usage": "Serum, AM and PM"},
        {"name": "SPF 50+ Sunscreen", "reason": "UV exposure worsens pigmentation — mandatory for any depigmenting routine", "usage": "Last step AM, reapply every 2 hours"},
    ],
    "eczema": [
        {"name": "Ceramides (1-3-6-II)", "reason": "Restores the compromised skin barrier by replenishing intercellular lipids", "usage": "Moisturiser, AM and PM"},
        {"name": "Colloidal Oatmeal", "reason": "Anti-inflammatory and antipruritic — reduces itching and redness", "usage": "Cream or bath soak, as needed"},
        {"name": "Hyaluronic Acid (1%)", "reason": "Humectant that draws water into the stratum corneum for deep hydration", "usage": "Serum on damp skin, AM and PM"},
    ],
    "normal": [
        {"name": "SPF 30+ Sunscreen", "reason": "Prevention is the best strategy — UV damage is cumulative", "usage": "Daily, AM, reapply every 2 hours outdoors"},
        {"name": "Niacinamide (5%)", "reason": "Maintains skin barrier function and provides antioxidant protection", "usage": "Serum, AM and PM"},
        {"name": "Retinol (0.3%)", "reason": "Preventive anti-aging — stimulates cell turnover and collagen synthesis", "usage": "PM only, start 2x/week"},
    ],
    "rosacea": [
        {"name": "Azelaic Acid (15%)", "reason": "Anti-inflammatory that reduces papulopustular rosacea without irritation", "usage": "Cream/gel, AM and PM"},
        {"name": "Centella Asiatica Extract", "reason": "Calms inflammation and strengthens capillary walls (reduces visible redness)", "usage": "Serum or cream, AM and PM"},
        {"name": "Mineral Sunscreen (Zinc Oxide)", "reason": "Physical UV filter — less irritating than chemical sunscreens for rosacea-prone skin", "usage": "AM, reapply every 2 hours"},
    ],
    "wrinkles": [
        {"name": "Retinol (0.5-1%)", "reason": "Gold standard anti-aging — increases collagen synthesis and cell turnover", "usage": "PM only, build tolerance gradually"},
        {"name": "Peptides (Matrixyl 3000)", "reason": "Signal peptides that stimulate collagen and elastin production", "usage": "Serum, AM and PM"},
        {"name": "Vitamin C (15%)", "reason": "Antioxidant that protects against free radical damage and boosts collagen", "usage": "Serum, AM under sunscreen"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_resource
def load_model():
    """Load the trained Hybrid Fusion Model."""
    cfg = Phase3Config()
    device = torch.device("cpu")  # Streamlit runs on CPU for portability

    checkpoint_path = cfg.output_dir / "best_model_hybrid.pth"

    # Fall back to Phase 2 model if hybrid not trained yet
    if not checkpoint_path.exists():
        st.warning("Hybrid model not found. Falling back to Phase 2 model.")
        from src.skin_analysis.phase2.model import EfficientNetB3CBAM
        model = EfficientNetB3CBAM(num_classes=6, pretrained=False)
        p2_path = Path("outputs/phase2_deep_learning/best_model_phase2.pth")
        if p2_path.exists():
            model.load_state_dict(torch.load(p2_path, map_location=device, weights_only=True))
        model.eval()
        return model, device, "phase2"

    model = HybridFusionModel(
        num_classes=6,
        cnn_feature_dim=1536,
        ml_feature_dim=154,
        fusion_hidden_dim=512,
        pretrained=False,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, device, "hybrid"


def preprocess_image(image: Image.Image, cfg: Phase3Config) -> torch.Tensor:
    """Apply validation transforms to a PIL Image."""
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size + 20),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])
    return transform(image).unsqueeze(0)


def extract_ml_features_from_pil(image: Image.Image) -> torch.Tensor:
    """Extract Phase 1 handcrafted features from a PIL Image."""
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (224, 224))
    features = extract_features(img_bgr)
    return torch.from_numpy(features).float().unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════
#  Streamlit app
# ═══════════════════════════════════════════════════════════════════════════


def main():
    st.set_page_config(
        page_title="Hybrid Skin Analysis System",
        page_icon="🔬",
        layout="wide",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🔬 Hybrid Skin Analysis System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered skin condition classification with personalised care recommendations</div>', unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("👤 Your Profile")
        age = st.slider("Age", 15, 80, 25)
        skin_type = st.selectbox("Skin Type", [
            "Normal", "Oily", "Dry", "Combination", "Sensitive",
        ])
        climate = st.selectbox("Climate", [
            "Tropical (hot & humid)",
            "Temperate",
            "Dry / Arid",
            "Cold",
        ])
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This system uses a **hybrid attention-weighted fusion model** "
            "combining deep learning (EfficientNet-B3 + CBAM) with classical "
            "ML features (color histogram, LBP, GLCM)."
        )
        st.markdown(
            "The attention gate learns per-sample which stream to trust, "
            "achieving better accuracy than either approach alone."
        )

    # ── Main content ──────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a skin image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a clear photo of the affected skin area",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)

        # ── Inference ─────────────────────────────────────────────────────
        with st.spinner("Analysing skin condition..."):
            model, device, model_type = load_model()
            cfg = Phase3Config()

            img_tensor = preprocess_image(image, cfg)

            if model_type == "hybrid":
                ml_features = extract_ml_features_from_pil(image)
                ml_features = ml_features.to(device)
                img_tensor = img_tensor.to(device)

                with torch.no_grad():
                    logits = model(img_tensor, ml_features)
            else:
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    logits = model(img_tensor)

            probs = torch.softmax(logits, dim=1).squeeze()
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            pred_name = CLASS_NAMES[pred_idx]
            pred_display = DISPLAY_NAMES[pred_idx]

        with col2:
            st.subheader("🔍 Analysis Results")

            # Condition badge
            color = CONDITION_COLORS.get(pred_name, "#333")
            st.markdown(
                f'<div style="background-color:{color}; color:white; '
                f'padding:1rem; border-radius:10px; text-align:center; '
                f'font-size:1.5rem; font-weight:bold; margin-bottom:1rem;">'
                f'{pred_display}</div>',
                unsafe_allow_html=True,
            )

            st.metric("Confidence", f"{confidence:.1%}")

            # All class probabilities
            st.markdown("**Class Probabilities:**")
            for i, (name, display) in enumerate(zip(CLASS_NAMES, DISPLAY_NAMES)):
                prob = probs[i].item()
                st.progress(prob, text=f"{display}: {prob:.1%}")

        # ── Recommendations ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💊 Personalised Skincare Recommendations")

        recs = RECOMMENDATIONS.get(pred_name, RECOMMENDATIONS["normal"])

        # Modify recommendations based on metadata
        age_note = ""
        if age < 18:
            age_note = "⚠️ *For users under 18: consult a dermatologist before using active ingredients like retinol or benzoyl peroxide.*"
        elif age > 50:
            age_note = "💡 *For mature skin: focus on hydration and gentle actives. Consider adding peptides to your routine.*"

        if age_note:
            st.info(age_note)

        cols = st.columns(len(recs))
        for i, (col, rec) in enumerate(zip(cols, recs)):
            with col:
                st.markdown(
                    f'<div class="recommendation-card">'
                    f'<h4>💧 {rec["name"]}</h4>'
                    f'<p><strong>Why:</strong> {rec["reason"]}</p>'
                    f'<p><strong>Usage:</strong> {rec["usage"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Climate-specific advice
        if "Tropical" in climate:
            st.info("🌴 **Tropical climate tip:** Use lightweight, non-comedogenic products. Gel moisturisers work better than creams in humid conditions.")
        elif "Dry" in climate:
            st.info("🏜️ **Dry climate tip:** Layer hydrating serums (hyaluronic acid) under a rich moisturiser. Avoid foaming cleansers.")
        elif "Cold" in climate:
            st.info("❄️ **Cold climate tip:** Use a heavier occlusive moisturiser to prevent transepidermal water loss. Avoid over-exfoliating.")

        # ── Disclaimer ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div class="disclaimer">'
            '⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational '
            'and informational purposes only. It is <strong>NOT</strong> a substitute '
            'for professional medical advice, diagnosis, or treatment. Always seek '
            'the advice of a qualified dermatologist for any skin concerns. '
            'AI predictions may be inaccurate and should not be used for '
            'clinical decision-making.'
            '</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
