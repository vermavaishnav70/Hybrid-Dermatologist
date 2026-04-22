from pathlib import Path
import json
import joblib
import numpy as np

from src.skin_analysis.data import load_and_preprocess_image
from src.skin_analysis.features import extract_features

image_path = Path("/Users/vaishnavverma/Downloads/1.jpeg")
model_path = Path("outputs/phase1_baseline/trained_pipeline_random_forest.joblib")
ocsvm_path = model_path.parent / "trained_ocsvm.joblib"
label_map_path = Path("data/processed/label_mapping.json")

model = joblib.load(model_path)
if ocsvm_path.exists():
    ocsvm_model = joblib.load(ocsvm_path)
else:
    ocsvm_model = None
label_mapping = json.loads(label_map_path.read_text())

loaded = load_and_preprocess_image(image_path, image_size=(224, 224))
if loaded is None:
    raise ValueError("Image could not be read.")

image_bgr, _ = loaded
features = extract_features(image_bgr).reshape(1, -1)

if ocsvm_model is not None:
    # ── Stage 1: Skin Detection Gate ──────────────────────────────────────
    skin_pred = ocsvm_model.predict(features)[0]
    score = ocsvm_model.decision_function(features)[0]
    
    print(f"Stage 1 - Skin detector score: {score:.4f} (positive = skin)")
    if (skin_pred == -1) or (score < 0.5):
        print(
            "\n⚠️  This image was rejected by the Stage 1 Skin Detector.\n"
            "    It falls outside the learned distribution (likely not human skin).\n"
            "    Classification skipped."
        )
        raise SystemExit(0)
    print("✅  Stage 1 - Passed: Image is within the skin distribution.\n")
    # ──────────────────────────────────────────────────────────────────────

pred = model.predict(features)[0]
proba = model.predict_proba(features)[0]
classes = model.classes_

print("Predicted class:", label_mapping.get(pred, pred))
print("\nClass probabilities:")
for cls, p in sorted(zip(classes, proba), key=lambda x: x[1], reverse=True):
    print(f"{label_mapping.get(cls, cls):15s} {p:.4f}")

