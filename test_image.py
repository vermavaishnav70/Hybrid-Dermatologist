from pathlib import Path
import json
import joblib
import numpy as np

from src.skin_analysis.data import load_and_preprocess_image
from src.skin_analysis.features import extract_features

image_path = Path("/Users/vaishnavverma/Downloads/Photo Dec 27 2025 copy.jpg")
model_path = Path("outputs/phase1_baseline_msc6/trained_pipeline_random_forest.joblib")
label_map_path = Path("outputs/phase1_baseline_msc6/label_mapping.json")

model = joblib.load(model_path)
label_mapping = json.loads(label_map_path.read_text())

loaded = load_and_preprocess_image(image_path, image_size=(224, 224))
if loaded is None:
    raise ValueError("Image could not be read.")

image_bgr, _ = loaded
features = extract_features(image_bgr).reshape(1, -1)

pred = model.predict(features)[0]
proba = model.predict_proba(features)[0]
classes = model.classes_

print("Predicted class:", label_mapping.get(pred, pred))
print("\nClass probabilities:")
for cls, p in sorted(zip(classes, proba), key=lambda x: x[1], reverse=True):
    print(f"{label_mapping.get(cls, cls):15s} {p:.4f}")

