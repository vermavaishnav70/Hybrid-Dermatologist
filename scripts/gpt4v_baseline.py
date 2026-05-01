"""GPT-4V zero-shot baseline for skin condition classification.

This script sends test images to GPT-4V (or GPT-4o) with a zero-shot
classification prompt and records the predictions for comparison with
the trained models in the ablation table.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/gpt4v_baseline.py --num-images 50

Cost estimate: ~$0.01 per image (GPT-4o with low-detail).

Why include GPT-4V baseline:
  The ablation table must include a state-of-the-art foundation model
  baseline to contextualise the trained models' performance.  GPT-4V
  is the strongest zero-shot medical image classifier available.
  Showing that our fine-tuned hybrid model exceeds GPT-4V validates
  the domain-specific training approach.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from tqdm.auto import tqdm


# Class names must match the project's canonical order
CLASS_NAMES = ("acne", "dark_spots", "eczema", "normal", "rosacea", "wrinkles")
DISPLAY_NAMES = ("acne", "dark spots", "eczema", "normal", "rosacea", "wrinkles")


def encode_image_base64(image_path: str | Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_image_gpt4v(
    client,
    image_path: str | Path,
    model: str = "gpt-4o",
) -> str | None:
    """Send a single image to GPT-4V/4o for zero-shot classification.

    Returns the predicted class name (lowercase, underscore-separated)
    or None if the API call fails.
    """
    base64_image = encode_image_base64(image_path)

    prompt = (
        "You are a dermatologist AI assistant. "
        "Classify this skin image into exactly ONE of the following categories:\n"
        "- acne\n"
        "- dark_spots\n"
        "- eczema\n"
        "- normal\n"
        "- rosacea\n"
        "- wrinkles\n\n"
        "Respond with ONLY the category name, nothing else. "
        "Use underscores, not spaces (e.g., 'dark_spots' not 'dark spots')."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",  # Cost-effective
                            },
                        },
                    ],
                }
            ],
            max_tokens=20,
            temperature=0.0,  # Deterministic for reproducibility
        )
        prediction = response.choices[0].message.content.strip().lower()
        # Normalise prediction
        prediction = prediction.replace(" ", "_").replace("-", "_")
        if prediction in CLASS_NAMES:
            return prediction
        # Try fuzzy match
        for name in CLASS_NAMES:
            if name in prediction:
                return name
        return prediction  # Return raw even if not in class list
    except Exception as e:
        print(f"  ⚠ API error for {image_path}: {e}")
        return None


def run_gpt4v_baseline(
    data_dir: str | Path,
    output_dir: str | Path,
    num_images: int = 50,
    model: str = "gpt-4o",
) -> dict:
    """Run GPT-4V zero-shot classification on a subset of val images.

    Args:
        data_dir: path to the MSC-6 val directory
        output_dir: where to save results
        num_images: max images to classify (for cost control)
        model: OpenAI model name

    Returns:
        dict with accuracy, f1_macro, f1_weighted, per_class_f1
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect test images
    samples = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        label = folder.name.strip().lower()
        label = label.replace(" ", "_")
        # Remove class prefix like "class0_"
        import re
        label = re.sub(r"^class\d+_", "", label)
        if label not in CLASS_NAMES:
            continue
        label_idx = CLASS_NAMES.index(label)
        for img_path in sorted(folder.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((img_path, label_idx, label))

    if not samples:
        print(f"No images found in {data_dir}")
        return {}

    # Stratified subsample
    rng = np.random.RandomState(42)
    if len(samples) > num_images:
        indices = rng.choice(len(samples), size=num_images, replace=False)
        samples = [samples[i] for i in indices]

    print(f"\n╔══ GPT-4V Zero-Shot Baseline ══╗")
    print(f"  Model: {model}")
    print(f"  Images: {len(samples)}")
    print(f"  Estimated cost: ~${len(samples) * 0.01:.2f}")

    # Run classification
    predictions = []
    true_labels = []
    results_log = []

    for img_path, true_idx, true_name in tqdm(samples, desc="  GPT-4V"):
        pred_name = classify_image_gpt4v(client, img_path, model=model)

        if pred_name is None:
            continue

        pred_idx = CLASS_NAMES.index(pred_name) if pred_name in CLASS_NAMES else -1

        predictions.append(pred_idx)
        true_labels.append(true_idx)
        results_log.append({
            "image": str(img_path),
            "true_label": true_name,
            "predicted": pred_name,
            "correct": pred_name == true_name,
        })

        # Rate limiting
        time.sleep(0.5)

    # Filter out invalid predictions
    valid_mask = [p >= 0 for p in predictions]
    predictions = [p for p, v in zip(predictions, valid_mask) if v]
    true_labels = [t for t, v in zip(true_labels, valid_mask) if v]

    if not predictions:
        print("  No valid predictions received.")
        return {}

    # Compute metrics
    acc = accuracy_score(true_labels, predictions)
    f1_m = f1_score(true_labels, predictions, average="macro", zero_division=0)
    f1_w = f1_score(true_labels, predictions, average="weighted", zero_division=0)

    report = classification_report(
        true_labels, predictions,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    per_class_f1 = {}
    for name in CLASS_NAMES:
        if name in report:
            per_class_f1[name] = report[name]["f1-score"]

    print(f"\n  Results:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1 Macro:    {f1_m:.4f}")
    print(f"    F1 Weighted: {f1_w:.4f}")

    # Save results
    csv_path = output_dir / "gpt4v_baseline_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "true_label", "predicted", "correct"])
        writer.writeheader()
        writer.writerows(results_log)
    print(f"  Detailed results saved to {csv_path}")

    # Save summary
    summary_path = output_dir / "gpt4v_baseline_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Model", model])
        writer.writerow(["Images Evaluated", len(predictions)])
        writer.writerow(["Accuracy", f"{acc:.4f}"])
        writer.writerow(["F1 Macro", f"{f1_m:.4f}"])
        writer.writerow(["F1 Weighted", f"{f1_w:.4f}"])
        for name in CLASS_NAMES:
            writer.writerow([f"F1 {name}", f"{per_class_f1.get(name, 0.0):.4f}"])
    print(f"  Summary saved to {summary_path}")

    return {
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "per_class_f1": per_class_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-4V zero-shot skin classification")
    parser.add_argument(
        "--data-dir", type=str,
        default="data/raw/Multi-Class Skin Condition Image Dataset (MSC-6)/val",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/gpt4v_baseline")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    run_gpt4v_baseline(args.data_dir, args.output_dir, args.num_images, args.model)


if __name__ == "__main__":
    main()
