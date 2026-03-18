from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from .features import extract_features


def _default_output_dir() -> Path:
    preferred_dirs = [
        Path("outputs/phase1_baseline_msc6"),
        Path("outputs/phase1_baseline"),
    ]
    return next((path for path in preferred_dirs if path.exists()), preferred_dirs[0])


def _default_label_map_path() -> Path:
    preferred_paths = [
        Path("data/processed/label_mapping.json"),
        Path("outputs/phase1_baseline_msc6/label_mapping.json"),
        Path("outputs/phase1_baseline/label_mapping.json"),
    ]
    return next((path for path in preferred_paths if path.exists()), preferred_paths[0])


def parse_args() -> argparse.Namespace:
    output_dir = _default_output_dir()
    label_map_path = _default_label_map_path()

    parser = argparse.ArgumentParser(
        description="Predict a skin-condition label for a single image using the Phase 1 classical ML model."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=output_dir / "trained_pipeline_random_forest.joblib",
        help="Path to a trained model artifact.",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=label_map_path,
        help="Path to the saved label mapping JSON.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square resize target used during training.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="If the top class probability is below this value, print 'uncertain' instead of forcing a label.",
    )
    parser.add_argument(
        "--face-padding",
        type=float,
        default=0.20,
        help="Extra padding ratio applied around the detected face crop.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="How many class probabilities to print.",
    )
    parser.add_argument(
        "--no-face-crop",
        action="store_true",
        help="Disable face-focused cropping and use the full image instead.",
    )
    parser.add_argument(
        "--save-crop",
        type=Path,
        default=None,
        help="Optional path to save the exact crop used for prediction.",
    )
    return parser.parse_args()


def _load_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image_bgr


def _load_label_mapping(label_map_path: Path) -> dict[str, str]:
    if not label_map_path.exists():
        return {}
    with label_map_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _resize_and_normalize(image_bgr: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(image_bgr, image_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    image_shape: tuple[int, int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return x0, y0, x1, y1


def _largest_face_box(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.equalizeHist(grayscale)

    cascade_paths = [
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
    ]

    best_box: tuple[int, int, int, int] | None = None
    best_area = -1
    for cascade_path in cascade_paths:
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            continue

        faces = classifier.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64),
        )
        for x, y, w, h in faces:
            area = int(w * h)
            if area > best_area:
                best_area = area
                best_box = int(x), int(y), int(w), int(h)

        if best_box is not None:
            break

    return best_box


def _center_crop(image_bgr: np.ndarray, crop_ratio: float = 0.85) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    side = int(min(height, width) * crop_ratio)
    x0 = max(0, (width - side) // 2)
    y0 = max(0, (height - side) // 2)
    x1 = x0 + side
    y1 = y0 + side
    return image_bgr[y0:y1, x0:x1]


def prepare_image_for_prediction(
    image_path: Path,
    image_size: tuple[int, int] = (224, 224),
    use_face_crop: bool = True,
    face_padding: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    raw_image = _load_image(image_path)
    crop_mode = "full_image"
    face_box = None
    cropped_image = raw_image

    if use_face_crop:
        face_box = _largest_face_box(raw_image)
        if face_box is not None:
            x, y, w, h = face_box
            x0, y0, x1, y1 = _expand_box(
                x=x,
                y=y,
                w=w,
                h=h,
                image_shape=raw_image.shape,
                padding_ratio=face_padding,
            )
            cropped_image = raw_image[y0:y1, x0:x1]
            crop_mode = "largest_face"
        else:
            cropped_image = _center_crop(raw_image)
            crop_mode = "center_crop_fallback"

    normalized = _resize_and_normalize(cropped_image, image_size=image_size)
    metadata = {
        "crop_mode": crop_mode,
        "face_box": face_box,
        "original_shape": tuple(int(value) for value in raw_image.shape[:2]),
        "crop_shape": tuple(int(value) for value in cropped_image.shape[:2]),
    }
    return normalized, cropped_image, metadata


def _format_label(label: str, label_mapping: dict[str, str]) -> str:
    return label_mapping.get(label, label.replace("_", " "))


def main() -> None:
    args = parse_args()

    model = joblib.load(args.model)
    label_mapping = _load_label_mapping(args.label_map)
    processed_image, cropped_image, metadata = prepare_image_for_prediction(
        image_path=args.image,
        image_size=(args.image_size, args.image_size),
        use_face_crop=not args.no_face_crop,
        face_padding=args.face_padding,
    )

    feature_vector = extract_features(processed_image).reshape(1, -1)
    predicted_label = model.predict(feature_vector)[0]

    probabilities: list[tuple[str, float]] = []
    if hasattr(model, "predict_proba"):
        probability_vector = model.predict_proba(feature_vector)[0]
        probabilities = list(zip(model.classes_, probability_vector.tolist()))
        probabilities.sort(key=lambda item: item[1], reverse=True)
        top_label, top_probability = probabilities[0]
    else:
        top_label = predicted_label
        top_probability = 1.0

    display_prediction = _format_label(top_label, label_mapping)
    final_prediction = display_prediction if top_probability >= args.confidence_threshold else "uncertain"

    print(f"Input image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Crop used: {metadata['crop_mode']}")
    print(f"Original shape: {metadata['original_shape']}")
    print(f"Crop shape: {metadata['crop_shape']}")
    if metadata["face_box"] is not None:
        print(f"Detected face box: {metadata['face_box']}")

    print(f"\nPrediction: {final_prediction}")
    print(f"Best class: {display_prediction}")
    print(f"Best class probability: {top_probability:.4f}")

    if top_probability < args.confidence_threshold:
        print(
            f"Confidence is below the threshold ({args.confidence_threshold:.2f}), "
            "so the result is marked as uncertain."
        )

    if probabilities:
        print("\nClass probabilities:")
        for label, probability in probabilities[: args.top_k]:
            print(f"{_format_label(label, label_mapping):15s} {probability:.4f}")

    if args.save_crop is not None:
        args.save_crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_crop), cropped_image)
        print(f"\nSaved inference crop to: {args.save_crop}")

    print(
        "\nNote: this Phase 1 model uses handcrafted global color/texture features, "
        "so even visible acne can be underweighted when the lesions occupy a small area or the image has distracting content."
    )


if __name__ == "__main__":
    main()
