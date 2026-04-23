from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SPLIT_DIRECTORY_NAMES = {"train", "val", "valid", "test"}

CANONICAL_LABELS = (
    "acne",
    "dark_spots",
    "eczema",
    "normal",
    "rosacea",
    "wrinkles",
)

DISPLAY_LABELS = {
    "acne": "acne",
    "dark_spots": "dark spots",
    "eczema": "eczema",
    "normal": "normal",
    "rosacea": "rosacea",
    "wrinkles": "wrinkles",
}


def normalize_label_name(label: str) -> str:
    """Normalize folder names so spaces, dashes, and underscores behave the same."""
    normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def canonicalize_label(label: str, accept_label_aliases: bool = True) -> str:
    """Map a raw folder name to one canonical internal class label."""
    normalized = normalize_label_name(label)
    normalized = re.sub(r"^class\d+_", "", normalized)
    if not accept_label_aliases:
        return normalized

    alias_map = {canonical: canonical for canonical in CANONICAL_LABELS}
    alias_map.update(
        {
            "dark_spots": "dark_spots",
            "dark_spot": "dark_spots",
            "darkspots": "dark_spots",
            "dark_spots_class": "dark_spots",
            "dark_spots_condition": "dark_spots",
        }
    )
    return alias_map.get(normalized, normalized)


def _list_class_directories(data_dir: Path) -> list[Path]:
    immediate_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not immediate_dirs:
        return []

    normalized_dir_names = {normalize_label_name(path.name) for path in immediate_dirs}
    if normalized_dir_names.issubset(SPLIT_DIRECTORY_NAMES):
        class_dirs: list[Path] = []
        for split_dir in immediate_dirs:
            class_dirs.extend(sorted(path for path in split_dir.iterdir() if path.is_dir()))
        return class_dirs

    return immediate_dirs


def discover_dataset_records(
    data_dir: str | Path,
    valid_exts: set[str] | None = None,
    accept_label_aliases: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Discover image files from class-named folders without loading them into memory.

    This supports either direct class folders or dataset roots that contain split
    directories such as train/val/test with class folders inside each split.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    class_directories = _list_class_directories(root)
    if not class_directories:
        raise ValueError(f"No class folders were found inside: {root}")

    valid_extensions = {ext.lower() for ext in (valid_exts or VALID_IMAGE_EXTENSIONS)}
    discovered_records: list[dict[str, str]] = []
    skipped_records: list[dict[str, str]] = []
    observed_display_mapping: dict[str, str] = {}

    for class_dir in class_directories:
        canonical_label = canonicalize_label(class_dir.name, accept_label_aliases=accept_label_aliases)
        display_label = DISPLAY_LABELS.get(canonical_label, canonical_label.replace("_", " "))
        observed_display_mapping[canonical_label] = display_label
        image_files = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in valid_extensions
        ]
        if not image_files:
            skipped_records.append(
                {
                    "path": str(class_dir),
                    "label": canonical_label,
                    "reason": "no_image_files_found",
                }
            )
            continue

        for file_path in image_files:
            discovered_records.append(
                {
                    "path": str(file_path),
                    "label": canonical_label,
                    "display_label": display_label,
                }
            )

    if not discovered_records:
        raise ValueError(
            "No image files were found. Check that the dataset contains supported image files."
        )

    records_df = pd.DataFrame(discovered_records, columns=["path", "label", "display_label"])
    skipped_df = pd.DataFrame(skipped_records, columns=["path", "label", "reason"])
    return records_df, skipped_df, observed_display_mapping


def load_and_preprocess_image(
    image_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Load one image, resize it, normalize it, and return the original size."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    original_height, original_width = image.shape[:2]
    resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized, (original_width, original_height)


def load_data(
    data_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    valid_exts: set[str] | None = None,
    accept_label_aliases: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Load images from class-named folders and return resized image arrays plus metadata.

    This is convenient for notebooks and smaller datasets. For large datasets, use
    discover_dataset_records() plus sequential feature extraction to avoid high memory use.
    """
    records_df, skipped_df, label_mapping = discover_dataset_records(
        data_dir=data_dir,
        valid_exts=valid_exts,
        accept_label_aliases=accept_label_aliases,
    )

    images: list[np.ndarray] = []
    labels: list[str] = []
    display_labels: list[str] = []
    paths: list[str] = []
    original_sizes: list[tuple[int, int]] = []
    skipped_records = skipped_df.to_dict(orient="records")

    for row in records_df.itertuples(index=False):
        loaded = load_and_preprocess_image(row.path, image_size=image_size)
        if loaded is None:
            skipped_records.append(
                {
                    "path": row.path,
                    "label": row.label,
                    "reason": "corrupted_or_unreadable",
                }
            )
            continue

        image, original_size = loaded
        images.append(image)
        labels.append(row.label)
        display_labels.append(row.display_label)
        paths.append(row.path)
        original_sizes.append(original_size)

    if not images:
        raise ValueError(
            "No readable images were found. Check that the dataset contains supported image files."
        )

    bundle = {
        "images": images,
        "labels": np.asarray(labels),
        "display_labels": np.asarray(display_labels),
        "paths": np.asarray(paths),
        "original_sizes": np.asarray(original_sizes, dtype=np.int32),
        "label_mapping": label_mapping,
        "class_names": sorted(set(labels)),
    }
    final_skipped_df = pd.DataFrame(skipped_records, columns=["path", "label", "reason"])
    return bundle, final_skipped_df
