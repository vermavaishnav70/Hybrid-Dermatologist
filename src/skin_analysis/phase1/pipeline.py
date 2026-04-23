from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .data import discover_dataset_records, load_and_preprocess_image
from .evaluate import evaluate_model
from .features import extract_features
from .models import train_model, train_ocsvm


def _save_feature_cache(
    X: np.ndarray,
    labels: np.ndarray,
    display_labels: np.ndarray,
    paths: np.ndarray,
    output_dir: Path,
) -> Path:
    feature_columns = [f"feature_{index:03d}" for index in range(X.shape[1])]
    feature_df = pd.DataFrame(X, columns=feature_columns)
    feature_df["label"] = labels
    feature_df["display_label"] = display_labels
    feature_df["path"] = paths
    cache_path = output_dir / "features_dataset.csv"
    feature_df.to_csv(cache_path, index=False)
    return cache_path


def _build_feature_dataset(
    records_df: pd.DataFrame,
    image_size: tuple[int, int],
    hist_bins: int,
    lbp_points: int,
    lbp_radius: int,
    glcm_distances: Iterable[int],
    glcm_angles: Iterable[float],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    feature_rows: list[np.ndarray] = []
    kept_records: list[dict[str, object]] = []
    skipped_records: list[dict[str, str]] = []

    iterator = tqdm(
        records_df.itertuples(index=False),
        total=len(records_df),
        desc="Extracting handcrafted features",
        unit="image",
    )
    for row in iterator:
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
        feature_vector = extract_features(
            image_bgr=image,
            hist_bins=hist_bins,
            lbp_points=lbp_points,
            lbp_radius=lbp_radius,
            glcm_distances=glcm_distances,
            glcm_angles=glcm_angles,
        )
        feature_rows.append(feature_vector)
        kept_records.append(
            {
                "path": row.path,
                "label": row.label,
                "display_label": row.display_label,
                "original_width": original_size[0],
                "original_height": original_size[1],
            }
        )

    if not feature_rows:
        raise ValueError("Feature extraction produced no valid samples.")

    feature_matrix = np.vstack(feature_rows).astype(np.float32)
    kept_records_df = pd.DataFrame(kept_records)
    skipped_df = pd.DataFrame(skipped_records, columns=["path", "label", "reason"])
    return feature_matrix, kept_records_df, skipped_df


def _save_split_manifest(
    labels: np.ndarray,
    paths: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    output_dir: Path,
) -> Path:
    split_df = pd.DataFrame(
        {
            "path": paths,
            "label": labels,
            "split": np.where(np.isin(np.arange(len(labels)), train_indices), "train", "test"),
        }
    )
    manifest_path = output_dir / "train_test_split.csv"
    split_df.to_csv(manifest_path, index=False)
    return manifest_path


def run_phase1_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    processed_dir: str | Path | None = None,
    image_size: tuple[int, int] = (224, 224),
    test_size: float = 0.2,
    random_state: int = 42,
    hist_bins: int = 32,
    lbp_points: int = 24,
    lbp_radius: int = 3,
    glcm_distances: Iterable[int] = (1, 2),
    glcm_angles: Iterable[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, object]:
    """Run the full Phase 1 classical ML baseline and save all requested artifacts."""
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    if processed_dir is None:
        raw_parent = Path(data_dir).resolve().parent
        if raw_parent.name == "raw":
            processed_root = raw_parent.parent / "processed"
        else:
            processed_root = Path(data_dir).resolve().parent / "data" / "processed"
    else:
        processed_root = Path(processed_dir).expanduser().resolve()
    processed_root.mkdir(parents=True, exist_ok=True)

    records_df, discovery_skipped_df, label_mapping = discover_dataset_records(data_dir=data_dir)
    X, feature_records_df, extraction_skipped_df = _build_feature_dataset(
        records_df=records_df,
        image_size=image_size,
        hist_bins=hist_bins,
        lbp_points=lbp_points,
        lbp_radius=lbp_radius,
        glcm_distances=glcm_distances,
        glcm_angles=glcm_angles,
    )
    skipped_df = pd.concat([discovery_skipped_df, extraction_skipped_df], ignore_index=True)

    labels = feature_records_df["label"].to_numpy()
    display_labels = feature_records_df["display_label"].to_numpy()
    paths = feature_records_df["path"].to_numpy()
    class_names = sorted(feature_records_df["label"].unique())
    y = labels

    feature_cache_path = _save_feature_cache(
        X=X,
        labels=y,
        display_labels=display_labels,
        paths=paths,
        output_dir=processed_root,
    )

    if not skipped_df.empty:
        skipped_df.to_csv(processed_root / "skipped_files.csv", index=False)

    with (processed_root / "label_mapping.json").open("w", encoding="utf-8") as file_obj:
        json.dump(label_mapping, file_obj, indent=2)
        file_obj.write("\n")

    indices = np.arange(len(y))
    try:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError as exc:
        raise ValueError(
            "Stratified train/test split failed. Ensure each class has enough readable images."
        ) from exc

    _save_split_manifest(
        labels=y,
        paths=paths,
        train_indices=train_indices,
        test_indices=test_indices,
        output_dir=processed_root,
    )

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    metrics_rows = []
    model_results: dict[str, object] = {}
    
    print("Training Stage-1 One-Class SVM skin detector...")
    trained_ocsvm = train_ocsvm(
        X_train=X_train,
        random_state=random_state,
        nu=0.05
    )
    joblib.dump(trained_ocsvm, output_root / "trained_ocsvm.joblib")

    for model_name in ("svm", "random_forest"):
        print(f"Training Stage-2 classifier: {model_name}...")
        trained_model = train_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )
        joblib.dump(trained_model, output_root / f"trained_pipeline_{model_name}.joblib")

        evaluation = evaluate_model(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            output_dir=output_root,
            model_name=model_name,
            average="weighted",
            label_mapping=label_mapping,
        )
        metrics_rows.append(evaluation["metrics"])
        model_results[model_name] = evaluation

    metrics_summary_df = pd.DataFrame(metrics_rows).sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    )
    metrics_summary_path = output_root / "metrics_summary.csv"
    metrics_summary_df.to_csv(metrics_summary_path, index=False)

    return {
        "metrics_summary": metrics_summary_df,
        "model_results": model_results,
        "feature_cache_path": str(feature_cache_path),
        "skipped_files": skipped_df,
        "label_mapping": label_mapping,
    }
