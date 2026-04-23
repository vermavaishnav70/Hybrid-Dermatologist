from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_phase1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 classical ML pipeline for facial skin-condition classification."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the image dataset root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1_baseline"),
        help="Directory where metrics, plots, and trained models will be saved.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square resize target. Images are resized to image-size x image-size.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test splits and model training.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=32,
        help="Number of bins per HSV channel in the color histogram.",
    )
    parser.add_argument(
        "--lbp-points",
        type=int,
        default=24,
        help="Number of neighboring points used for LBP.",
    )
    parser.add_argument(
        "--lbp-radius",
        type=int,
        default=3,
        help="Radius used for LBP texture extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_phase1_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=(args.image_size, args.image_size),
        test_size=args.test_size,
        random_state=args.random_state,
        hist_bins=args.hist_bins,
        lbp_points=args.lbp_points,
        lbp_radius=args.lbp_radius,
    )

    metrics_summary = results["metrics_summary"]
    print("\nPhase 1 classical ML pipeline completed successfully.\n")
    print(metrics_summary.to_string(index=False))

    skipped_files = results["skipped_files"]
    if skipped_files is not None and not skipped_files.empty:
        print(f"\nSkipped {len(skipped_files)} unreadable or empty entries. See skipped_files.csv.")


if __name__ == "__main__":
    main()
