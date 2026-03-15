from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _display_names(class_names: list[str], label_mapping: dict[str, str]) -> list[str]:
    return [label_mapping.get(name, name.replace("_", " ")) for name in class_names]


def plot_confusion_matrix_heatmap(
    confusion_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a confusion matrix heatmap to disk."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="YlOrRd", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_classwise_performance(report_df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    """Plot precision, recall, and F1 score for each class."""
    class_rows = report_df.loc[
        ~report_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    ].copy()
    if class_rows.empty:
        return

    class_rows = class_rows.reset_index().rename(columns={"index": "class_name"})
    plot_df = class_rows.melt(
        id_vars="class_name",
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="class_name", y="score", hue="metric", palette="Set2")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_model(
    model,
    X_test,
    y_test,
    class_names: list[str],
    output_dir: str | Path,
    model_name: str,
    average: str = "weighted",
    label_mapping: dict[str, str] | None = None,
) -> dict[str, object]:
    """
    Evaluate a trained model and save report-ready artifacts.

    Weighted averages are used by default because real skin datasets often have
    class imbalance, but class-wise results are also exported for transparency.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    label_mapping = label_mapping or {}

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

    display_names = _display_names(class_names=class_names, label_mapping=label_mapping)
    confusion = confusion_matrix(y_test, y_pred, labels=class_names)
    confusion_df = pd.DataFrame(confusion, index=display_names, columns=display_names)
    confusion_csv_path = output_root / f"confusion_matrix_{model_name}.csv"
    confusion_df.to_csv(confusion_csv_path)

    confusion_png_path = output_root / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix_heatmap(
        confusion_df=confusion_df,
        output_path=confusion_png_path,
        title=f"Confusion Matrix - {model_name.replace('_', ' ').title()}",
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = output_root / f"classification_report_{model_name}.csv"
    report_df.to_csv(report_csv_path)

    classwise_png_path = output_root / f"classwise_metrics_{model_name}.png"
    plot_classwise_performance(
        report_df=report_df,
        output_path=classwise_png_path,
        title=f"Class-wise Performance - {model_name.replace('_', ' ').title()}",
    )

    metrics_summary = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_method": average,
    }

    return {
        "metrics": metrics_summary,
        "y_true": y_test,
        "y_pred": y_pred,
        "classification_report": report_df,
        "confusion_matrix": confusion_df,
    }
