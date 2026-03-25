"""
Generate detailed evaluation visuals for the support ticket classifier.

Produces:
    - Confusion matrix heatmap (PNG)
    - Per-class precision / recall / F1 bar chart (PNG)
    - Top predictive features per department (PNG)

Usage:
    python evaluate.py                         # uses default paths
    python evaluate.py --model models/ticket_classifier.joblib --data data/support_tickets.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_test_data(data_path: Path) -> tuple[pd.Series, pd.Series, list[str]]:
    """Reproduce the same train/test split used during training."""
    df = pd.read_csv(data_path)
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    _, X_test, _, y_test = train_test_split(
        df["text"],
        df["department"],
        test_size=TEST_SIZE,
        stratify=df["department"],
        random_state=RANDOM_STATE,
    )
    labels = sorted(df["department"].unique().tolist())
    return X_test, y_test, labels


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> None:
    """Save a normalized confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
    ax.set_title("Normalized Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Department", fontsize=11)
    ax.set_ylabel("Actual Department", fontsize=11)
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Confusion matrix → {path}")


def plot_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> None:
    """Save a grouped bar chart of precision, recall, and F1 per class."""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    precision = [report[l]["precision"] for l in labels]
    recall = [report[l]["recall"] for l in labels]
    f1 = [report[l]["f1-score"] for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4A90D9")
    ax.bar(x, recall, width, label="Recall", color="#50C878")
    ax.bar(x + width, f1, width, label="F1-Score", color="#FF8C42")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Per-Department Classification Metrics", fontsize=14, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "classification_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Classification metrics → {path}")


def plot_top_features(model, labels: list[str], output_dir: Path, top_n: int = 10) -> None:
    """Show the top TF-IDF features driving each class prediction."""
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())

    n_classes = len(labels)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 5), sharey=False)

    for idx, (label, ax) in enumerate(zip(labels, axes)):
        coefs = clf.coef_[idx]
        top_idx = np.argsort(coefs)[-top_n:]
        top_features = feature_names[top_idx]
        top_scores = coefs[top_idx]

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
        ax.barh(range(top_n), top_scores, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=9)

    fig.suptitle("Top Predictive Features by Department", fontsize=14, y=1.02)
    plt.tight_layout()
    path = output_dir / "top_features.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Top features → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ticket classifier")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models" / "ticket_classifier.joblib",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "support_tickets.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "reports",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading model and test data …")
    model = joblib.load(args.model)
    X_test, y_test, labels = load_test_data(args.data)
    y_pred = model.predict(X_test)

    print(f"\nGenerating evaluation reports ({len(X_test)} test samples) …")
    plot_confusion_matrix(y_test, y_pred, labels, args.output)
    plot_classification_metrics(y_test, y_pred, labels, args.output)
    plot_top_features(model, labels, args.output)

    # Save text summary
    report_text = classification_report(y_test, y_pred, target_names=labels)
    summary_path = args.output / "classification_report.txt"
    with open(summary_path, "w") as f:
        f.write("Support Ticket Classifier — Test Set Evaluation\n")
        f.write("=" * 55 + "\n\n")
        f.write(report_text)
    print(f"  ✓ Text report → {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
