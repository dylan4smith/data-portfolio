"""
Invoice Payment Delay Predictor — Model Training & Evaluation

Trains a classification model to predict whether a consulting invoice
will be paid late, enabling proactive follow-up and cash flow planning.

Usage:
    python train_model.py [--data DATA_PATH] [--output MODEL_DIR]
"""

import argparse
import json
import logging
import os
import sys
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "invoice_amount",
    "payment_terms_days",
    "project_duration_weeks",
    "invoices_ytd",
    "prior_late_payments",
    "prior_late_rate",
    "has_purchase_order",
    "contact_responsiveness",
    "issue_month",
    "issue_quarter",
]

CATEGORICAL_FEATURES = [
    "client_size",
    "client_industry",
    "service_type",
    "issue_day_of_week",
]

TARGET = "is_late"


def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate the invoice dataset."""
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Loaded %d records (%d late, %d on-time)",
                len(df), df[TARGET].sum(), len(df) - df[TARGET].sum())
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create a column transformer for numeric + categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_pipelines(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """Return a dict of named candidate pipelines to evaluate."""
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            )),
        ]),
    }


def evaluate_candidates(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[str, Pipeline, Dict[str, float]]:
    """Run cross-validation on each candidate and return the best one."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: Dict[str, float] = {}

    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mean_auc = scores.mean()
        results[name] = round(mean_auc, 4)
        logger.info("  %-25s CV ROC-AUC: %.4f (+/- %.4f)", name, mean_auc, scores.std())

    best_name = max(results, key=results.get)  # type: ignore[arg-type]
    logger.info("Best model: %s (AUC=%.4f)", best_name, results[best_name])
    return best_name, pipelines[best_name], results


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["On-Time", "Late"], yticklabels=["On-Time", "Late"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Invoice Payment Delay")
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)


def plot_feature_importance(pipeline: Pipeline, output_dir: str) -> None:
    """Save a feature importance bar chart (works for tree-based models)."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        logger.info("Skipping feature importance (model type has no feature_importances_)")
        return

    # Reconstruct feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # top 15

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [all_feature_names[i] for i in reversed(indices)],
        [importances[i] for i in reversed(indices)],
        color="#3b82f6",
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Features — Invoice Payment Delay Predictor")
    fig.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance chart to %s", path)


def plot_precision_recall(y_true: np.ndarray, y_proba: np.ndarray, output_dir: str) -> None:
    """Save a precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#3b82f6", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Late Payment Detection")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "precision_recall_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved precision-recall curve to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train invoice payment delay predictor")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "invoices.csv"),
                        help="Path to invoice CSV")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "output"),
                        help="Directory for model artifacts and plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & split
    # ------------------------------------------------------------------
    df = load_data(args.data)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # 2. Model selection via cross-validation
    # ------------------------------------------------------------------
    logger.info("Evaluating candidate models...")
    preprocessor = build_preprocessor()
    pipelines = build_candidate_pipelines(preprocessor)
    best_name, best_pipeline, cv_results = evaluate_candidates(pipelines, X_train, y_train)

    # ------------------------------------------------------------------
    # 3. Final training on full train set
    # ------------------------------------------------------------------
    logger.info("Training final model (%s) on full training set...", best_name)
    best_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluation on held-out test set
    # ------------------------------------------------------------------
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    test_ap = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["On-Time", "Late"])

    logger.info("\n--- Test Set Results ---")
    logger.info("ROC-AUC:  %.4f", test_auc)
    logger.info("Avg Precision: %.4f", test_ap)
    logger.info("\n%s", report)

    # ------------------------------------------------------------------
    # 5. Save artifacts
    # ------------------------------------------------------------------
    # Model
    model_path = os.path.join(args.output, "model.joblib")
    joblib.dump(best_pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    # Metrics JSON
    metrics = {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_roc_auc": round(test_auc, 4),
        "test_avg_precision": round(test_ap, 4),
        "test_classification_report": classification_report(
            y_test, y_pred, target_names=["On-Time", "Late"], output_dict=True
        ),
    }
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    # Plots
    plot_confusion_matrix(y_test.values, y_pred, args.output)
    plot_feature_importance(best_pipeline, args.output)
    plot_precision_recall(y_test.values, y_proba, args.output)

    logger.info("Done. All artifacts saved to %s/", args.output)


if __name__ == "__main__":
    main()
