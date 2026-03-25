"""
Train a multi-class text classification model for routing support tickets.

Pipeline:
    1. Load and preprocess ticket data (subject + body → combined text).
    2. Split into stratified train / test sets (80/20).
    3. Build a TF-IDF + Logistic Regression pipeline with hyperparameter tuning.
    4. Evaluate on the held-out test set and persist the best model.

Usage:
    python train.py                       # train with defaults
    python train.py --data data/support_tickets.csv --output models/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path: Path) -> pd.DataFrame:
    """Read ticket CSV and combine subject + body into a single text field."""
    df = pd.read_csv(path)
    required_cols = {"subject", "body", "department"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    logger.info("Loaded %d tickets with %d departments", len(df), df["department"].nunique())
    return df


def build_pipeline() -> Pipeline:
    """Construct a TF-IDF → Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    max_features=10_000,
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


PARAM_GRID: dict[str, list[Any]] = {
    "tfidf__max_features": [5_000, 10_000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}


def tune_and_train(
    X_train: pd.Series,
    y_train: pd.Series,
) -> tuple[Pipeline, dict[str, Any]]:
    """Run grid search with stratified k-fold and return the best estimator."""
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info("Best CV F1 (weighted): %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_, search.best_params_


def evaluate(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    labels: list[str],
) -> dict[str, Any]:
    """Compute test-set metrics and return them as a dict."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    logger.info("Test Accuracy : %.4f", accuracy)
    logger.info("Test F1 (wt.) : %.4f", f1_weighted)

    return {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1_weighted, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }


def save_artifacts(
    model: Pipeline,
    metrics: dict[str, Any],
    best_params: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist the trained model and evaluation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "ticket_classifier.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved → %s", model_path)

    results = {"best_params": best_params, "test_metrics": metrics}
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Metrics saved → %s", metrics_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train support ticket classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "support_tickets.csv",
        help="Path to ticket CSV (default: data/support_tickets.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models",
        help="Directory for model and metrics output (default: models/)",
    )
    args = parser.parse_args(argv)

    # 1. Load data
    df = load_data(args.data)

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["department"],
        test_size=TEST_SIZE,
        stratify=df["department"],
        random_state=RANDOM_STATE,
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # 3. Tune & train
    best_model, best_params = tune_and_train(X_train, y_train)

    # 4. Evaluate
    labels = sorted(df["department"].unique().tolist())
    metrics = evaluate(best_model, X_test, y_test, labels)

    # 5. Save artifacts
    save_artifacts(best_model, metrics, best_params, args.output)

    # 6. Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (test set)")
    print("=" * 60)
    print(classification_report(y_test, best_model.predict(X_test), target_names=labels))
    print(f"Overall Accuracy : {metrics['accuracy']:.4f}")
    print(f"Weighted F1      : {metrics['f1_weighted']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
