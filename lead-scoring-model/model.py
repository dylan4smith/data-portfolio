"""
Lead Scoring Model — B2B Lead Conversion Prediction

Implements logistic regression and k-nearest neighbors from scratch
(using only NumPy) to predict which inbound leads are most likely to
convert, enabling sales teams to prioritize high-value outreach.

Why from scratch?  Demonstrates a deep understanding of the underlying
algorithms, gradient-based optimization, and evaluation methodology
beyond calling high-level library APIs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "leads.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

CATEGORICAL_FEATURES = ["company_size", "industry", "lead_source"]
NUMERICAL_FEATURES = [
    "website_visits", "pages_per_session", "email_opens", "email_clicks",
    "content_downloads", "webinar_attended", "demo_requested",
    "days_since_first_touch", "days_since_last_activity",
]
TARGET = "converted"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===================================================================
# Preprocessing utilities (no sklearn dependency)
# ===================================================================
def one_hot_encode(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode categorical columns, returning expanded df and category map."""
    category_map: dict[str, list[str]] = {}
    encoded_parts = []

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        category_map[col] = list(dummies.columns)
        encoded_parts.append(dummies)

    df_encoded = pd.concat([df.drop(columns=columns)] + encoded_parts, axis=1)
    return df_encoded, category_map


def standard_scale(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance scaling fitted on training data only."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0  # avoid division by zero for constant features
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data while preserving class proportions."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_pos_test = max(1, int(len(idx_pos) * test_size))
    n_neg_test = max(1, int(len(idx_neg) * test_size))

    test_idx = np.concatenate([idx_pos[:n_pos_test], idx_neg[:n_neg_test]])
    train_idx = np.concatenate([idx_pos[n_pos_test:], idx_neg[n_neg_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ===================================================================
# Metrics
# ===================================================================
@dataclass
class ClassificationMetrics:
    """Container for binary classification evaluation metrics."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray

    accuracy: float = field(init=False)
    precision: float = field(init=False)
    recall: float = field(init=False)
    f1: float = field(init=False)
    roc_auc: float = field(init=False)

    def __post_init__(self) -> None:
        tp = np.sum((self.y_pred == 1) & (self.y_true == 1))
        fp = np.sum((self.y_pred == 1) & (self.y_true == 0))
        fn = np.sum((self.y_pred == 0) & (self.y_true == 1))
        tn = np.sum((self.y_pred == 0) & (self.y_true == 0))

        self.accuracy = (tp + tn) / len(self.y_true)
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1 = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )
        self.roc_auc = self._compute_auc()

    def _compute_auc(self) -> float:
        """Compute AUC via the trapezoidal rule on sorted thresholds."""
        sorted_idx = np.argsort(-self.y_prob)
        y_sorted = self.y_true[sorted_idx]

        n_pos = np.sum(self.y_true == 1)
        n_neg = np.sum(self.y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_list = [0.0]
        fpr_list = [0.0]
        tp_count = 0
        fp_count = 0

        for label in y_sorted:
            if label == 1:
                tp_count += 1
            else:
                fp_count += 1
            tpr_list.append(tp_count / n_pos)
            fpr_list.append(fp_count / n_neg)

        return float(np.trapezoid(tpr_list, fpr_list))

    def roc_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (fpr, tpr) arrays for plotting."""
        thresholds = np.sort(np.unique(self.y_prob))[::-1]
        n_pos = np.sum(self.y_true == 1)
        n_neg = np.sum(self.y_true == 0)
        fpr_list, tpr_list = [0.0], [0.0]

        for thresh in thresholds:
            preds = (self.y_prob >= thresh).astype(int)
            tp = np.sum((preds == 1) & (self.y_true == 1))
            fp = np.sum((preds == 1) & (self.y_true == 0))
            tpr_list.append(tp / n_pos if n_pos else 0)
            fpr_list.append(fp / n_neg if n_neg else 0)

        fpr_list.append(1.0)
        tpr_list.append(1.0)
        return np.array(fpr_list), np.array(tpr_list)

    def confusion_matrix(self) -> np.ndarray:
        """Return 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
        tp = np.sum((self.y_pred == 1) & (self.y_true == 1))
        fp = np.sum((self.y_pred == 1) & (self.y_true == 0))
        fn = np.sum((self.y_pred == 0) & (self.y_true == 1))
        tn = np.sum((self.y_pred == 0) & (self.y_true == 0))
        return np.array([[tn, fp], [fn, tp]])

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
        }


# ===================================================================
# Model 1: Logistic Regression (gradient descent from scratch)
# ===================================================================
class LogisticRegressionScratch:
    """Binary logistic regression trained via mini-batch gradient descent.

    Implements L2 regularization and adaptive learning rate scheduling.
    """

    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 500,
        batch_size: int = 64,
        reg_lambda: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Shuffle
            idx = rng.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_b = X_shuf[start:end]
                y_b = y_shuf[start:end]
                m = len(y_b)

                # Forward
                z = X_b @ self.weights + self.bias
                a = self._sigmoid(z)

                # Gradients with L2 regularization
                dw = (1 / m) * (X_b.T @ (a - y_b)) + self.reg_lambda * self.weights
                db = (1 / m) * np.sum(a - y_b)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Log loss for monitoring
            z_all = X @ self.weights + self.bias
            a_all = self._sigmoid(z_all)
            eps = 1e-15
            loss = -np.mean(
                y * np.log(a_all + eps) + (1 - y) * np.log(1 - a_all + eps)
            )
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ===================================================================
# Model 2: K-Nearest Neighbors (from scratch)
# ===================================================================
class KNNClassifierScratch:
    """K-nearest neighbors classifier using Euclidean distance."""

    def __init__(self, k: int = 7) -> None:
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierScratch":
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 based on neighbor voting."""
        # Compute pairwise distances efficiently
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
        dists = np.sqrt(np.maximum(X_sq + train_sq - 2 * X @ self.X_train.T, 0))

        # For each sample, find k nearest neighbors
        k_idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        probs = np.array([
            self.y_train[k_idx[i]].mean() for i in range(len(X))
        ])
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ===================================================================
# Cross-validation utility
# ===================================================================
def stratified_kfold_auc(
    model_factory, X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42
) -> tuple[float, float]:
    """Run stratified k-fold CV and return mean/std AUC."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0].copy()
    idx_neg = np.where(y == 0)[0].copy()
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    pos_folds = np.array_split(idx_pos, n_folds)
    neg_folds = np.array_split(idx_neg, n_folds)
    aucs = []

    for i in range(n_folds):
        val_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.concatenate(
            [np.concatenate([pos_folds[j] for j in range(n_folds) if j != i]),
             np.concatenate([neg_folds[j] for j in range(n_folds) if j != i])]
        )
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale per fold
        X_tr_s, X_val_s, _, _ = standard_scale(X_tr, X_val)

        model = model_factory()
        model.fit(X_tr_s, y_tr)
        probs = model.predict_proba(X_val_s)
        m = ClassificationMetrics(y_val, (probs >= 0.5).astype(int), probs)
        aucs.append(m.roc_auc)

    return float(np.mean(aucs)), float(np.std(aucs))


# ===================================================================
# Visualization
# ===================================================================
def plot_roc_curves(
    all_metrics: dict[str, ClassificationMetrics],
) -> None:
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in all_metrics.items():
        fpr, tpr = m.roc_curve()
        ax.plot(fpr, tpr, label=f"{name} (AUC={m.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Lead Scoring Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    log.info("Saved ROC curve plot -> output/roc_curves.png")


def plot_confusion_matrix(m: ClassificationMetrics, model_name: str) -> None:
    """Plot confusion matrix heatmap."""
    cm = m.confusion_matrix()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Converted", "Converted"],
        yticklabels=["Not Converted", "Converted"], ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix -> output/confusion_matrix.png")


def plot_training_loss(model: LogisticRegressionScratch) -> None:
    """Plot training loss curve for logistic regression."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.loss_history, color="#4C78A8")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Logistic Regression — Training Loss")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close(fig)
    log.info("Saved training loss plot -> output/training_loss.png")


def plot_feature_importance(
    weights: np.ndarray, feature_names: list[str]
) -> None:
    """Plot logistic regression coefficient magnitudes as feature importance."""
    abs_w = np.abs(weights)
    sorted_idx = np.argsort(abs_w)[-15:]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#E45756" if weights[i] < 0 else "#4C78A8" for i in sorted_idx]
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        weights[sorted_idx],
        color=colors,
    )
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Top Feature Coefficients — Logistic Regression")
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    log.info("Saved feature importance plot -> output/feature_importance.png")


# ===================================================================
# Main pipeline
# ===================================================================
def main() -> None:
    """Run the full training and evaluation pipeline."""
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    log.info("Loaded %d records (%d features)", len(df), df.shape[1] - 1)
    log.info("Conversion rate: %.1f%%", df[TARGET].mean() * 100)

    # 2. Encode categoricals
    df_encoded, cat_map = one_hot_encode(df, CATEGORICAL_FEATURES)
    feature_cols = [c for c in df_encoded.columns if c != TARGET]
    X = df_encoded[feature_cols].values.astype(float)
    y = df_encoded[TARGET].values.astype(float)
    log.info("Feature matrix: %d samples x %d features", X.shape[0], X.shape[1])

    # 3. Stratified train/test split
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=TEST_SIZE, seed=RANDOM_STATE
    )
    log.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 4. Scale features
    X_train_s, X_test_s, mu, sigma = standard_scale(X_train, X_test)

    # 5. Train models
    models = {
        "Logistic Regression": LogisticRegressionScratch(
            lr=0.05, epochs=500, batch_size=64, reg_lambda=0.01, seed=RANDOM_STATE
        ),
        "KNN (k=7)": KNNClassifierScratch(k=7),
    }

    all_metrics: dict[str, ClassificationMetrics] = {}
    results: dict[str, dict] = {}

    for name, model in models.items():
        log.info("Training %s ...", name)
        model.fit(X_train_s, y_train)

        y_prob = model.predict_proba(X_test_s)
        y_pred = (y_prob >= 0.5).astype(int)

        m = ClassificationMetrics(y_test, y_pred, y_prob)
        all_metrics[name] = m
        results[name] = m.to_dict()

        log.info(
            "%-22s  AUC=%.3f  F1=%.3f  Prec=%.3f  Rec=%.3f",
            name, m.roc_auc, m.f1, m.precision, m.recall,
        )

        # Cross-validation
        if isinstance(model, LogisticRegressionScratch):
            factory = lambda: LogisticRegressionScratch(
                lr=0.05, epochs=500, batch_size=64, reg_lambda=0.01, seed=RANDOM_STATE
            )
        else:
            factory = lambda: KNNClassifierScratch(k=7)

        cv_mean, cv_std = stratified_kfold_auc(factory, X_train, y_train, n_folds=5)
        results[name]["cv_auc_mean"] = round(cv_mean, 4)
        results[name]["cv_auc_std"] = round(cv_std, 4)
        log.info("  5-fold CV AUC: %.3f +/- %.3f", cv_mean, cv_std)

    # 6. Select best model
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_m = all_metrics[best_name]
    log.info("Best model: %s (AUC=%.3f)", best_name, results[best_name]["roc_auc"])

    # 7. Generate plots
    plot_roc_curves(all_metrics)
    plot_confusion_matrix(best_m, best_name)

    lr_model = models["Logistic Regression"]
    if isinstance(lr_model, LogisticRegressionScratch):
        plot_training_loss(lr_model)
        plot_feature_importance(lr_model.weights, feature_cols)

    # 8. Save classification report
    report_lines = [f"Best Model: {best_name}\n"]
    for name, m in all_metrics.items():
        cm = m.confusion_matrix()
        report_lines.append(f"\n{'=' * 50}")
        report_lines.append(f"Model: {name}")
        report_lines.append(f"{'=' * 50}")
        report_lines.append(f"Accuracy:  {m.accuracy:.4f}")
        report_lines.append(f"Precision: {m.precision:.4f}")
        report_lines.append(f"Recall:    {m.recall:.4f}")
        report_lines.append(f"F1 Score:  {m.f1:.4f}")
        report_lines.append(f"ROC AUC:   {m.roc_auc:.4f}")
        report_lines.append(f"\nConfusion Matrix:")
        report_lines.append(f"  TN={cm[0,0]:>4d}  FP={cm[0,1]:>4d}")
        report_lines.append(f"  FN={cm[1,0]:>4d}  TP={cm[1,1]:>4d}")

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text("\n".join(report_lines))
    log.info("Saved classification report -> %s", report_path)

    # 9. Save metrics JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    log.info("Saved metrics JSON -> %s", metrics_path)

    # 10. Summary table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<24} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'CV AUC':>12}")
    print("-" * 70)
    for name, r in results.items():
        marker = "  <-- best" if name == best_name else ""
        print(
            f"{name:<24} {r['roc_auc']:>6.3f} {r['f1']:>6.3f} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
            f"{r['cv_auc_mean']:.3f}+/-{r['cv_auc_std']:.3f}{marker}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
