"""
Classify new support tickets using the trained model.

Usage:
    # Classify a single ticket from the command line
    python predict.py --subject "I was charged twice" --body "My card was billed $49.99 two times for order 12345."

    # Classify tickets from a CSV file (must have 'subject' and 'body' columns)
    python predict.py --file new_tickets.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "ticket_classifier.joblib"


def load_model(path: Path = MODEL_PATH):
    """Load the persisted scikit-learn pipeline."""
    if not path.exists():
        print(f"Error: model not found at {path}. Run train.py first.", file=sys.stderr)
        sys.exit(1)
    return joblib.load(path)


def predict_single(model, subject: str, body: str) -> dict[str, object]:
    """Return predicted department and class probabilities for one ticket."""
    text = f"{subject} {body}"
    prediction = model.predict([text])[0]
    probas = model.predict_proba([text])[0]
    classes = model.classes_.tolist()
    confidence = dict(zip(classes, [round(p, 4) for p in probas]))
    return {"predicted_department": prediction, "confidence": confidence}


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """Add prediction columns to a DataFrame of tickets."""
    df = df.copy()
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["predicted_department"] = model.predict(df["text"])

    probas = model.predict_proba(df["text"])
    for i, cls in enumerate(model.classes_):
        df[f"prob_{cls}"] = probas[:, i].round(4)

    df.drop(columns=["text"], inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict support ticket department")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", type=str, help="Ticket subject line (use with --body)")
    group.add_argument("--file", type=Path, help="CSV file with 'subject' and 'body' columns")

    parser.add_argument("--body", type=str, default="", help="Ticket body text")
    parser.add_argument("--output", type=Path, help="Output CSV path for batch predictions")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to trained model (default: models/ticket_classifier.joblib)",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.subject is not None:
        result = predict_single(model, args.subject, args.body)
        print(f"\nPredicted Department: {result['predicted_department']}")
        print("Confidence scores:")
        for dept, score in sorted(result["confidence"].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 40)
            print(f"  {dept:<12s}  {score:.4f}  {bar}")
    else:
        df = pd.read_csv(args.file)
        result_df = predict_batch(model, df)
        out = args.output or args.file.with_stem(args.file.stem + "_predictions")
        result_df.to_csv(out, index=False)
        print(f"Predictions written to {out}  ({len(result_df)} tickets)")


if __name__ == "__main__":
    main()
