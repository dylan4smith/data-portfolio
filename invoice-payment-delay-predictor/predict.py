"""
Invoice Payment Delay Predictor — Inference Script

Loads a trained model and scores new invoices, outputting a risk-ranked
CSV that the collections team can use for proactive follow-up.

Usage:
    python predict.py --input data/invoices.csv --model output/model.joblib
"""

import argparse
import logging
import os
import sys

import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
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
    "client_size",
    "client_industry",
    "service_type",
    "issue_day_of_week",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score invoices for late-payment risk")
    parser.add_argument("--input", required=True, help="Path to invoice CSV to score")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "output", "model.joblib"),
                        help="Path to trained model")
    parser.add_argument("--output", default=None, help="Output CSV path (default: input dir / scored_invoices.csv)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for flagging as high-risk (default: 0.5)")
    args = parser.parse_args()

    # Load model
    logger.info("Loading model from %s", args.model)
    pipeline = joblib.load(args.model)

    # Load data
    df = pd.read_csv(args.input)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        logger.error("Missing columns in input: %s", missing)
        sys.exit(1)

    X = df[FEATURE_COLUMNS]
    logger.info("Scoring %d invoices...", len(X))

    # Predict
    probabilities = pipeline.predict_proba(X)[:, 1]
    df["late_payment_probability"] = probabilities.round(4)
    df["risk_flag"] = (probabilities >= args.threshold).astype(int)
    df["risk_tier"] = pd.cut(
        probabilities,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["low", "medium", "high", "critical"],
    )

    # Sort by risk descending
    df = df.sort_values("late_payment_probability", ascending=False)

    # Output
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "scored_invoices.csv")
    df.to_csv(args.output, index=False)

    flagged = df["risk_flag"].sum()
    logger.info("Flagged %d / %d invoices as high-risk (%.1f%%)",
                flagged, len(df), flagged / len(df) * 100)
    logger.info("Scored output saved to %s", args.output)


if __name__ == "__main__":
    main()
