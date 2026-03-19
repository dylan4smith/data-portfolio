# Invoice Payment Delay Predictor

A supervised machine learning model that predicts whether consulting invoices will be paid late, enabling proactive collections follow-up and improved cash flow forecasting.

## Business Problem

Late invoice payments are one of the most common cash flow challenges for professional services firms. When 30-40% of invoices are paid past their terms, it creates budgeting uncertainty and forces reactive collections efforts. This project builds a predictive model that scores each invoice at issuance, letting the finance team prioritize follow-up on high-risk invoices before they become overdue.

## Approach

1. **Synthetic data generation** — Created a realistic dataset of 2,000 consulting invoices with features informed by real-world payment behavior patterns (client size, industry, payment terms, historical late rate, contact responsiveness, etc.).

2. **Model selection** — Evaluated three candidate classifiers via 5-fold stratified cross-validation using ROC-AUC:
   - Logistic Regression (balanced class weights)
   - Random Forest (200 trees, balanced weights)
   - Gradient Boosting (200 estimators)

3. **Evaluation** — The best model is assessed on a held-out 20% test set using ROC-AUC, average precision, and a full classification report. Feature importance and precision-recall curves are generated automatically.

4. **Inference pipeline** — A separate `predict.py` script loads the trained model and scores new invoices, assigning each a probability, risk flag, and risk tier (low / medium / high / critical).

## Tech Stack

- Python 3.10+
- scikit-learn (modeling, preprocessing, evaluation)
- pandas (data handling)
- matplotlib / seaborn (visualization)
- joblib (model serialization)

## Project Structure

```
invoice-payment-delay-predictor/
├── data/
│   └── invoices.csv              # Synthetic invoice dataset
├── output/
│   ├── model.joblib              # Trained model pipeline
│   ├── metrics.json              # Evaluation metrics
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── feature_importance.png    # Top feature importances
│   └── precision_recall_curve.png
├── generate_data.py              # Synthetic data generator
├── train_model.py                # Model training & evaluation
├── predict.py                    # Inference on new invoices
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Regenerate synthetic data
python generate_data.py

# Train and evaluate the model
python train_model.py

# Score new invoices
python predict.py --input data/invoices.csv --model output/model.joblib
```

The training script outputs evaluation metrics to the console and saves plots and the serialized model to `output/`.

## Sample Output

After training, `metrics.json` contains cross-validation and test set results:

```json
{
  "best_model": "random_forest",
  "test_roc_auc": 0.64,
  "test_avg_precision": 0.54
}
```

The `predict.py` script produces a scored CSV with columns:
- `late_payment_probability` — model's estimated probability of late payment
- `risk_flag` — binary flag (1 = high risk at default 0.5 threshold)
- `risk_tier` — low / medium / high / critical

## Key Findings

- **Prior late payment rate** and **contact responsiveness** are the strongest predictors of future late payments.
- **Enterprise clients with purchase orders** are significantly less likely to pay late.
- **Q4 invoices** show elevated late payment risk, likely due to year-end budget constraints.
- A risk-tiered follow-up workflow could reduce overdue invoices by focusing effort on the ~25% of invoices in the "high" and "critical" tiers.
