# Support Ticket Classifier

**Multi-class NLP model that automatically routes customer support tickets to the correct department, reducing manual triage time and improving first-response SLA.**

## Business Problem

Customer support teams at mid-size SaaS companies typically receive hundreds of tickets daily across Billing, Technical Support, Account Management, Shipping, and Product Feedback queues. Manual triage is slow, error-prone, and delays resolution — especially when agents mis-route tickets to the wrong department.

This project builds an ML-powered classifier that reads the ticket subject and body, then predicts the correct department with **97% accuracy**, enabling automated routing that can cut average first-response time by 40–60%.

## Approach

1. **Synthetic Data Generation** — 2,000 realistic support tickets across 5 departments, including intentionally ambiguous cross-department cases and ~3% label noise to simulate real-world data quality.
2. **Text Vectorization** — TF-IDF with unigrams, sublinear term-frequency scaling, and Unicode normalization.
3. **Model Selection** — Logistic Regression with balanced class weights, tuned via 5-fold stratified cross-validation over a grid of regularization strengths and n-gram ranges.
4. **Evaluation** — Held-out test set (20%) with precision/recall/F1 per class, confusion matrix, and feature importance analysis.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 0.97 |
| Weighted F1 | 0.97 |
| Macro F1 | 0.97 |

Per-department performance is consistent, with all classes achieving ≥0.96 F1. The model handles ambiguous tickets (e.g., "charged for a returned item" blending Billing and Shipping) with reasonable confidence distributions.

## Tech Stack

- **Python 3.10+**
- **pandas** — data loading and manipulation
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, GridSearchCV, evaluation metrics
- **matplotlib** — confusion matrix, metrics charts, feature importance plots
- **joblib** — model serialization

## Project Structure

```
support-ticket-classifier/
├── data/
│   └── support_tickets.csv      # Synthetic dataset (2,000 tickets)
├── models/
│   ├── ticket_classifier.joblib # Trained model (git-ignored)
│   └── metrics.json             # Test-set evaluation metrics
├── reports/
│   ├── confusion_matrix.png     # Normalized confusion matrix
│   ├── classification_metrics.png
│   ├── top_features.png         # Top TF-IDF features per department
│   └── classification_report.txt
├── generate_data.py             # Synthetic data generator
├── train.py                     # Training pipeline with hyperparameter tuning
├── predict.py                   # Inference CLI (single ticket or batch CSV)
├── evaluate.py                  # Evaluation charts and reports
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Generate synthetic data
python generate_data.py

# 2. Train the model (runs grid search, saves to models/)
python train.py

# 3. Generate evaluation reports (saves charts to reports/)
python evaluate.py

# 4. Classify a single ticket
python predict.py --subject "I was charged twice" \
                  --body "My card shows two charges of \$49.99 for order 12345."

# 5. Batch-classify tickets from a CSV
python predict.py --file new_tickets.csv --output predictions.csv
```

## Sample Output

```
Predicted Department: Billing
Confidence scores:
  Billing       0.5590  ██████████████████████
  Shipping      0.1719  ██████
  Account       0.1008  ████
  Product       0.0945  ███
  Technical     0.0738  ██
```

## Key Design Decisions

- **Label noise (3%)** simulates real-world mis-labeling in historical ticket data, preventing unrealistically perfect metrics.
- **Ambiguous templates (15%)** test the model on tickets that genuinely span multiple departments, reflecting how real customers write.
- **Balanced class weights** in Logistic Regression prevent the model from favoring majority classes.
- **TF-IDF over embeddings** was chosen for interpretability — stakeholders can inspect which words drive each classification via the feature importance charts.

## Extending This Project

- Swap in a real ticket dataset (e.g., from Zendesk or Freshdesk exports) with minimal changes to `train.py`.
- Add priority prediction as a secondary classification head.
- Deploy as a REST API with FastAPI for real-time ticket routing.
- Integrate with a helpdesk webhook to auto-assign tickets on creation.
