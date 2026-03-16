# Lead Scoring Model — B2B Conversion Prediction

## Business Problem

Sales teams waste significant time pursuing low-quality leads. At a typical B2B company, only ~25-30% of inbound leads convert, meaning reps spend the majority of their effort on prospects that will never close. A data-driven lead scoring system lets the team rank incoming leads by conversion probability, focus outreach on high-value opportunities, and improve pipeline efficiency.

## Approach

This project implements a **full ML pipeline from scratch** using only NumPy — no scikit-learn or other high-level ML libraries. This demonstrates a ground-up understanding of the algorithms rather than just API familiarity.

**Models implemented:**

- **Logistic Regression** — Binary cross-entropy loss minimized via mini-batch gradient descent with L2 regularization. Includes learning rate scheduling and convergence monitoring.
- **K-Nearest Neighbors (k=7)** — Euclidean-distance-based voting classifier with efficient vectorized distance computation.

**Pipeline components (all from scratch):**

- One-hot encoding for categorical features
- Stratified train/test split preserving class balance
- Standard scaling (fit on train, transform both)
- Stratified 5-fold cross-validation for model stability
- ROC AUC, precision, recall, F1, and confusion matrix evaluation
- Automated visualization (ROC curves, feature importance, training loss)

## Key Results

| Model               | AUC   | F1    | Precision | Recall | CV AUC (5-fold)   |
|---------------------|-------|-------|-----------|--------|--------------------|
| Logistic Regression | 0.714 | 0.446 | 0.696     | 0.328  | 0.670 ± 0.044     |
| KNN (k=7)           | 0.585 | 0.217 | 0.447     | 0.143  | 0.515 ± 0.015     |

The logistic regression model achieves a **0.714 AUC**, meaning it correctly ranks a random positive lead above a random negative lead ~71% of the time. Top conversion drivers include `demo_requested`, `webinar_attended`, `content_downloads`, and `email_clicks`.

## Tech Stack

- **Python 3.10+** — Core language
- **NumPy** — Linear algebra, gradient computation, vectorized distance calculations
- **pandas** — Data loading and feature engineering
- **matplotlib / seaborn** — Evaluation plots and visualizations

## Project Structure

```
lead-scoring-model/
├── data/
│   └── leads.csv              # Synthetic CRM export (2,000 leads)
├── output/
│   ├── roc_curves.png         # ROC comparison plot
│   ├── confusion_matrix.png   # Best model confusion matrix
│   ├── training_loss.png      # Logistic regression convergence
│   ├── feature_importance.png # Top coefficient magnitudes
│   ├── classification_report.txt
│   └── metrics.json           # Machine-readable results
├── generate_data.py           # Synthetic data generator
├── model.py                   # Full ML pipeline
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic lead data
python generate_data.py

# 3. Train models and generate evaluation outputs
python model.py
```

All outputs (plots, metrics, reports) are saved to the `output/` directory.

## Sample Output

Running `model.py` produces a comparison table:

```
======================================================================
MODEL COMPARISON SUMMARY
======================================================================
Model                       AUC     F1   Prec    Rec       CV AUC
----------------------------------------------------------------------
Logistic Regression       0.714  0.446  0.696  0.328 0.670±0.044  <-- best
KNN (k=7)                 0.585  0.217  0.447  0.143 0.515±0.015
======================================================================
```

## Dataset

The synthetic dataset simulates a realistic CRM export with 2,000 B2B leads. Features include firmographic data (company size, industry), engagement signals (website visits, email clicks, content downloads, demo requests), lead source channel, and time-based recency indicators. Conversion probability is modeled via a logistic function with realistic feature correlations — no real client data is used.
