# ML Project: Telco Customer Churn (Classification)

This project predicts customer churn using the Kaggle Telco Customer Churn dataset.
It is designed to be interview-ready with clear preprocessing, modeling, evaluation, and
explainability steps.

## Dataset
- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Expected file: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run EDA
```bash
python src/eda.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir artifacts
```

## Train and Evaluate
```bash
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir artifacts
```

Outputs:
- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/feature_importance.csv`
- `artifacts/training_info.json`

## Predict (optional)
```bash
python src/predict.py \
  --model artifacts/model.joblib \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --out artifacts/predictions.csv
```

Use the best threshold from metrics:
```bash
python src/predict.py \
  --model artifacts/model.joblib \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --out artifacts/predictions.csv \
  --use-best-threshold \
  --metrics artifacts/metrics.json
```

## Docs
See the docs folder for deeper explanations:
- docs/overview.md
- docs/modeling.md
- docs/metrics-guide.md
- docs/interview-notes.md

## Interview Talking Points
- **Data cleaning**: `TotalCharges` has blanks; converted to numeric with NaN handling.
- **Preprocessing**: categorical features are one-hot encoded, numerics imputed with median.
- **Class imbalance**: `class_weight='balanced'` used to reduce bias toward non-churn.
- **Metrics**: accuracy, ROC-AUC, average precision, F1 at 0.5 and tuned threshold.
- **Explainability**: logistic regression coefficients exported to `feature_importance.csv`.

## Next Improvements
- Add cross-validation and hyperparameter tuning.
- Compare against tree-based models (XGBoost/LightGBM).
- Add calibration and threshold selection based on business cost.
- Add model registry and data drift checks.

# MLProject
