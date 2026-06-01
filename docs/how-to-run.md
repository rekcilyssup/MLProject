# How to Run

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## EDA
```bash
python src/eda.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir artifacts
```

## Train
```bash
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir artifacts
```

Run with custom CV folds:
```bash
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir artifacts --cv-folds 5
```

## Predict
```bash
python src/predict.py \
  --model artifacts/model.joblib \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --out artifacts/predictions.csv
```

Use best threshold from metrics:
```bash
python src/predict.py \
  --model artifacts/model.joblib \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --out artifacts/predictions.csv \
  --use-best-threshold \
  --metrics artifacts/metrics.json
```

## What to Look For
- metrics.json shows overall performance
- feature_importance.csv shows top drivers of churn
- predictions.csv provides churn probabilities per customer
