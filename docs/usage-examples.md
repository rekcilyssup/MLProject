# Usage Examples

## Train and evaluate
```bash
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir artifacts
```

## Run EDA
```bash
python src/eda.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir artifacts
```

## Predict
```bash
python src/predict.py --model artifacts/model.joblib --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --out artifacts/predictions.csv
```
