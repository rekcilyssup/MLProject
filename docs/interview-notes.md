# Interview Notes

## Project Summary
This is a churn prediction pipeline using a real-world Kaggle dataset.
It demonstrates data cleaning, preprocessing, modeling, evaluation, and explainability.

## What you should highlight
- Handling of TotalCharges blanks and missing values
- One-hot encoding for categorical features
- Standardization for numeric features
- Balanced class weights for churn imbalance
- Multiple metrics (ROC-AUC, AP, F1)
- Cross-validation ROC-AUC for generalization signal
- Threshold tuning for business needs
- Model explainability via coefficients

## Improvement Ideas
- Cross-validation and hyperparameter tuning
- Feature selection and interaction features
- Probability calibration and cost-sensitive thresholds
- Model monitoring and drift detection
