# ML Project Overview: Telco Customer Churn

## Problem Statement
Predict whether a customer will churn (Yes/No) based on account and service features.

## Dataset
- Source: Kaggle (blastchar/telco-customer-churn)
- File: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

## Target
- Column: Churn
- Mapping: Yes -> 1, No -> 0

## Core Features
Typical features include tenure, contract type, monthly charges, payment method,
and service subscriptions (InternetService, OnlineSecurity, etc.).

## Pipeline Summary
1) Load CSV and clean TotalCharges
2) Split into train/test with stratification
3) Impute missing values
4) One-hot encode categorical features
5) Standardize numeric features
6) Train Logistic Regression with class_weight=balanced
7) Evaluate accuracy, ROC-AUC, average precision, F1, precision, recall
8) Save model and artifacts

## Artifacts
- artifacts/model.joblib
- artifacts/metrics.json
- artifacts/feature_importance.csv
- artifacts/training_info.json

## Why Logistic Regression?
- Strong baseline for tabular classification
- Interpretable coefficients
- Fast to train and easy to deploy

## Next Steps
- Add cross-validation and hyperparameter tuning
- Try tree-based models (XGBoost/LightGBM)
- Calibrate probabilities and optimize threshold for business cost
