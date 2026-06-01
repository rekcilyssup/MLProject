# Troubleshooting

## TotalCharges conversion error
Ensure the raw CSV matches the Kaggle dataset and has TotalCharges column.
The script replaces blank strings with NaN and converts to numeric.

## Missing Churn column
Verify the correct file is loaded: WA_Fn-UseC_-Telco-Customer-Churn.csv

## Low F1 score
Try tuning the decision threshold or use a tree-based model.
