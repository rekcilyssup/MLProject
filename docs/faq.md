# FAQ

## Why not use deep learning?
This project focuses on a strong, interpretable baseline. For tabular data,
linear models and tree-based models are often competitive and easier to explain.

## Why use ROC-AUC and Average Precision?
Churn is imbalanced, so accuracy can be misleading. ROC-AUC and AP capture
ranking quality and performance on the minority class.

## What is the business value?
The model can prioritize retention offers for high-risk customers, which
improves ROI and reduces churn-related revenue loss.
