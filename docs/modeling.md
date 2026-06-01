# Modeling and Evaluation Details

## Data Cleaning
- TotalCharges contains blank strings in the raw CSV.
- We convert to numeric and coerce invalid values to NaN.

## Feature Handling
- customerID is dropped from features (identifier only).
- Categorical features are one-hot encoded.
- Numeric features are median-imputed.
- Numeric features are standardized after imputation.

## Train/Test Split
- Stratified train/test split to preserve churn ratio.
- Default test size: 20%.

## Cross-Validation
- Stratified K-fold CV (ROC-AUC) runs on the training split.
- Reported mean and std help estimate generalization.

## Class Imbalance
- class_weight=balanced is enabled for Logistic Regression.
- This reduces bias toward the majority class.

## Metrics
- Accuracy: overall correctness
- ROC-AUC: ranking quality across thresholds
- Average Precision: useful for imbalanced data
- Precision and recall at the default 0.5 threshold
- F1 at 0.5: baseline decision threshold
- Best threshold by maximizing F1 from PR curve

## Explainability
- Coefficients are exported to feature_importance.csv
- Positive coefficients increase churn probability
- Negative coefficients decrease churn probability

## Limitations
- Linear model may miss nonlinear interactions
- No time-based validation (assumes IID data)
