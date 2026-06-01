# Preprocessing Notes

- Numeric columns: median imputation
- Categorical columns: most-frequent imputation + one-hot encoding
- customerID: removed from features (identifier only)
- TotalCharges: converted to numeric, blanks to NaN
