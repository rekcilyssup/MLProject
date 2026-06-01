import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    return df


def split_features_target(df: pd.DataFrame):
    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column in dataset.")

    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn"])

    customer_ids = None
    if "customerID" in X.columns:
        customer_ids = X["customerID"].copy()
        X = X.drop(columns=["customerID"])

    return X, y, customer_ids


def build_pipeline(categorical_cols, numeric_cols, seed: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_feature_names(preprocessor: ColumnTransformer):
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return None


def compute_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5, float(f1_scores.max())
    best_index = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_index]), float(f1_scores[best_index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds for ROC-AUC")
    args = parser.parse_args()

    df = load_data(args.data)
    X, y, _ = split_features_target(df)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    pipeline = build_pipeline(categorical_cols, numeric_cols, args.seed)

    cv_scores = None
    if args.cv_folds and args.cv_folds > 1:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "precision_at_0.5": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_at_0.5": float(f1_score(y_test, y_pred)),
        "confusion_matrix_at_0.5": confusion_matrix(y_test, y_pred).tolist(),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
    }

    if cv_scores is not None:
        metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
        metrics["cv_roc_auc_std"] = float(cv_scores.std())

    best_threshold, best_f1 = compute_best_threshold(y_test, y_proba)
    best_pred = (y_proba >= best_threshold).astype(int)
    metrics["best_threshold"] = float(best_threshold)
    metrics["f1_at_best_threshold"] = float(best_f1)
    metrics["confusion_matrix_at_best_threshold"] = confusion_matrix(y_test, best_pred).tolist()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    training_info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "cv_folds": int(args.cv_folds),
    }

    joblib.dump(pipeline, outdir / "model.joblib")
    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (outdir / "training_info.json").open("w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)
    if feature_names is not None:
        coefs = model.coef_.ravel()
        feature_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
            }
        ).sort_values("abs_coefficient", ascending=False)
        feature_df.to_csv(outdir / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
