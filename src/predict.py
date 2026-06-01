import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--use-best-threshold", action="store_true", help="Use best_threshold from metrics.json")
    parser.add_argument("--metrics", help="Path to metrics.json with best_threshold")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = load_data(args.data)

    customer_ids = None
    if "customerID" in df.columns:
        customer_ids = df["customerID"].copy()

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    threshold = args.threshold
    if args.use_best_threshold:
        if not args.metrics:
            raise ValueError("--metrics is required when using --use-best-threshold")
        with open(args.metrics, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        threshold = float(metrics.get("best_threshold", threshold))

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= threshold).astype(int)

    out_df = pd.DataFrame({"churn_probability": proba, "churn_prediction": pred})
    if customer_ids is not None:
        out_df.insert(0, "customerID", customer_ids)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out} (threshold={threshold})")


if __name__ == "__main__":
    main()
