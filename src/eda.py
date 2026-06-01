import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    return df


def summarize(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_by_column": df.isna().sum().to_dict(),
        "churn_rate": None,
    }

    if "Churn" in df.columns:
        churn_rate = (df["Churn"] == "Yes").mean()
        summary["churn_rate"] = float(churn_rate)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    summary["numeric_columns"] = numeric_cols
    summary["numeric_summary"] = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    summary["categorical_columns"] = categorical_cols
    summary["categorical_cardinality"] = {
        col: int(df[col].nunique(dropna=True)) for col in categorical_cols
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--outdir", default="artifacts", help="Output directory")
    args = parser.parse_args()

    df = load_data(args.data)
    report = summarize(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "eda_summary.json"
    with outpath.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
