"""Agent: EDA
Computes descriptive statistics, missing values, duplicates, correlations,
outliers, and distribution shape - no LLM calls needed, pure pandas/numpy so
results are fast and 100% grounded.
"""
import pandas as pd
import numpy as np
from agents.cleaning import detect_outliers_iqr


def run_eda(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    results = {
        "summary_stats": df.describe(include="all").fillna("").astype(str).to_dict(),
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "correlation": (
            df[numeric_cols].corr().round(3).to_dict() if len(numeric_cols) > 1 else {}
        ),
        "outliers": {},
        "distributions": {},
    }

    for col in numeric_cols:
        try:
            mask = detect_outliers_iqr(df, col)
            results["outliers"][col] = int(mask.sum())
        except Exception:
            results["outliers"][col] = 0
        skew = None
        if df[col].notna().any():
            try:
                skew = round(float(df[col].skew()), 3)
            except Exception:
                skew = None
        results["distributions"][col] = {"skew": skew}

    return results
