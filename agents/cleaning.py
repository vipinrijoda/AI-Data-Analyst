"""Agent: Cleaning
Suggests a cleaning plan (missing value strategy, duplicate removal, column
drops) and applies it only after the user approves - never destructive by
default.
"""
import pandas as pd
from utils.llm import call_llm_json
from prompts.prompts import cleaning_prompt


def suggest_cleaning(profile: dict) -> dict:
    prompt = cleaning_prompt(profile)
    try:
        result = call_llm_json(prompt, system="You are a meticulous data cleaning expert. Respond only in JSON.")
        if "error" in result:
            return _rule_based_suggestions(profile)
        result.setdefault("columns", [])
        result.setdefault("drop_duplicates", False)
        return result
    except Exception:
        return _rule_based_suggestions(profile)


def _rule_based_suggestions(profile: dict) -> dict:
    """Deterministic fallback used if the LLM call fails, so the app never breaks."""
    suggestions = {"columns": [], "drop_duplicates": profile.get("n_duplicates", 0) > 0}
    for col in profile["columns"]:
        if col["pct_missing"] == 0:
            continue
        is_numeric = "mean" in col
        strategy = "median" if is_numeric else "mode"
        if col["pct_missing"] > 50:
            strategy = "drop_column"
        suggestions["columns"].append({
            "column": col["name"],
            "issue": f"{col['pct_missing']}% missing",
            "strategy": strategy,
        })
    return suggestions


def apply_cleaning(df: pd.DataFrame, suggestions: dict, approved_columns: list = None) -> pd.DataFrame:
    """Apply the suggested plan. If approved_columns is given, only those
    column-level fixes are applied (human-in-the-loop approval)."""
    df = df.copy()
    if suggestions.get("drop_duplicates"):
        df = df.drop_duplicates()

    for item in suggestions.get("columns", []):
        col = item.get("column")
        if col not in df.columns:
            continue
        if approved_columns is not None and col not in approved_columns:
            continue
        strategy = item.get("strategy", "mode")
        if strategy == "drop_column":
            df = df.drop(columns=[col])
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
        elif strategy == "drop":
            df = df.dropna(subset=[col])
    return df


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask of IQR-based outliers for a numeric column."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (df[col] < lower) | (df[col] > upper)
