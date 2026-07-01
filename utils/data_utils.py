"""Dataset loading, multi-file merging, and profiling utilities."""
import pandas as pd
import numpy as np


def load_dataset(file) -> pd.DataFrame:
    """Load a single uploaded file (CSV or Excel) into a DataFrame."""
    name = getattr(file, "name", str(file))
    lower = name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(file)
    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    raise ValueError(f"Unsupported file type: {name}. Use CSV or Excel.")


def merge_datasets(dfs: dict) -> pd.DataFrame:
    """Best-effort auto-join of multiple uploaded dataframes on shared column names.
    Falls back to a plain concat-free left join chain; if no shared columns exist
    for a given pair, they're joined on index instead."""
    names = list(dfs.keys())
    if len(names) == 1:
        return dfs[names[0]]

    merged = dfs[names[0]]
    for name in names[1:]:
        other = dfs[name]
        shared_cols = [c for c in merged.columns if c in other.columns]
        if shared_cols:
            key = shared_cols[0]
            merged = merged.merge(other, on=key, how="left", suffixes=("", f"__{name}"))
        else:
            merged = merged.join(other, how="left", rsuffix=f"__{name}")
    return merged


def profile_dataset(df: pd.DataFrame, sample_rows: int = 5) -> dict:
    """Produce a JSON-friendly profile of the dataframe used to prompt the LLM
    and to drive the cleaning/EDA/visualization agents."""
    profile = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": [],
        "sample": df.head(sample_rows).to_dict(orient="records"),
        "n_duplicates": int(df.duplicated().sum()),
    }
    for col in df.columns:
        s = df[col]
        col_info = {
            "name": str(col),
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(float(s.isna().mean()) * 100, 2),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            col_info.update({
                "mean": _safe_round(s.mean()),
                "std": _safe_round(s.std()),
                "min": _safe_round(s.min()),
                "max": _safe_round(s.max()),
            })
        profile["columns"].append(col_info)
    return profile


def _safe_round(x, n=3):
    try:
        if pd.isna(x):
            return None
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def infer_data_quality_score(df: pd.DataFrame) -> float:
    """0-100 heuristic quality score: penalizes missing values and duplicate rows."""
    if len(df) == 0:
        return 0.0
    missing_penalty = float(df.isna().mean().mean()) * 50
    dup_penalty = (float(df.duplicated().sum()) / len(df)) * 30
    return round(max(0.0, 100 - missing_penalty - dup_penalty), 1)


def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to auto-convert object columns that look like dates into datetime64."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().all():
            continue
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format=None)
            if parsed.notna().mean() > 0.8:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except (ValueError, TypeError):
            continue
    return df
