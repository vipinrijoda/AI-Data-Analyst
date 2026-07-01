"""Tests for the deterministic (non-LLM) parts of the pipeline: EDA,
visualization, ML, SQL execution, and data utils. These run without any API
key so they're safe for CI.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from utils.data_utils import profile_dataset, infer_data_quality_score, merge_datasets
from agents.eda import run_eda
from agents.visualization import auto_visualize
from agents.ml import run_ml, infer_task
from agents.sql import run_sql
from agents.cleaning import apply_cleaning, detect_outliers_iqr


def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "revenue": rng.normal(1000, 200, 300).round(2),
        "region": rng.choice(["North", "South", "East", "West"], 300),
        "signup_date": pd.date_range("2024-01-01", periods=300, freq="D"),
        "churned": rng.choice([0, 1], 300, p=[0.8, 0.2]),
    })


def test_profile_dataset():
    df = sample_df()
    profile = profile_dataset(df)
    assert profile["n_rows"] == 300
    assert profile["n_cols"] == 4
    assert len(profile["columns"]) == 4


def test_data_quality_score_range():
    df = sample_df()
    score = infer_data_quality_score(df)
    assert 0 <= score <= 100


def test_run_eda():
    df = sample_df()
    eda = run_eda(df)
    assert "revenue" in eda["outliers"]
    assert eda["duplicate_rows"] == 0


def test_auto_visualize_returns_charts():
    df = sample_df()
    charts = auto_visualize(df)
    assert len(charts) > 0
    assert all("fig" in c for c in charts)


def test_ml_classification():
    df = sample_df().drop(columns=["signup_date"])
    assert infer_task(df, "churned") == "classification"
    result = run_ml(df, target="churned")
    assert result["task"] == "classification"
    assert "accuracy" in result


def test_ml_clustering():
    df = sample_df().drop(columns=["signup_date", "region"])
    result = run_ml(df, target=None)
    assert result["task"] == "clustering"
    assert "n_clusters" in result


def test_sql_execution():
    df = sample_df()
    result = run_sql(df, "SELECT region, COUNT(*) as n FROM data GROUP BY region")
    assert "n" in result.columns


def test_outlier_detection():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    mask = detect_outliers_iqr(df, "x")
    assert mask.sum() >= 1


def test_apply_cleaning_drops_column():
    df = pd.DataFrame({"a": [1, None, None, None], "b": [1, 2, 3, 4]})
    suggestions = {"drop_duplicates": False,
                   "columns": [{"column": "a", "issue": "75% missing", "strategy": "drop_column"}]}
    cleaned = apply_cleaning(df, suggestions)
    assert "a" not in cleaned.columns


def test_merge_single_dataset():
    df = sample_df()
    merged = merge_datasets({"only.csv": df})
    assert merged.equals(df)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
