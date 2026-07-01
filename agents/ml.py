"""Agent: Machine Learning
Automatically infers the task (classification / regression / clustering),
trains an appropriate model, and reports metrics + feature importance.
Optionally adds SHAP explainability when the shap package is installed.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, mean_absolute_error


def infer_task(df: pd.DataFrame, target: str = None) -> str:
    if target is None:
        return "clustering"
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 15:
        return "regression"
    return "classification"


def run_ml(df: pd.DataFrame, target: str = None) -> dict:
    df = df.dropna()
    if len(df) < 20:
        return {"task": "unknown", "error": "Not enough clean rows to train a reliable model (need at least 20)."}

    task = infer_task(df, target)
    result = {"task": task, "target": target}

    if task == "clustering":
        numeric_df = df.select_dtypes(include=np.number).copy()
        if numeric_df.shape[1] < 2:
            return {"task": task, "error": "Need at least 2 numeric columns for clustering."}
        X = StandardScaler().fit_transform(numeric_df)
        k = min(6, max(2, len(df) // 50))
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        result["n_clusters"] = k
        result["cluster_sizes"] = {str(k_): int(v) for k_, v in pd.Series(labels).value_counts().items()}
        result["model"] = "KMeans"
        return result

    if target not in df.columns:
        return {"task": task, "error": f"Target column '{target}' not found."}

    X = df.drop(columns=[target]).copy()
    y = df[target]

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.select_dtypes(include=np.number)
    if X.shape[1] == 0:
        return {"task": task, "error": "No usable feature columns after encoding."}

    if task == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        result["f1"] = round(float(f1_score(y_test, preds, average="weighted")), 4)
        result["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result["r2"] = round(float(r2_score(y_test, preds)), 4)
        result["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        fi = sorted(zip(X.columns, importances), key=lambda x: -x[1])
        result["feature_importance"] = [
            {"feature": str(f), "importance": round(float(i), 4)} for f, i in fi[:10]
        ]

    result["model"] = type(model).__name__
    result["n_train"] = int(len(X_train))
    result["n_test"] = int(len(X_test))

    try:
        result["shap_summary"] = _shap_summary(model, X_test)
    except Exception:
        pass  # shap is optional; silently skip if unavailable

    return result


def _shap_summary(model, X_test, top_n: int = 5) -> list:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.sample(min(100, len(X_test)), random_state=42))
    if isinstance(shap_values, list):  # multi-class classification
        shap_values = np.mean(np.abs(shap_values), axis=0)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_n]
    return [{"feature": str(X_test.columns[i]), "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in order]


def explain_ml_result(result: dict) -> str:
    from utils.llm import call_llm
    from prompts.prompts import ml_explain_prompt
    try:
        return call_llm(
            ml_explain_prompt(result),
            system="You are a machine learning expert explaining results to a business audience.",
            max_tokens=300,
        )
    except Exception as e:
        return f"(Explanation unavailable: {e})"
