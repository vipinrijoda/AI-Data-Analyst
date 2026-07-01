"""Agent: Visualization
Chooses chart types automatically based on column dtype/cardinality instead
of hardcoded chart requests: categorical -> bar, numeric -> histogram,
datetime+numeric -> line, two numerics -> scatter, all numerics -> heatmap.
"""
import pandas as pd
import numpy as np
import plotly.express as px


def auto_visualize(df: pd.DataFrame, max_charts: int = 8) -> list:
    charts = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # Time series: datetime x numeric -> line chart
    for col in datetime_cols[:1]:
        for num in numeric_cols[:1]:
            try:
                ts = df[[col, num]].dropna().sort_values(col).groupby(col)[num].sum().reset_index()
                fig = px.line(ts, x=col, y=num, title=f"{num} over {col}")
                charts.append({"title": f"{num} over time", "fig": fig})
            except Exception:
                pass

    # Low-cardinality categoricals -> bar chart of counts
    for col in categorical_cols[:3]:
        if 0 < df[col].nunique(dropna=True) <= 20:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count", title=f"Count by {col}")
            charts.append({"title": f"Count by {col}", "fig": fig})

    # Numeric columns -> histograms
    for col in numeric_cols[:4]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append({"title": f"Distribution of {col}", "fig": fig})

    # Correlation heatmap
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap",
                         color_continuous_scale="RdBu_r", aspect="auto")
        charts.append({"title": "Correlation Heatmap", "fig": fig})

    # Two numerics -> scatter
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                          title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
        charts.append({"title": f"{numeric_cols[0]} vs {numeric_cols[1]}", "fig": fig})

    # Categorical vs numeric -> boxplot
    if categorical_cols and numeric_cols:
        cat = next((c for c in categorical_cols if 1 < df[c].nunique(dropna=True) <= 10), None)
        if cat:
            fig = px.box(df, x=cat, y=numeric_cols[0], title=f"{numeric_cols[0]} by {cat}")
            charts.append({"title": f"{numeric_cols[0]} by {cat}", "fig": fig})

    return charts[:max_charts]
