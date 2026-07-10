"""
Agentic Data Analyst - Streamlit Dashboard

Upload a dataset -> a team of specialized agents (Supervisor, Data
Understanding, Cleaning, EDA, Visualization, Insight, ML, SQL, Chat, Report)
understands it, cleans it, analyzes it, and produces a downloadable report.
"""
import os
import io
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.data_utils import load_dataset, merge_datasets, profile_dataset, infer_data_quality_score, try_parse_dates
from agents.supervisor import run_full_pipeline, detect_intent
from agents.cleaning import suggest_cleaning, apply_cleaning
from agents.eda import run_eda
from agents.visualization import auto_visualize
from agents.insight import generate_insights
from agents.ml import run_ml, explain_ml_result
from agents.sql import ask_sql
from agents.chat import answer_question
from agents.report import generate_report

st.set_page_config(page_title="Agentic Data Analyst", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- session state
DEFAULTS = {
    "df": None,
    "original_df": None,
    "profile": None,
    "understanding": None,
    "eda_results": None,
    "charts": None,
    "insights": None,
    "recommendations": None,
    "cleaning_suggestions": None,
    "ml_results": None,
    "chat_history": [],
    "pipeline_ran": False,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------- sidebar: upload
with st.sidebar:
    st.title("📊 Agentic Data Analyst")
    st.caption("Multi-agent AI data analysis, powered by a supervisor + specialist agents.")

    provider = os.getenv("LLM_PROVIDER", "anthropic")
    st.markdown(f"**LLM provider:** `{provider}`")
    _required_key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}
    key_name = _required_key.get(provider)
    if key_name and not os.getenv(key_name):
        st.warning(f"Set {key_name} in your environment (.env) to enable AI agents.")

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload dataset(s)", type=["csv", "xls", "xlsx"], accept_multiple_files=True
    )
    parse_dates = st.checkbox("Auto-detect date columns", value=True)

    if uploaded_files and st.button("Load & Analyze", type="primary", use_container_width=True):
        with st.spinner("Loading dataset(s)..."):
            dfs = {f.name: load_dataset(f) for f in uploaded_files}
            df = merge_datasets(dfs)
            if parse_dates:
                df = try_parse_dates(df)
            st.session_state.original_df = df.copy()
            st.session_state.df = df
            st.session_state.pipeline_ran = False

        with st.spinner("Supervisor is routing to Data Understanding, EDA, Visualization, and Insight agents..."):
            final_state = run_full_pipeline(st.session_state.df)
            st.session_state.profile = final_state.get("profile")
            st.session_state.understanding = final_state.get("understanding")
            st.session_state.eda_results = final_state.get("eda_results")
            st.session_state.charts = final_state.get("charts")
            st.session_state.insights = final_state.get("insights")
            st.session_state.recommendations = final_state.get("recommendations")
            st.session_state.pipeline_ran = True
            if final_state.get("error"):
                st.error(f"Pipeline warning: {final_state['error']}")

    if st.session_state.df is not None:
        st.divider()
        st.metric("Rows", len(st.session_state.df))
        st.metric("Columns", len(st.session_state.df.columns))
        quality = infer_data_quality_score(st.session_state.df)
        st.metric("Data quality score", f"{quality}/100")

# ---------------------------------------------------------------- main area
if st.session_state.df is None:
    st.info("👈 Upload a CSV or Excel file (or several) in the sidebar, then click **Load & Analyze**.")
    st.markdown("""
### What this app does
1. **Understands** your dataset in plain English (business domain, what it tracks)
2. **Cleans** it - suggests a plan, you approve, then it applies the fixes
3. Runs **automatic EDA** - stats, missing values, duplicates, correlation, outliers
4. **Visualizes** it - chooses the right chart type per column automatically
5. Generates **business insights and recommendations**, not just numbers
6. Lets you **ask questions in natural language** (routed to a SQL agent or a chat agent)
7. Runs **ML** (classification / regression / clustering) with explainability
8. Produces a **downloadable PDF report**
""")
    st.stop()

df = st.session_state.df

tabs = st.tabs([
    "🧠 Overview", "🧹 Cleaning", "📈 EDA", "📊 Visualizations",
    "💡 Insights", "🤖 ML", "💬 Ask Questions", "📄 Report",
])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader("AI Dataset Understanding")
    if st.session_state.understanding:
        st.success(st.session_state.understanding)
    else:
        st.info("Run analysis from the sidebar to generate this.")

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    if st.session_state.profile:
        st.subheader("Column Profile")
        st.dataframe(pd.DataFrame(st.session_state.profile["columns"]), use_container_width=True)

# ---------------------------------------------------------------- Cleaning
with tabs[1]:
    st.subheader("Cleaning Agent")
    if st.session_state.profile is None:
        st.info("Run analysis first.")
    else:
        if st.button("Suggest cleaning plan"):
            with st.spinner("Analyzing data quality issues..."):
                st.session_state.cleaning_suggestions = suggest_cleaning(st.session_state.profile)

        suggestions = st.session_state.cleaning_suggestions
        if suggestions:
            st.write(f"Drop {suggestions.get('drop_duplicates') and 'duplicate rows: yes' or 'duplicate rows: no'}")
            cols = suggestions.get("columns", [])
            if cols:
                plan_df = pd.DataFrame(cols)
                st.dataframe(plan_df, use_container_width=True)
                approved = st.multiselect(
                    "Approve which column fixes to apply",
                    options=[c["column"] for c in cols],
                    default=[c["column"] for c in cols],
                )
                if st.button("Apply approved cleaning", type="primary"):
                    st.session_state.df = apply_cleaning(df, suggestions, approved_columns=approved)
                    st.success("Cleaning applied. Re-run analysis from the sidebar to refresh EDA/visuals.")
            else:
                st.success("No column-level issues detected.")

        st.download_button(
            "Download cleaned dataset (CSV)",
            data=st.session_state.df.to_csv(index=False).encode(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------- EDA
with tabs[2]:
    st.subheader("Automatic EDA")
    if st.button("Refresh EDA"):
        st.session_state.eda_results = run_eda(df)

    eda = st.session_state.eda_results
    if eda:
        c1, c2, c3 = st.columns(3)
        c1.metric("Duplicate rows", eda["duplicate_rows"])
        c2.metric("Columns with missing values", sum(1 for v in eda["missing_values"].values() if v > 0))
        c3.metric("Columns with outliers", sum(1 for v in eda["outliers"].values() if v > 0))

        st.markdown("**Missing values per column**")
        st.bar_chart(pd.Series(eda["missing_values"]))

        if eda["correlation"]:
            st.markdown("**Correlation matrix**")
            st.dataframe(pd.DataFrame(eda["correlation"]).round(3), use_container_width=True)

        if eda["outliers"]:
            st.markdown("**Outlier counts (IQR method)**")
            st.dataframe(pd.DataFrame([eda["outliers"]]).T.rename(columns={0: "outlier_count"}))
    else:
        st.info("Run analysis first.")

# ---------------------------------------------------------------- Visualizations
with tabs[3]:
    st.subheader("Auto-Selected Visualizations")
    if st.button("Regenerate charts"):
        st.session_state.charts = auto_visualize(df)

    charts = st.session_state.charts
    if charts:
        for i in range(0, len(charts), 2):
            cols = st.columns(2)
            for col, chart in zip(cols, charts[i:i + 2]):
                with col:
                    st.plotly_chart(chart["fig"], use_container_width=True, key=chart["title"])
    else:
        st.info("Run analysis first.")

# ---------------------------------------------------------------- Insights
with tabs[4]:
    st.subheader("Business Insights & Recommendations")
    if st.button("Generate insights"):
        with st.spinner("Insight agent is analyzing patterns..."):
            result = generate_insights(st.session_state.profile or profile_dataset(df),
                                        st.session_state.eda_results or run_eda(df))
            st.session_state.insights = result.get("insights", [])
            st.session_state.recommendations = result.get("recommendations", [])

    if st.session_state.insights:
        st.markdown("**Insights**")
        for insight in st.session_state.insights:
            st.markdown(f"- {insight}")
    if st.session_state.recommendations:
        st.markdown("**Recommendations**")
        for i, rec in enumerate(st.session_state.recommendations, 1):
            st.markdown(f"{i}. {rec}")
    if not st.session_state.insights:
        st.info("Click 'Generate insights' or run analysis from the sidebar.")

# ---------------------------------------------------------------- ML
with tabs[5]:
    st.subheader("Machine Learning Agent")
    st.caption("Leave target blank for unsupervised clustering, or pick a column to predict.")
    target = st.selectbox("Target column (optional)", options=["(none - cluster)"] + list(df.columns))
    target = None if target == "(none - cluster)" else target

    if st.button("Run ML", type="primary"):
        with st.spinner("Training model..."):
            st.session_state.ml_results = run_ml(df, target=target)

    result = st.session_state.ml_results
    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            st.write(f"**Task:** {result['task']}  |  **Model:** {result.get('model')}")
            cols = st.columns(3)
            metric_keys = [k for k in ("accuracy", "f1", "r2", "mae", "n_clusters") if k in result]
            for c, k in zip(cols, metric_keys):
                c.metric(k, result[k])

            if "feature_importance" in result:
                st.markdown("**Feature importance**")
                st.dataframe(pd.DataFrame(result["feature_importance"]), use_container_width=True)

            if "confusion_matrix" in result:
                st.markdown("**Confusion matrix**")
                st.dataframe(pd.DataFrame(result["confusion_matrix"]))

            if "cluster_sizes" in result:
                st.markdown("**Cluster sizes**")
                st.bar_chart(pd.Series(result["cluster_sizes"]))

            if "shap_summary" in result:
                st.markdown("**SHAP feature impact (mean |SHAP|)**")
                st.dataframe(pd.DataFrame(result["shap_summary"]), use_container_width=True)

            if st.button("Explain these results in plain language"):
                with st.spinner("Explaining..."):
                    st.info(explain_ml_result(result))
    else:
        st.info("Configure a target (or leave blank for clustering) and click Run ML.")

# ---------------------------------------------------------------- Ask Questions (SQL + Chat)
with tabs[6]:
    st.subheader("Ask Questions")
    st.caption("Routed automatically: aggregation/lookup questions go to the SQL agent; "
               "explanatory/follow-up questions go to the Chat agent.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None:
                st.dataframe(msg["table"], use_container_width=True)

    question = st.chat_input("e.g. 'Which city generated the highest revenue?' or 'Why are profits falling?'")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        profile = st.session_state.profile or profile_dataset(df)
        intent = detect_intent(question)

        with st.chat_message("assistant"):
            with st.spinner(f"Routed to {intent} agent..."):
                if intent == "sql":
                    sql_result = ask_sql(question, df)
                    if sql_result["error"]:
                        st.error(f"SQL agent error: {sql_result['error']}")
                        content = f"I couldn't answer that with a query: {sql_result['error']}"
                        table = None
                    else:
                        st.code(sql_result["sql"], language="sql")
                        st.dataframe(sql_result["result"], use_container_width=True)
                        content = f"Query executed. See table above.\n```sql\n{sql_result['sql']}\n```"
                        table = sql_result["result"]
                else:
                    chat_result = answer_question(question, df, st.session_state.chat_history, profile)
                    st.markdown(chat_result["answer"])
                    if chat_result.get("table") is not None:
                        st.dataframe(chat_result["table"], use_container_width=True)
                    content = chat_result["answer"]
                    table = chat_result.get("table")

        st.session_state.chat_history.append({"role": "assistant", "content": content, "table": table})

# ---------------------------------------------------------------- Report
with tabs[7]:
    st.subheader("Report Generator")
    if st.button("Generate PDF report", type="primary"):
        with st.spinner("Rendering charts and assembling report..."):
            tmpdir = tempfile.mkdtemp()
            chart_images = []
            charts = st.session_state.charts or auto_visualize(df)
            for i, chart in enumerate(charts):
                img_path = os.path.join(tmpdir, f"chart_{i}.png")
                try:
                    chart["fig"].write_image(img_path, width=900, height=540, scale=2)
                    chart_images.append((img_path, chart["title"]))
                except Exception:
                    pass  # kaleido may not be installed; report still builds without images

            report_path = os.path.join(tmpdir, "report.pdf")
            generate_report(
                output_path=report_path,
                understanding=st.session_state.understanding or "",
                profile=st.session_state.profile or profile_dataset(df),
                eda_results=st.session_state.eda_results or {},
                insights=st.session_state.insights or [],
                recommendations=st.session_state.recommendations or [],
                ml_results=st.session_state.ml_results or {},
                chart_images=chart_images,
            )
            with open(report_path, "rb") as f:
                st.session_state["report_bytes"] = f.read()
        st.success("Report generated.")

    if st.session_state.get("report_bytes"):
        st.download_button(
            "Download report.pdf",
            data=st.session_state["report_bytes"],
            file_name="data_analysis_report.pdf",
            mime="application/pdf",
        )

# Agentic Data Analyst
# Powered by Groq LLM and LangGraph
