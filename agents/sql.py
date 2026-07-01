"""Agent: SQL
Converts a natural language question into a SQLite query against the
uploaded dataset, executes it, and returns the grounded result. This is the
mechanism used to prevent LLM hallucination when answering data questions -
the LLM only writes the query; pandas/sqlite compute the actual numbers.
"""
import sqlite3
import pandas as pd
from utils.llm import call_llm
from prompts.prompts import sql_prompt


def nl_to_sql(question: str, df: pd.DataFrame) -> str:
    schema = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
    prompt = sql_prompt(question, schema)
    raw = call_llm(
        prompt,
        system="You are an expert SQLite query generator. Output ONLY the SQL query, nothing else.",
        max_tokens=300,
    )
    sql = raw.strip().strip("`")
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


def run_sql(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql("data", conn, index=False, if_exists="replace")
        return pd.read_sql_query(sql_query, conn)
    finally:
        conn.close()


def ask_sql(question: str, df: pd.DataFrame) -> dict:
    """End-to-end NL -> SQL -> executed result, with error surfaced (not raised)
    so calling agents can decide how to recover."""
    try:
        sql_query = nl_to_sql(question, df)
    except Exception as e:
        return {"sql": None, "result": None, "error": f"Could not generate SQL: {e}"}

    try:
        result = run_sql(df, sql_query)
        return {"sql": sql_query, "result": result, "error": None}
    except Exception as e:
        return {"sql": sql_query, "result": None, "error": str(e)}
