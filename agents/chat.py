"""Agent: Chat
Answers free-form follow-up questions, grounded in real query results (via
the SQL agent) rather than the LLM's own guesses, and keeps short-term
conversational memory so "explain chart 3" style follow-ups work.
"""
import pandas as pd
from utils.llm import call_llm
from agents.sql import ask_sql
from prompts.prompts import chat_prompt


def answer_question(question: str, df: pd.DataFrame, chat_history: list, profile: dict) -> dict:
    sql_result = ask_sql(question, df)
    if sql_result["error"] is None and sql_result["result"] is not None:
        grounding = sql_result["result"].head(20).to_string(index=False)
    else:
        grounding = "No structured query result available for this question; answer from the dataset profile only, and say so if the profile doesn't contain the answer."

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history[-6:])
    prompt = chat_prompt(question, grounding, profile, history_text)
    try:
        answer = call_llm(
            prompt,
            system=("You are a data analyst answering questions grounded strictly in the "
                     "provided query results. Never invent numbers not present in the data."),
            max_tokens=500,
        )
    except Exception as e:
        answer = f"(Chat response unavailable: {e})"

    return {"answer": answer, "sql": sql_result.get("sql"), "table": sql_result.get("result")}
