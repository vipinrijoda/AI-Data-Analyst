"""Shared state object passed between agents in the LangGraph pipeline."""
from typing import TypedDict, Any, List, Dict, Optional
import pandas as pd


class AgentState(TypedDict, total=False):
    df: pd.DataFrame
    original_df: pd.DataFrame
    profile: dict
    user_query: str
    intent: str
    understanding: str
    eda_results: dict
    cleaning_suggestions: dict
    cleaned: bool
    charts: List[dict]
    insights: List[str]
    recommendations: List[str]
    ml_results: dict
    sql_query: str
    sql_result: Any
    chat_history: List[dict]
    chat_answer: str
    report_path: str
    error: Optional[str]
