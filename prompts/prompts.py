"""Centralized prompt templates. Keeping them here (instead of inline in each
agent) makes them easy to tune without touching agent logic."""
import json


def data_understanding_prompt(profile: dict) -> str:
    cols = "\n".join(
        f"- {c['name']} ({c['dtype']}, {c['n_unique']} unique, {c['pct_missing']}% missing)"
        for c in profile["columns"]
    )
    sample = json.dumps(profile["sample"][:3], default=str, indent=2)
    return f"""Given this dataset profile, write 2-3 sentences describing what business domain this
dataset most likely represents and what it tracks. Be specific and confident, the way a senior
analyst would describe it after a first look. Do not just restate the column list.

Rows: {profile['n_rows']}, Columns: {profile['n_cols']}

Columns:
{cols}

Sample rows:
{sample}
"""


def cleaning_prompt(profile: dict) -> str:
    cols = "\n".join(
        f"- {c['name']} ({c['dtype']}): {c['pct_missing']}% missing, {c['n_unique']} unique"
        for c in profile["columns"]
    )
    return f"""Analyze this dataset profile and produce a data cleaning plan.

Return JSON with EXACTLY this shape and nothing else:
{{"drop_duplicates": true or false,
  "columns": [{{"column": "name", "issue": "short description", "strategy": "mean|median|mode|drop|drop_column"}}]}}

Rules:
- Only include columns that actually need cleaning (missing values, or >50% missing -> drop_column).
- Use "mean" or "median" only for numeric columns; use "mode" for categorical columns.
- "drop" means drop rows with missing values in that column (use sparingly).

Duplicate rows: {profile['n_duplicates']}
Columns:
{cols}
"""


def insight_prompt(profile: dict, eda_results: dict) -> str:
    return f"""You are analyzing a business dataset. Based on the profile and EDA results below,
generate concrete, specific business insights (not generic statistics restated in words) and
actionable recommendations a stakeholder could act on this week.

Return JSON with EXACTLY this shape and nothing else:
{{"insights": ["...", "..."], "recommendations": ["...", "..."]}}

Generate 4-6 insights and 3-5 recommendations. Reference actual numbers/columns from the data
where possible.

Dataset profile: {json.dumps(profile, default=str)[:3000]}
EDA results: {json.dumps(eda_results, default=str)[:3000]}
"""


def ml_explain_prompt(result: dict) -> str:
    return f"""Explain these machine learning results in plain business language, 3-4 sentences.
Focus on what the model relies on to make predictions and what that means practically for the
business. Avoid restating raw numbers without interpreting them.

Results: {json.dumps(result, default=str)[:2000]}
"""


def sql_prompt(question: str, schema: str) -> str:
    return f"""Table name: data
Schema: {schema}

Convert this natural language question into a single valid SQLite SQL query against the "data"
table. Use only columns from the schema above. Do not invent columns.

Question: {question}

Output ONLY the SQL query - no explanation, no markdown fences.
"""


def chat_prompt(question: str, grounding: str, profile: dict, history: str) -> str:
    col_names = [c["name"] for c in profile.get("columns", [])]
    return f"""Conversation so far:
{history}

Query result grounding the answer (produced by executing SQL against the real uploaded data):
{grounding}

Dataset context: {profile.get('n_rows')} rows. Columns: {col_names}

User question: {question}

Answer using ONLY the grounding data above. If the grounding does not contain enough information
to answer confidently, say so honestly instead of guessing.
"""


def intent_prompt(question: str) -> str:
    return f"""Classify this user question into exactly one category:
"sql" - it asks for an aggregation, filter, ranking, or lookup that a database query could answer
        (e.g. "top 5 customers by revenue", "average salary by department")
"chat" - it's a general question, an explanation request, or a follow-up about prior results
        (e.g. "why are profits falling", "explain chart 3", "what does this mean")

Question: {question}

Respond with only one word: sql or chat.
"""
