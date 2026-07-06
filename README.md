# Agentic Data Analyst

Upload any dataset → a supervisor routes work to specialized AI agents that understand the business
context, clean the data, run EDA, choose the right visualizations, train ML models, answer natural
language questions, and generate a professional PDF report.

## Architecture

```
                User
                  │
                  ▼
         Streamlit Dashboard
                  │
                  ▼
          Supervisor Agent (LangGraph)
                  │
     ─────────────┼─────────────
                  │
      Intent Detection (sql | chat)
                  │
 ───────────────────────────────────────────
 EDA · Cleaning · Visualization · ML
 Insight · SQL · Report · Chat
 ───────────────────────────────────────────
                  │
             LLM (Claude / GPT)
                  │
          Pandas / Scikit-learn
                  │
            Final Dashboard
```

## Agents

| Agent | Responsibility |
|---|---|
| Supervisor | Runs the core LangGraph pipeline and routes chat questions |
| Data Understanding | Describes the dataset's business context in plain English |
| Cleaning | Detects and (with approval) fixes data quality issues |
| EDA | Computes descriptive statistics, missing values, duplicates, correlation, outliers |
| Visualization | Chooses and generates the right chart type per column automatically |
| Insight | Writes concrete, actionable business insights and recommendations |
| ML | Infers classification / regression / clustering, trains a model, explains it (SHAP optional) |
| SQL | Converts natural language into SQLite queries, executes them, returns grounded results |
| Chat | Answers follow-up questions, grounded in SQL agent output, with short-term memory |
| Report | Assembles a downloadable PDF: summary, EDA, charts, insights, ML results, recommendations |

## Quickstart

```bash
cd agentic-data-analyst
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
streamlit run app.py
```

### Using Groq instead of Anthropic

Groq's API is OpenAI-compatible, so it's supported as a third `LLM_PROVIDER` option (no extra
packages needed - it reuses the `openai` SDK pointed at Groq's endpoint). In `.env`:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here      # free tier at https://console.groq.com/keys
GROQ_MODEL=llama-3.3-70b-versatile   # or llama-3.1-8b-instant, mixtral-8x7b-32768, etc.
```

Groq is fast and has a generous free tier, but keep in mind smaller/faster models are less reliable
at strict JSON output (used by the Cleaning, Insight, and Intent-detection agents) - if you see
JSON parsing fallbacks in the sidebar, try a larger Groq model like `llama-3.3-70b-versatile`.

Optional: `pip install kaleido` to enable chart images inside the generated PDF report.

## How it prevents hallucination

The Chat and SQL agents never let the LLM invent numbers. The LLM's only job is to translate a
natural-language question into a SQL query; that query is executed with `sqlite3`/`pandas` against
the real uploaded data, and the LLM's final answer is grounded in the *executed result*, not its own
guess. EDA, cleaning suggestions, and ML metrics are all computed directly with pandas/scikit-learn -
the LLM is only used for narrative explanation layered on top of real numbers.

## Error handling

Every agent node in the Supervisor's LangGraph pipeline is wrapped in try/except. If one agent fails
(e.g. the LLM call times out), the error is captured in shared state and surfaced in the Streamlit UI
instead of crashing the app; downstream agents in the on-demand tabs (ML, SQL, Chat, Report) can still
be run independently.

## Large datasets

For datasets with 1M+ rows, consider sampling before the automatic EDA/visualization pass
(`df.sample(n=100_000, random_state=42)`) and running the full pipeline on the sample, while keeping
SQL agent queries against the full data for accurate aggregates. This is not enabled by default so the
app stays predictable for smaller datasets - wire it into `app.py`'s upload handler if needed.

## Project structure

```
agentic-data-analyst/
├── app.py                  # Streamlit dashboard
├── agents/
│   ├── supervisor.py        # LangGraph orchestration + intent routing
│   ├── data_understanding.py
│   ├── cleaning.py
│   ├── eda.py
│   ├── visualization.py
│   ├── insight.py
│   ├── ml.py
│   ├── sql.py
│   ├── report.py
│   └── chat.py
├── utils/
│   ├── llm.py                # Anthropic/OpenAI abstraction
│   └── data_utils.py         # load/merge/profile helpers
├── prompts/
│   └── prompts.py            # all prompt templates in one place
├── datasets/                 # sample data to try the app with
├── reports/                  # generated PDF reports land here
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## What interviewers are likely to ask (and where the answers live)

- **Why multi-agent instead of one LLM call?** Each agent has a narrow, testable responsibility
  (`agents/*.py`); the Supervisor (`agents/supervisor.py`) composes them, so failures are isolated
  and each step can be swapped/improved independently.
- **How does the supervisor decide which agent to invoke?** The core pipeline is a fixed LangGraph
  (`understand → eda → visualize → insight`); free-form questions go through `detect_intent()`, an
  LLM-based classifier that routes to the SQL agent or Chat agent.
- **How do agents communicate?** Through a shared `AgentState` TypedDict (`agents/state.py`) passed
  along LangGraph edges.
- **How do you prevent hallucination?** See "How it prevents hallucination" above.
- **How do you handle large datasets?** See "Large datasets" above.
- **How do you evaluate insight quality?** Insights are grounded in the same EDA/profile JSON passed
  to the LLM (`prompts/prompts.py::insight_prompt`), so they can be spot-checked against the EDA tab.
  Add a regression test in `tests/` that asserts key phrases/numbers from `eda_results` appear in
  generated insights for stronger evaluation.
- **What happens when an agent fails?** See "Error handling" above.
# Add new content to README
echo "" >> README.md
echo "## Features" >> README.md
echo "- Multi-agent architecture" >> README.md
echo "- Natural language queries" >> README.md
echo "- Automatic visualization" >> README.md
echo "- ML model training" >> README.md
echo "- Professional PDF reports" >> README.md

## Features
- Multi-agent architecture
- Natural language queries
- Automatic visualization
- ML model training
- Professional PDF reports

## Project Structure
```
data_analysis_agent/
├── app.py              # Streamlit dashboard
├── agents/             # Multi-agent system
├── utils/              # Helper functions
├── prompts/            # Prompt templates
└── requirements.txt    # Dependencies
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Architecture
```
User -> Streamlit -> Supervisor -> Agents -> LLM -> Results
```
