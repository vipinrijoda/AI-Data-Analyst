"""Agent: Supervisor
Orchestrates the core pipeline (Data Understanding -> EDA -> Visualization ->
Insight) as a LangGraph state graph, and exposes an Intent Detection helper
used by the Streamlit app to route free-form questions to either the SQL
agent or the Chat agent. Each node is wrapped in try/except so one agent
failing doesn't crash the whole pipeline - the error is captured in state
and surfaced in the UI instead.
"""
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.data_understanding import understand_dataset
from agents.eda import run_eda
from agents.visualization import auto_visualize
from agents.insight import generate_insights
from utils.data_utils import profile_dataset


def node_understand(state: AgentState) -> AgentState:
    try:
        state["profile"] = profile_dataset(state["df"])
        state["understanding"] = understand_dataset(state["profile"])
    except Exception as e:
        state["error"] = f"data_understanding failed: {e}"
    return state


def node_eda(state: AgentState) -> AgentState:
    try:
        state["eda_results"] = run_eda(state["df"])
    except Exception as e:
        state["error"] = f"eda failed: {e}"
    return state


def node_visualization(state: AgentState) -> AgentState:
    try:
        state["charts"] = auto_visualize(state["df"])
    except Exception as e:
        state["error"] = f"visualization failed: {e}"
    return state


def node_insight(state: AgentState) -> AgentState:
    try:
        result = generate_insights(state.get("profile", {}), state.get("eda_results", {}))
        state["insights"] = result.get("insights", [])
        state["recommendations"] = result.get("recommendations", [])
    except Exception as e:
        state["error"] = f"insight failed: {e}"
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("understand", node_understand)
    graph.add_node("eda", node_eda)
    graph.add_node("visualize", node_visualization)
    graph.add_node("insight", node_insight)

    graph.set_entry_point("understand")
    graph.add_edge("understand", "eda")
    graph.add_edge("eda", "visualize")
    graph.add_edge("visualize", "insight")
    graph.add_edge("insight", END)
    return graph.compile()


def run_full_pipeline(df) -> AgentState:
    """Runs the automatic (no-click) analysis pipeline end to end."""
    graph = build_graph()
    initial_state: AgentState = {"df": df, "original_df": df.copy(), "chat_history": []}
    return graph.invoke(initial_state)


def detect_intent(question: str) -> str:
    """Routes a free-form user question to 'sql' or 'chat'."""
    from utils.llm import call_llm
    from prompts.prompts import intent_prompt
    try:
        raw = call_llm(intent_prompt(question), max_tokens=10).strip().lower()
        return "sql" if "sql" in raw else "chat"
    except Exception:
        return "chat"
