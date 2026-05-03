"""
LangGraph Orchestrator — wires all agents into a StateGraph.

Flow:
  query_understanding
       ↓
  query_rewriting  (skipped if no rewrite needed)
       ↓
  retrieval_router
       ↓
  retriever
       ↓
  context_synthesis
       ↓
  conversation_summary  (conditional: only if long conversation)
       ↓
  memory_manager
       ↓
  END
"""

from langgraph.graph import StateGraph, END
from app.core.state import ConversationState
from app.agents.agents import (
    query_understanding_agent,
    query_rewriting_agent,
    retrieval_router_agent,
    retriever_agent,
    context_synthesis_agent,
    conversation_summary_agent,
    memory_manager_agent,
)
from app.core.config import get_settings

settings = get_settings()


def should_rewrite(state: ConversationState) -> str:
    return "rewrite" if state.get("needs_rewrite", False) else "route"


def should_summarize(state: ConversationState) -> str:
    return "summarize" if len(state.get("chat_history", [])) >= settings.summary_threshold else "memory"


def build_graph() -> StateGraph:
    graph = StateGraph(ConversationState)

    # Register nodes
    graph.add_node("understand", query_understanding_agent)
    graph.add_node("rewrite", query_rewriting_agent)
    graph.add_node("route", retrieval_router_agent)
    graph.add_node("retrieve", retriever_agent)
    graph.add_node("synthesize", context_synthesis_agent)
    graph.add_node("summarize", conversation_summary_agent)
    graph.add_node("memory", memory_manager_agent)

    # Entry point
    graph.set_entry_point("understand")

    # Conditional: rewrite or go straight to route
    graph.add_conditional_edges(
        "understand",
        should_rewrite,
        {"rewrite": "rewrite", "route": "route"},
    )

    graph.add_edge("rewrite", "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "synthesize")

    # Conditional: summarize or go straight to memory
    graph.add_conditional_edges(
        "synthesize",
        should_summarize,
        {"summarize": "summarize", "memory": "memory"},
    )

    graph.add_edge("summarize", "memory")
    graph.add_edge("memory", END)

    return graph.compile()


# Compiled graph singleton
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    chat_history: list[dict],
    conversation_summary: str | None = None,
) -> dict:
    """Run the full RAG pipeline and return response + sources."""
    graph = get_graph()

    initial_state: ConversationState = {
        "session_id": session_id,
        "user_id": user_id,
        "query": query,
        "rewritten_query": None,
        "needs_rewrite": False,
        "retrieval_strategy": "hybrid",
        "retrieved_docs": [],
        "chat_history": chat_history,
        "conversation_summary": conversation_summary,
        "response": "",
        "sources": [],
        "messages": [],
    }

    final_state = await graph.ainvoke(initial_state)

    return {
        "response": final_state["response"],
        "sources": final_state["sources"],
        "rewritten_query": final_state.get("rewritten_query"),
        "retrieval_strategy": final_state.get("retrieval_strategy"),
    }

# ── Only the run_pipeline function changes in graph.py ────────────────────────
# Add these two imports at the top of graph.py:
#
from app.db.semantic_cache import check_cache, store_cache
#
# app\db\semantic_cache.py
# Then replace run_pipeline with this:

async def run_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    chat_history: list[dict],
    conversation_summary: str | None = None,
) -> dict:
    """Run pipeline — checks semantic cache first, skips agents on hit."""

    # ── Cache check (before touching LangGraph) ───────────────────────────────
    cached = check_cache(query)
    if cached:
        return {
            "response": cached["answer"],
            "sources": cached["sources"],
            "rewritten_query": None,
            "retrieval_strategy": "cache",          # signals a cache hit to caller
            "from_cache": True,
            "cache_score": cached["cache_score"],
            "cached_query": cached["cached_query"],
        }

    # ── Full LangGraph pipeline ───────────────────────────────────────────────
    graph = get_graph()

    initial_state: ConversationState = {
        "session_id": session_id,
        "user_id": user_id,
        "query": query,
        "rewritten_query": None,
        "needs_rewrite": False,
        "retrieval_strategy": "hybrid",
        "retrieved_docs": [],
        "chat_history": chat_history,
        "conversation_summary": conversation_summary,
        "response": "",
        "sources": [],
        "messages": [],
    }

    final_state = await graph.ainvoke(initial_state)

    # ── Store result in cache for future similar queries ──────────────────────
    store_cache(query, final_state["response"], final_state["sources"])

    return {
        "response": final_state["response"],
        "sources": final_state["sources"],
        "rewritten_query": final_state.get("rewritten_query"),
        "retrieval_strategy": final_state.get("retrieval_strategy"),
        "from_cache": False,
    }