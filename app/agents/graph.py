"""
LangGraph Orchestrator — wires all agents into a StateGraph.
Streaming: stream_pipeline() uses astream_events(v2) to yield only synthesize tokens.

Optimized parallel flow:

  START
   ├──► understand         (LLM: detect needs_rewrite)
   └──► prefetch           (hybrid search on original query)
          ↓  (both complete, merge at decide)
       decide
        ├── needs_rewrite=False ──► synthesize  (prefetched docs reused, no extra LLM call)
        └── needs_rewrite=True  ──► rewrite ──► route ──► retrieve ──► synthesize
                                       ↓
                               conversation_summary  (conditional)
                                       ↓
                               memory_manager  ──► END
"""

import json
from typing import AsyncGenerator
from langgraph.graph import StateGraph, END, START
from app.core.state import ConversationState
from app.db.semantic_cache import check_cache, store_cache
from app.agents.agents import (
    query_understanding_agent,
    query_rewriting_agent,
    retrieval_router_agent,
    retriever_agent,
    prefetch_retriever_agent,
    decide_agent,
    context_synthesis_agent,
    conversation_summary_agent,
    memory_manager_agent,
)
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)

settings = get_settings()


def should_rewrite(state: ConversationState) -> str:
    """After decide_agent runs, route to synthesize directly or go through rewrite."""
    return "rewrite" if state.get("needs_rewrite", False) else "synthesize"


def should_summarize(state: ConversationState) -> str:
    return "summarize" if len(state.get("chat_history", [])) >= settings.summary_threshold else "memory"


def build_graph() -> StateGraph:
    graph = StateGraph(ConversationState)

    # Register nodes
    graph.add_node("understand", query_understanding_agent)
    graph.add_node("prefetch", prefetch_retriever_agent)
    graph.add_node("decide", decide_agent)
    graph.add_node("rewrite", query_rewriting_agent)
    graph.add_node("route", retrieval_router_agent)
    graph.add_node("retrieve", retriever_agent)
    graph.add_node("synthesize", context_synthesis_agent)
    graph.add_node("summarize", conversation_summary_agent)
    graph.add_node("memory", memory_manager_agent)

    # ── Parallel fan-out from START ───────────────────────────────────────────
    # Both `understand` (LLM intent detection) and `prefetch` (hybrid search)
    # kick off simultaneously so retrieval never waits on the LLM.
    graph.add_edge(START, "understand")
    graph.add_edge(START, "prefetch")

    # ── Merge: both branches must complete before `decide` runs ──────────────
    graph.add_edge("understand", "decide")
    graph.add_edge("prefetch", "decide")

    # ── decide → synthesize (fast path) OR → rewrite (slow path) ─────────────
    graph.add_conditional_edges(
        "decide",
        should_rewrite,
        {"synthesize": "synthesize", "rewrite": "rewrite"},
    )

    # ── Slow path: rewrite → route → fresh retrieve → synthesize ─────────────
    graph.add_edge("rewrite", "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "synthesize")

    # ── Post-synthesis: optional summarization, then memory ──────────────────
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


# Function for create Graph Diagram 

app = build_graph()

# 2. Generate the PNG data
# draw_mermaid_png() uses an external API by default to render the image
graph_image_data = app.get_graph().draw_mermaid_png()

# 3. Save to a file
with open("agent_workflow.png", "wb") as f:
    f.write(graph_image_data)

logger.info("Agent diagram saved as 'agent_workflow.png'")



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
    """Run pipeline — checks semantic cache first, skips agents on hit."""

    # ── Cache check (before touching LangGraph) ───────────────────────────────
    cached = check_cache(query)
    if cached:
        return {
            "response": cached["answer"],
            "sources": cached["sources"],
            "rewritten_query": None,
            "retrieval_strategy": "cache",
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
        "prefetched_docs": [],
        "retrieved_docs": [],
        "chat_history": chat_history,
        "conversation_summary": conversation_summary,
        "response": "",
        "sources": [],
        "messages": [],
    }

    final_state = await graph.ainvoke(initial_state)

    store_cache(query, final_state["response"], final_state["sources"])

    return {
        "response": final_state["response"],
        "sources": final_state["sources"],
        "rewritten_query": final_state.get("rewritten_query"),
        "retrieval_strategy": final_state.get("retrieval_strategy"),
        "from_cache": False,
    }


async def stream_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    chat_history: list[dict],
    conversation_summary: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream only the synthesize agent tokens via SSE.

    Yields newline-delimited JSON strings:
      - Token chunks: {"type": "token", "content": "..."}
      - Final metadata: {"type": "done", "sources": [...], "rewritten_query": ...,
                         "retrieval_strategy": ..., "from_cache": bool}
      - Cache hit:     {"type": "done", "from_cache": true, "response": "...", ...}
    """

    # ── Cache check — return immediately without streaming ────────────────────
    cached = check_cache(query)
    if cached:
        yield json.dumps({
            "type": "done",
            "response": cached["answer"],
            "sources": cached["sources"],
            "rewritten_query": None,
            "retrieval_strategy": "cache",
            "from_cache": True,
            "cache_score": cached["cache_score"],
            "cached_query": cached["cached_query"],
        }) + "\n"
        return

    # ── Full pipeline — stream synthesize tokens ──────────────────────────────
    graph = get_graph()

    initial_state: ConversationState = {
        "session_id": session_id,
        "user_id": user_id,
        "query": query,
        "rewritten_query": None,
        "needs_rewrite": False,
        "retrieval_strategy": "hybrid",
        "prefetched_docs": [],
        "retrieved_docs": [],
        "chat_history": chat_history,
        "conversation_summary": conversation_summary,
        "response": "",
        "sources": [],
        "messages": [],
    }

    full_response = ""
    final_state = None

    async for event in graph.astream_events(initial_state, version="v2"):
        kind = event["event"]
        tags = event.get("tags", []) or []
        metadata = event.get("metadata", {}) or {}

        # LangGraph tags the node name in metadata["langgraph_node"]
        node = metadata.get("langgraph_node", "")

        # ── Stream tokens only from synthesize node ───────────────────────────
        if kind == "on_chat_model_stream" and node == "synthesize":
            chunk = event["data"]["chunk"]
            token = chunk.content  # AIMessageChunk.content
            if token:
                full_response += token
                yield json.dumps({"type": "token", "content": token}) + "\n"

        # ── Capture final graph state after all nodes finish ──────────────────
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            final_state = event["data"].get("output", {})

    # ── Store in cache and emit metadata chunk ────────────────────────────────
    sources = final_state.get("sources", []) if final_state else []
    store_cache(query, full_response, sources)

    yield json.dumps({
        "type": "done",
        "sources": sources,
        "rewritten_query": final_state.get("rewritten_query") if final_state else None,
        "retrieval_strategy": final_state.get("retrieval_strategy") if final_state else None,
        "from_cache": False,
    }) + "\n"