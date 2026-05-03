"""
LangGraph Orchestrator — with streaming support on final response only.
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.state import ConversationState
from app.agents.agents import (
    query_understanding_agent,
    query_rewriting_agent,
    retrieval_router_agent,
    retriever_agent,
    conversation_summary_agent,
    memory_manager_agent,
)
from app.db.semantic_cache import check_cache, store_cache
from app.core.config import get_settings
from typing import AsyncIterator

settings = get_settings()

llm_stream = ChatOpenAI(
    model=settings.llm_model,
    openai_api_key=settings.openai_api_key,
    temperature=0.4,
    streaming=True,
)


def should_rewrite(state: ConversationState) -> str:
    print('dddddd',state)
    return "rewrite" if state.get("needs_rewrite", False) else "route"


def should_summarize(state: ConversationState) -> str:
    return "summarize" if len(state.get("chat_history", [])) >= settings.summary_threshold else "memory"


def _history_text(chat_history: list[dict], limit: int = 6) -> str:
    recent = chat_history[-limit:] if len(chat_history) > limit else chat_history
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)


def build_graph() -> StateGraph:
    graph = StateGraph(ConversationState)

    graph.add_node("understand", query_understanding_agent)
    graph.add_node("rewrite", query_rewriting_agent)
    graph.add_node("route", retrieval_router_agent)
    graph.add_node("retrieve", retriever_agent)
    graph.add_node("summarize", conversation_summary_agent)
    graph.add_node("memory", memory_manager_agent)

    graph.set_entry_point("understand")

    graph.add_conditional_edges(
        "understand",
        should_rewrite,
        {"rewrite": "rewrite", "route": "route"},
    )
    graph.add_edge("rewrite", "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", END)          # retrieval done; streaming handled outside

    return graph.compile()


_graph = None


# Function for create Graph Diagram 

app = build_graph()

# 2. Generate the PNG data
# draw_mermaid_png() uses an external API by default to render the image
graph_image_data = app.get_graph().draw_mermaid_png()

# 3. Save to a file
with open("agent_workflow.png", "wb") as f:
    f.write(graph_image_data)

print("Agent diagram saved as 'agent_workflow.png'")



def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _build_synthesis_prompt(state: ConversationState) -> tuple[SystemMessage, HumanMessage]:
    """Build the context synthesis prompt from pipeline state."""
    docs = state.get("retrieved_docs", [])
    history = _history_text(state.get("chat_history", []))
    summary = state.get("conversation_summary", "")

    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Source {i}]\n{doc['text']}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
    memory_context = f"\nConversation Summary:\n{summary}\n" if summary else ""

    system = SystemMessage(content=(
        "You are a helpful technical assistant. Answer based on the provided context. "
        "Be precise and cite sources using [Source N] notation. "
        "If information is not in context, say so."
    ))
    user = HumanMessage(content=(
        f"{memory_context}"
        f"Recent Conversation:\n{history}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"User Question: {state['query']}\n\nAnswer:"
    ))
    return system, user


def _extract_sources(state: ConversationState) -> list[dict]:
    return [
        {
            "index": i,
            "source": doc["metadata"].get("source", "unknown"),
            "section": doc["metadata"].get("section", ""),
            "score": round(doc.get("score", 0), 3),
        }
        for i, doc in enumerate(state.get("retrieved_docs", []), 1)
    ]


async def run_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    chat_history: list[dict],
    conversation_summary: str | None = None,
) -> dict:
    """Non-streaming pipeline (used internally / cache path)."""

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

    # Non-streaming synthesis for non-stream callers
    system_msg, user_msg = _build_synthesis_prompt(final_state)
    llm_plain = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.4,
    )
    response = await llm_plain.ainvoke([system_msg, user_msg])
    answer = response.content.strip()
    sources = _extract_sources(final_state)

    store_cache(query, answer, sources)

    return {
        "response": answer,
        "sources": sources,
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
) -> AsyncIterator[dict]:
    """
    Streaming pipeline. Yields dicts:
      {"type": "meta",  "sources": [...], "from_cache": bool, ...}
      {"type": "token", "content": "word"}
      {"type": "done",  "full_response": "..."}
    """

    # ── Cache hit → stream the cached answer token-by-token ──────────────────
    cached = check_cache(query)
    if cached:
        yield {
            "type": "meta",
            "sources": cached["sources"],
            "from_cache": True,
            "cache_score": cached["cache_score"],
            "cached_query": cached["cached_query"],
            "retrieval_strategy": "cache",
        }
        # stream cached answer word by word so UI feels consistent
        words = cached["answer"].split(" ")
        for i, word in enumerate(words):
            yield {"type": "token", "content": word + ("" if i == len(words) - 1 else " ")}
        yield {"type": "done", "full_response": cached["answer"]}
        return

    # ── Run agents up to retrieval ────────────────────────────────────────────
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
    sources = _extract_sources(final_state)

    # ── Emit metadata first so client can show sources immediately ────────────
    yield {
        "type": "meta",
        "sources": sources,
        "from_cache": False,
        "retrieval_strategy": final_state.get("retrieval_strategy", "hybrid"),
        "rewritten_query": final_state.get("rewritten_query"),
    }

    # ── Stream only the final LLM synthesis ──────────────────────────────────
    system_msg, user_msg = _build_synthesis_prompt(final_state)
    full_response = []

    async for chunk in llm_stream.astream([system_msg, user_msg]):
        token = chunk.content
        if token:
            full_response.append(token)
            yield {"type": "token", "content": token}

    answer = "".join(full_response)

    # ── Cache + done ──────────────────────────────────────────────────────────
    store_cache(query, answer, sources)
    yield {"type": "done", "full_response": answer}