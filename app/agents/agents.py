"""
LangGraph Agents — each agent is a pure function: state -> partial state update.
Agents:
  1. query_understanding  — detect intent, needs_rewrite
  2. query_rewriting      — reformulate ambiguous query
  3. retrieval_router     — pick strategy: semantic / keyword / hybrid
  4. retriever            — fetch docs from Qdrant
  5. context_synthesis    — build answer from docs + history
  6. conversation_summary — summarize long conversations
  7. memory_manager       — decide what to persist
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_settings
from app.core.state import ConversationState
from app.db.vector_store import get_vector_store
from app.db.sqlite_store import update_session_summary

settings = get_settings()

llm = ChatOpenAI(
    model=settings.llm_model,
    openai_api_key=settings.openai_api_key,
    temperature=0.1,
)

llm_creative = ChatOpenAI(
    model=settings.llm_model,
    openai_api_key=settings.openai_api_key,
    temperature=0.4,
)


def _history_text(chat_history: list[dict], limit: int = 6) -> str:
    recent = chat_history[-limit:] if len(chat_history) > limit else chat_history
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)


# ── Agent 1: Query Understanding ──────────────────────────────────────────────

def query_understanding_agent(state: ConversationState) -> dict:
    """Determine if query needs context from history or rewriting."""
    history = _history_text(state["chat_history"])

    if not history.strip():
        # No history → direct query
        return {"needs_rewrite": False, "retrieval_strategy": "hybrid"}

    prompt = f"""Analyze the user query in the context of the conversation.

Conversation History:
{history}

Current Query: {state['query']}

Answer in this exact format:
NEEDS_REWRITE: yes/no
REASON: one sentence
STRATEGY: semantic/keyword/hybrid"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    
    needs_rewrite = "NEEDS_REWRITE: yes" in text.lower()
    strategy = "hybrid"
    if "STRATEGY: semantic" in text:
        strategy = "semantic"
    elif "STRATEGY: keyword" in text:
        strategy = "keyword"
    
    return {"needs_rewrite": needs_rewrite, "retrieval_strategy": strategy}


# ── Agent 2: Query Rewriting ──────────────────────────────────────────────────

def query_rewriting_agent(state: ConversationState) -> dict:
    """Reformulate query using conversation context."""
    if not state.get("needs_rewrite", False):
        return {"rewritten_query": state["query"]}

    history = _history_text(state["chat_history"])
    prompt = f"""Rewrite the user query to be self-contained using conversation history.
Keep it concise and search-optimized.

Conversation History:
{history}

Original Query: {state['query']}

Rewritten Query (only output the query, nothing else):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"rewritten_query": response.content.strip()}


# ── Agent 3: Retrieval Router ─────────────────────────────────────────────────

def retrieval_router_agent(state: ConversationState) -> dict:
    """Route to the correct retrieval strategy (already determined in understanding)."""
    # Strategy is set in query_understanding; this node just confirms/logs
    strategy = state.get("retrieval_strategy", "hybrid")
    return {"retrieval_strategy": strategy}


# ── Agent 4: Retriever ────────────────────────────────────────────────────────

def retriever_agent(state: ConversationState) -> dict:
    """Fetch documents from Qdrant using the chosen strategy."""
    store = get_vector_store()
    query = state.get("rewritten_query") or state["query"]
    strategy = state.get("retrieval_strategy", "hybrid")

    if strategy == "semantic":
        docs = store.semantic_search(query, top_k=5)
    elif strategy == "keyword":
        docs = store.keyword_search(query, top_k=5)
    else:
        docs = store.hybrid_search(query, top_k=5)

    return {"retrieved_docs": docs}


# ── Agent 5: Context Synthesis ────────────────────────────────────────────────

def context_synthesis_agent(state: ConversationState) -> dict:
    """Generate final answer from docs + conversation history."""
    docs = state.get("retrieved_docs", [])
    history = _history_text(state["chat_history"])
    summary = state.get("conversation_summary", "")

    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Source {i}]\n{doc['text']}")
        sources.append({
            "index": i,
            "source": doc["metadata"].get("source", "unknown"),
            "section": doc["metadata"].get("section", ""),
            "score": round(doc.get("score", 0), 3),
        })

    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

    memory_context = ""
    if summary:
        memory_context = f"\nConversation Summary:\n{summary}\n"

    system = """You are a helpful technical assistant. Answer based on the provided context.
Be precise and cite sources using [Source N] notation. If information is not in context, say so."""

    user_prompt = f"""{memory_context}
Recent Conversation:
{history}

Retrieved Context:
{context}

User Question: {state['query']}

Answer:"""

    response = llm_creative.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ])

    return {"response": response.content.strip(), "sources": sources}


# ── Agent 6: Conversation Summarization ───────────────────────────────────────

def conversation_summary_agent(state: ConversationState) -> dict:
    """Summarize conversation when it gets too long."""
    history = state["chat_history"]
    if len(history) < settings.summary_threshold:
        return {}

    history_text = _history_text(history, limit=len(history))
    prompt = f"""Summarize the key points of this conversation concisely (max 200 words).
Focus on: topics discussed, decisions made, user's main questions and answers given.

Conversation:
{history_text}

Summary:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    summary = response.content.strip()

    # Persist to SQLite
    update_session_summary(state["session_id"], summary)

    return {"conversation_summary": summary}


# ── Agent 7: Memory Manager ───────────────────────────────────────────────────

def memory_manager_agent(state: ConversationState) -> dict:
    """Decide if this exchange is worth long-term summarization."""
    # Simple heuristic: if we have a good response and sources, it's valuable
    # The actual persistence is handled by the API layer (add_message)
    # This agent can flag important exchanges for future use
    return {}
