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

import logging
logger = logging.getLogger(__name__)
from app.db.vector_store import get_vector_store
from app.core.prompt import (
    QUERY_UNDERSTANDING_PROMPT,
    QUERY_REWRITING_PROMPT,
    CONTEXT_SYNTHESIS_SYSTEM_PROMPT,
    CONTEXT_SYNTHESIS_USER_PROMPT,
    CONVERSATION_SUMMARY_PROMPT,
    PREFERENCE_EXTRACTION_PROMPT,
    build_preferences_block,
)
from app.db.sqlite_store import update_session_summary, upsert_user_preferences
import json
import re

settings = get_settings()

llm = ChatOpenAI(
    model=settings.light_llm_model,
    openai_api_key=settings.openai_api_key,
    temperature=0.1,
    max_tokens=6000,
)

llm_creative = ChatOpenAI(
    model=settings.llm_model,
    openai_api_key=settings.openai_api_key,
    temperature=0.4,
    max_tokens=6000,
)


def _history_text(chat_history: list[dict], limit: int = 6) -> str:
    recent = chat_history[-limit:] if len(chat_history) > limit else chat_history
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)


def extract_json(text: str) -> str:
    """Extract JSON from markdown/code block if present."""
    # Remove ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return text.strip()

# ── Agent 1: Query Understanding ──────────────────────────────────────────────



def query_understanding_agent(state: ConversationState) -> dict:
    """Determine if query needs context from history or rewriting."""
    
    history = _history_text(state["chat_history"])
    logger.debug(f"In query understanding agent, history is: {history}")
    if not history.strip():
        logger.info("No history available.")
        return {
            "needs_rewrite": False,
            "retrieval_strategy": "hybrid",
            "reason": "No history available"
        }

    prompt = QUERY_UNDERSTANDING_PROMPT.format(history=history, query=state['query'])

    response = llm.invoke([HumanMessage(content=prompt)])
    logger.debug(f"query understanding response: {response.content}")
    try:
        clean_text = extract_json(response.content)

        parsed = json.loads(clean_text)
        logger.debug(f"Parsed query understanding output: {parsed}")

        result = {
            "needs_rewrite": parsed.get("needs_rewrite", None),
            "retrieval_strategy": parsed.get("retrieval_strategy", "hybrid")
        }

        logger.info(f"Final query understanding result: {result}")
        return result


    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from query understanding agent. Raw response: {response.content}")
        # fallback (important for production reliability)
        return {
            "needs_rewrite": False,
            "retrieval_strategy": "hybrid",
            "reason": "Fallback due to invalid JSON response",
            "raw_output": response.content  # useful for debugging
        }


# ── Agent 2: Query Rewriting ──────────────────────────────────────────────────

def query_rewriting_agent(state: ConversationState) -> dict:
    logger.debug("Entering query_rewriting_agent")
    """Reformulate query using conversation context."""
    if not state.get("needs_rewrite", False):
        return {"rewritten_query": state["query"]}

    history = _history_text(state["chat_history"])
    prompt = QUERY_REWRITING_PROMPT.format(history=history, query=state['query'])

    response = llm.invoke([HumanMessage(content=prompt)])
    logger.debug(f"rewriting response: {response.content}")
    return {"rewritten_query": response.content.strip()}


# ── Agent 3: Prefetch Retriever (parallel with understand) ───────────────────

def prefetch_retriever_agent(state: ConversationState) -> dict:
    """Hybrid search on the original query — runs in parallel with query_understanding.
    Results are stored in `prefetched_docs` so they can be reused if no rewrite is needed.
    """
    store = get_vector_store()
    query = state["query"]
    logger.info(f"Prefetch retriever: hybrid search on original query='{query}'")
    docs = store.hybrid_search(query, top_k=5)
    logger.info(f"Prefetch retriever: fetched {len(docs)} docs")
    return {"prefetched_docs": docs}


# ── Agent 4: Decide — merge parallel branches ─────────────────────────────────

def decide_agent(state: ConversationState) -> dict:
    """Merge node that runs after both `understand` and `prefetch_retrieve` finish.

    - needs_rewrite=False → promote prefetched_docs to retrieved_docs (skip retrieval)
    - needs_rewrite=True  → leave retrieved_docs empty; `rewrite` → `retrieve` will fill it
    """
    needs_rewrite = state.get("needs_rewrite", False)
    if not needs_rewrite:
        prefetched = state.get("prefetched_docs", [])
        logger.info(f"decide_agent: no rewrite needed — using {len(prefetched)} prefetched docs")
        return {"retrieved_docs": prefetched}
    logger.info("decide_agent: rewrite needed — fresh retrieval will follow")
    return {}


# ── Agent 5: Retrieval Router (kept for rewrite path) ─────────────────────────

def retrieval_router_agent(state: ConversationState) -> dict:
    """Route to the correct retrieval strategy (already determined in understanding)."""
    # Strategy is set in query_understanding; this node just confirms/logs
    strategy = state.get("retrieval_strategy", "hybrid")
    return {"retrieval_strategy": strategy}


# ── Agent 6: Retriever (rewrite path only) ───────────────────────────────────

def retriever_agent(state: ConversationState) -> dict:
    """Fetch documents from Qdrant using the chosen strategy.
    Only executed on the rewrite path (needs_rewrite=True).
    """
    store = get_vector_store()
    query = state.get("rewritten_query") or state["query"]
    strategy = state.get("retrieval_strategy", "hybrid")
    logger.info(f"retriever_agent: strategy={strategy}, query='{query}'")

    if strategy == "semantic":
        docs = store.semantic_search(query, top_k=5)
    elif strategy == "keyword":
        docs = store.keyword_search(query, top_k=5)
    else:
        docs = store.hybrid_search(query, top_k=5)

    logger.info(f"retriever_agent: fetched {len(docs)} docs")
    return {"retrieved_docs": docs}


# ── Agent 5: Context Synthesis ────────────────────────────────────────────────

def context_synthesis_agent(state: ConversationState) -> dict:
    """Generate final answer from docs + conversation history + user preferences."""
    docs = state.get("retrieved_docs", [])
    history = _history_text(state["chat_history"])
    summary = state.get("conversation_summary", "")
    user_preferences = state.get("user_preferences") or {}

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

    # Inject user preferences into the system prompt
    prefs_block = build_preferences_block(user_preferences)
    system = CONTEXT_SYNTHESIS_SYSTEM_PROMPT.format(user_preferences_block=prefs_block)

    user_prompt = CONTEXT_SYNTHESIS_USER_PROMPT.format(
        memory_context=memory_context,
        history=history,
        context=context,
        query=state['query']
    )

    response = llm_creative.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ])

    return {"response": response.content.strip(), "sources": sources}


# ── Agent 6: Conversation Summarization ───────────────────────────────────────

def conversation_summary_agent(state: ConversationState) -> dict:
    """Summarize conversation when it gets too long, and extract user preferences."""
    history = state["chat_history"]
    if len(history) < settings.summary_threshold:
        return {}

    history_text = _history_text(history, limit=len(history))

    # ── Summarize ─────────────────────────────────────────────────────────────────
    prompt = CONVERSATION_SUMMARY_PROMPT.format(history=history_text)
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.debug(f"conversation summary response: {response.content}")
    summary = response.content.strip()
    update_session_summary(state["session_id"], summary)

    # ── Extract and persist user preferences ───────────────────────────────────────
    pref_prompt = PREFERENCE_EXTRACTION_PROMPT.format(history=history_text)
    pref_response = llm.invoke([HumanMessage(content=pref_prompt)])
    logger.debug(f"preference extraction response: {pref_response.content}")
    try:
        extracted = json.loads(extract_json(pref_response.content))
        if extracted and isinstance(extracted, dict):
            upsert_user_preferences(state["user_id"], extracted)
            logger.info(f"Saved preferences for user {state['user_id']}: {extracted}")
    except Exception as e:
        logger.warning(f"Could not extract preferences: {e}")

    return {"conversation_summary": summary}


# ── Agent 7: Memory Manager ───────────────────────────────────────────────────

def memory_manager_agent(state: ConversationState) -> dict:
    """Decide if this exchange is worth long-term summarization."""
    # Simple heuristic: if we have a good response and sources, it's valuable
    # The actual persistence is handled by the API layer (add_message)
    # This agent can flag important exchanges for future use
    return {}
