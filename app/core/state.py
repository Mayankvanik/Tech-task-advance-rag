from typing import TypedDict, Annotated, Optional
from operator import add


class ConversationState(TypedDict):
    # Input
    session_id: str
    user_id: str
    query: str

    # Query processing
    rewritten_query: Optional[str]
    needs_rewrite: bool
    retrieval_strategy: str  # "semantic", "keyword", "hybrid"

    # Retrieval
    prefetched_docs: list[dict]   # hybrid-search on original query (parallel path)
    retrieved_docs: list[dict]    # final docs used by synthesizer

    # Memory & history
    chat_history: list[dict]
    conversation_summary: Optional[str]

    # Output
    response: str
    sources: list[dict]

    # Control
    messages: Annotated[list, add]  # LangGraph internal messages
