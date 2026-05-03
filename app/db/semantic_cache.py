"""
Semantic Cache — stores Q&A pairs in a dedicated Qdrant collection.

Flow:
  1. On every user query → check_cache(query)
     - embed query, search cache collection by cosine similarity
     - if score >= threshold → return cached answer (skip pipeline)
     - else → return None (run pipeline normally)

  2. After pipeline generates answer → store_cache(query, answer, metadata)
     - embed query, upsert into cache collection with answer + metadata

Usage in graph.py / API layer:
    cached = check_cache(query)
    if cached:
        return cached  # {"answer": ..., "sources": ...}
    result = await run_pipeline(...)
    store_cache(query, result["response"], result["sources"])
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from langchain_openai import OpenAIEmbeddings
from app.core.config import get_settings

settings = get_settings()

CACHE_COLLECTION = "semantic_cache_new"
SIMILARITY_THRESHOLD = 0.92   # tune: higher = stricter match required


class SemanticCache:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if CACHE_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def check(self, query: str) -> dict | None:
        """
        Search cache for a semantically similar question.
        Returns {"answer": str, "sources": list, "cached_query": str} or None.
        """
        vector = self.embeddings.embed_query(query)

        response = self.client.query_points(
            collection_name=CACHE_COLLECTION,
            query=vector,
            limit=1,
            with_payload=True,
        )

        if not response.points:
            return None

        top = response.points[0]
        if top.score < SIMILARITY_THRESHOLD:
            return None

        return {
            "answer": top.payload.get("answer", ""),
            "sources": top.payload.get("sources", []),
            "cached_query": top.payload.get("question", ""),
            "cache_score": round(top.score, 4),
        }

    def store(self, query: str, answer: str, sources: list | None = None):
        """Embed the query and store Q&A pair in cache."""
        vector = self.embeddings.embed_query(query)
        self.client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "question": query,
                        "answer": answer,
                        "sources": sources or [],
                    },
                )
            ],
        )

    def invalidate(self, query: str):
        """Remove a cached entry by exact question match (optional utility)."""
        self.client.delete(
            collection_name=CACHE_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="question", match=MatchValue(value=query))]
            ),
        )

    def clear(self):
        """Wipe entire cache (useful for testing)."""
        self.client.delete_collection(CACHE_COLLECTION)
        self._ensure_collection()


# Singleton
_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


# ── Convenience functions (import these in graph.py / main.py) ────────────────

def check_cache(query: str) -> dict | None:
    """Returns cached result or None."""
    return get_semantic_cache().check(query)


def store_cache(query: str, answer: str, sources: list | None = None):
    """Store a new Q&A pair in the semantic cache."""
    get_semantic_cache().store(query, answer, sources)