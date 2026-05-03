from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
)
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
import uuid
from app.core.config import get_settings

settings = get_settings()


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(self, chunks: list[dict]) -> int:
        """Add document chunks. Each chunk: {text, metadata}"""
        texts = [c["text"] for c in chunks]
        vectors = self.embeddings.embed_documents(texts)
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    **chunk.get("metadata", {}),
                },
            ))
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def _build_filter(self, filters: dict | None) -> Filter | None:
        """Build Qdrant Filter from a plain dict.
        Supports exact match and range filters.
        Range format: {"field": {"gte": 1, "lte": 10}}
        """
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                conditions.append(FieldCondition(
                    key=key,
                    range=Range(
                        gte=value.get("gte"),
                        lte=value.get("lte"),
                        gt=value.get("gt"),
                        lt=value.get("lt"),
                    ),
                ))
            else:
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                ))
        return Filter(must=conditions)

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Pure semantic (dense) search with optional metadata filters."""
        query_vector = self.embeddings.embed_query(query)

        # v1.17.1: query_points returns QueryResponse with .points: list[ScoredPoint]
        # ScoredPoint fields: id, version, score, payload, vector, shard_key, order_value
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=self._build_filter(filters),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": point.payload,
            }
            for point in response.points
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Hybrid search: semantic + BM25 re-ranking."""
        candidates = self.semantic_search(query, top_k=top_k * 3, filters=filters)
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Combine: 0.6 semantic + 0.4 BM25 (normalized)
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        for i, c in enumerate(candidates):
            c["score"] = 0.6 * c["score"] + 0.4 * float(bm25_norm[i])

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Keyword-focused search: BM25 re-rank over a broad semantic fetch."""
        return self.hybrid_search(query, top_k=top_k, filters=filters)


# Singleton
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store