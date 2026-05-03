QUERY_UNDERSTANDING_PROMPT = """
Analyze the user's latest query to determine if it lacks context that was established earlier in the conversation, and select the optimal retrieval strategy.

Guidelines:
- needs_rewrite: Set to true if the query uses pronouns (it, they, this) or refers to topics implicitly based on previous turns.
- retrieval_strategy:
  * "semantic": Best for conceptual questions, meanings, or general knowledge.
  * "keyword": Best for exact names, specific IDs, error codes, or technical terms.
  * "hybrid": Best for queries combining both concepts and specific terms (default).

Conversation History:
{history}

Current Query: {query}

Respond ONLY in valid JSON format:
{{
  "needs_rewrite": true/false,
  "reason": "Brief explanation of your decision",
  "retrieval_strategy": "semantic" | "keyword" | "hybrid"
}}
"""

QUERY_REWRITING_PROMPT = """
Reformulate the user's latest query so that it is fully self-contained and can be understood without the conversation history.
Resolve any pronouns or implicit references by substituting them with the specific subjects discussed earlier.
Ensure the rewritten query is concise, highly relevant, and optimized for searching a vector database. Do not add conversational filler.

Conversation History:
{history}

Original Query: {query}

Rewritten Query (output ONLY the final query text):"""

CONTEXT_SYNTHESIS_SYSTEM_PROMPT = """You are an expert technical assistant. Your goal is to provide accurate, helpful, and clear answers based strictly on the provided context.
- Base your answer entirely on the provided document context and relevant conversation memory.
- Always include inline citations using the exact [Source N] format when referencing information.
- If the answer cannot be confidently derived from the provided context, clearly state that you do not have enough information rather than guessing.
- Keep your formatting clean using Markdown where appropriate."""

CONTEXT_SYNTHESIS_USER_PROMPT = """{memory_context}

Recent Conversation History:
{history}

Retrieved Context Documents:
{context}

User Question: {query}

Synthesized Answer:"""

CONVERSATION_SUMMARY_PROMPT = """
Create a concise summary of the conversation below to serve as long-term memory for an AI assistant.
Keep it under 200 words. Focus strictly on:
- The core topics and concepts discussed.
- Key questions asked by the user and the primary conclusions or answers provided.
- Any decisions made or user preferences stated.

Conversation:
{history}

Summary:"""