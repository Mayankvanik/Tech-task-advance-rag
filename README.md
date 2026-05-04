# Conversational RAG System

Multi-user RAG system with LangGraph orchestration, streaming, persistent memory, and hybrid search.

---

## Architecture

```text
User → Streamlit UI → FastAPI
                        │
                        ▼
                 Semantic Cache
           (Matches previous question?)
                 /              \
          Yes (Hit)             No (Miss)
             │                      │
             ▼                      ▼
       Return Instantly     LangGraph Pipeline
      (Bypass AI Agents)            │
                                    ▼
                             Qdrant + SQLite
```

### LangGraph Agent Flow

```text
                 START
                   │
            ┌──────┴──────┐
            │             │
       understand      prefetch (Hybrid Search)
            │             │
            └──────┬──────┘
                   │
                 decide
                   │
            should_rewrite?
                   │
            ┌──────┴──────┐
          (no)          (yes)
            │             │
            │          rewrite
            │             │
            │           route
            │             │
            │         retrieve (Qdrant: semantic / keyword / hybrid+BM25)
            │             │
            └──────┬──────┘
                   │
               synthesize (GPT-4o-mini)
                   │
           should_summarize?
                   │
            ┌──────┴──────┐
          (no)          (yes)
            │             │
            │         summarize (SQLite persist)
            │             │
            └──────┬──────┘
                   │
                 memory
                   │
                  END
```

### Memory & State Management

The system uses a highly optimized **Summary Buffer with Sliding Window** approach combined with **User Preference Extraction** to manage context:

- **Conversation State**: The LangGraph pipeline orchestrates data using a typed `ConversationState` that holds the active `chat_history`, the long-term `conversation_summary`, and persistent `user_preferences`.
- **Sliding Window**: To keep LLM token limits strictly under control, the FastAPI layer only loads the `N` most recent messages into the active `chat_history`.
- **Summary Buffer**: When a conversation gets too long (exceeds `summary_threshold`), the `summarize` agent condenses the older history into a dense text summary. This summary is stored in SQLite and injected into the LLM prompt, ensuring no old context is forgotten even when it slides out of the active window.
- **Auto-Extracted User Preferences**: During the summarization phase, the system uses an LLM call to automatically extract any explicitly or implicitly stated user preferences (e.g., tone, formatting, detail level). These are persisted in SQLite under the user's ID (not just the session ID), meaning the AI remembers how the user likes their answers formatted across *all* of their separate chat sessions!

### Chunking and Retrieval

**Document Ingestion & Chunking:**
- Documents are parsed per format: **PDFs** are converted to clean Markdown via `pymupdf4llm` (preserving headers, tables, and code blocks); **HTML** has noise tags (`script`, `style`, `nav`, `footer`) stripped via BeautifulSoup; **Markdown/TXT** are read directly.
- Text is split using LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=800` and `chunk_overlap=100`, using natural separators (`\n\n`, `\n`, `.`, ` `) to avoid breaking mid-sentence.
- Each chunk carries rich metadata: `source`, `doc_type`, `version` (auto-detected), `section` (tracked by walking through markdown headers), and a `has_code` flag.

**Retrieval Strategies (selected per query by `query_understanding_agent`):**
- **Semantic:** Pure dense vector search — query is embedded with `text-embedding-3-small` (1536d) and retrieved from Qdrant using Cosine Similarity.
- **Keyword:** Runs via the hybrid path but weights BM25 more heavily, ideal for exact technical terms, IDs, or error codes.
- **Hybrid (default):** Fetches `top_k × 3` semantic candidates from Qdrant, then re-ranks them using **BM25 Okapi** in a weighted combination:
  ```
  final_score = 0.6 × semantic_score + 0.4 × bm25_score_normalized
  ```
  The top `top_k` results are returned. This balances conceptual recall (dense) with precise keyword matching (sparse).

### LLM

The system uses a dual-LLM strategy to balance cost, latency, and reasoning capability:
- **Fast & Cheap (Reasoning/Routing)**: A lighter model is used for intent detection, query understanding, and routing tasks where speed is critical and the logic is straightforward.
- **Accurate & Creative (Synthesis)**: A more powerful reasoning model is used for the final context synthesis, ensuring high-quality, accurate, and well-cited answers based on the retrieved data.

### Observability

- **LangSmith Integration**: The system is fully instrumented with LangSmith for real-time observability, trace logging, and agent performance monitoring. By providing the `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` in the `.env` file, every agent interaction and LangGraph trace is recorded for deep debugging and performance evaluation.

### Vector DB Collections

The system utilizes two distinct collections within Qdrant to handle different aspects of the RAG pipeline:
- **`documents`**: Stores the processed chunks of uploaded documents (PDFs, Markdown, etc.) for retrieval during the synthesis phase.
- **`semantic_cache`**: Stores previously asked user questions. When a question is asked, its embedding is compared against this collection. If a match is found, the pre-stored answer and sources (kept in the point's metadata) are returned instantly, bypassing the entire generation pipeline.


### Tech Stack

| Layer | Technology |
|-------|------------|
| Document Parsing | LlamaParser (PDF, MD, HTML, TXT) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536d) |
| Vector DB | Qdrant (Docker) |
| Search | Hybrid: Dense (cosine) + BM25 re-rank |
| Orchestration | LangGraph StateGraph |
| Chat Storage | SQLite (aiosqlite) |
| API | FastAPI |
| UI | Streamlit |

---

## Setup

### 1. Clone and install

```bash
cd rag_system
uv init
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your API_KEYS 
```

### 3. Start Qdrant (Docker)

```bash
cd docker
docker-compose up -d
# Qdrant UI: http://localhost:6333/dashboard
```

### 4. Run the API

```bash
cd ..  # back to project root
uv run run_api.py
# API docs: http://localhost:8000/docs
```

### 5. Run Streamlit UI

```bash
streamlit run streamlit_app/app.py
# UI: http://localhost:8501
```

---

## API Endpoints

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload & index PDF/MD/HTML/TXT |

### Sessions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/sessions` | Create new session → returns `session_id` |
| GET | `/sessions/{id}` | Get session info |
| GET | `/sessions/{id}/history` | Get chat messages |
| GET | `/users/{user_id}/sessions` | List user's sessions |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message, get RAG response |

---

## Example Usage

```python
import requests

BASE = "http://localhost:8000"

# 1. Create session
session = requests.post(f"{BASE}/sessions", json={"user_id": "alice"}).json()
sid = session["session_id"]

# 2. Upload doc
with open("docs/manual.pdf", "rb") as f:
    requests.post(f"{BASE}/documents/upload", files={"file": f})

# 3. Chat
resp = requests.post(f"{BASE}/chat", json={
    "session_id": sid,
    "user_id": "alice",
    "query": "What are the main features?"
}).json()

print(resp["response"])
print(resp["sources"])
```

---

## Key Design Decisions

- **Semantic Caching**: The system first checks new queries against a database of previously asked questions. If it finds an exact or *semantically similar* match (same meaning, different words), it immediately returns the cached answer. This completely bypasses the LangGraph agents and LLMs, making responses instantaneous and saving API costs.
- **Parallel execution**: Intent detection (`understand`) and initial retrieval (`prefetch`) run concurrently from the start, minimizing latency before deciding whether to rewrite the query.
- **Hybrid search** = Dense cosine (Qdrant) + BM25 re-rank. Balances semantic and keyword matching.
- **Query rewriting** only triggers when history exists and the query is ambiguous — saves LLM calls.
- **Summarization** triggers at `SUMMARY_THRESHOLD` (default 20) messages to keep context window manageable.
- **Multi-user isolation**: Sessions are keyed by `session_id`, users can only access their own sessions.
- **SQLite** stores all messages and summaries — no external DB needed for development.

---

## Project Structure

```
rag_system/
├── app/
│   ├── agents/
│   │   ├── agents.py      # All 9 specialized agents
│   │   └── graph.py       # LangGraph wiring + run_pipeline()
│   ├── api/
│   │   └── main.py        # FastAPI endpoints
│   ├── core/
│   │   ├── config.py      # Settings (pydantic-settings)
│   │   ├── prompt.py      # System and user prompts for agents
│   │   └── state.py       # LangGraph ConversationState
│   ├── db/
│   │   ├── semantic_cache.py # Semantic caching logic
│   │   ├── sqlite_store.py  # Chat history + sessions
│   │   └── vector_store.py  # Qdrant + BM25 hybrid search
│   └── ingestion/
│       └── document_processor.py  # LlamaParser + chunking
├── streamlit_app/
│   └── app.py             # Streamlit UI
├── docker/
│   └── docker-compose.yml # Qdrant service
├── .env.example
├── requirements.txt
└── run_api.py
```
