# Conversational RAG System

Multi-user RAG system with LangGraph orchestration, persistent memory, and hybrid search.

---

## Architecture

```text
User → Streamlit UI → FastAPI → Semantic Cache (Hit? → Return) → LangGraph Pipeline → Qdrant + SQLite
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
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY and LLAMA_CLOUD_API_KEY
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
python run_api.py
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

- **Semantic Caching**: Previous Q&A interactions are cached. Before invoking the LangGraph pipeline, the system checks for semantically similar queries. If found, it returns the cached response immediately, bypassing the LLM and saving time/cost.
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
