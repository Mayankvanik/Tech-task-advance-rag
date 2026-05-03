import os
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.db.sqlite_store import (
    init_db, create_session, get_session,
    add_message, get_messages, get_message_count,
    list_user_sessions,
)
from app.db.vector_store import get_vector_store
from app.ingestion.document_processor import parse_document
from app.agents.graph import run_pipeline

from dotenv import load_dotenv
load_dotenv() 

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_vector_store()  # warm up connection
    yield


app = FastAPI(
    title="Conversational RAG API",
    description="Multi-user RAG system with LangGraph orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    query: str


class ChatResponse(BaseModel):
    response: str
    sources: list[dict]
    session_id: str
    rewritten_query: str | None = None
    retrieval_strategy: str | None = None


# ── Document Ingestion ────────────────────────────────────────────────────────

@app.post("/documents/upload", summary="Upload and process a document")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".html", ".htm", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = await parse_document(tmp_path)
        # Override source to original filename
        for c in chunks:
            c["metadata"]["source"] = file.filename

        store = get_vector_store()
        count = store.add_documents(chunks)
        return {"status": "success", "filename": file.filename, "chunks_indexed": count}
    finally:
        os.unlink(tmp_path)


# ── Session Management ────────────────────────────────────────────────────────

@app.post("/sessions", summary="Start a new conversation session")
async def new_session(body: SessionCreate):
    session_id = create_session(body.user_id)
    return {"session_id": session_id, "user_id": body.user_id}


@app.get("/sessions/{session_id}", summary="Get session info")
async def get_session_info(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.get("/users/{user_id}/sessions", summary="List all sessions for a user")
async def user_sessions(user_id: str):
    return list_user_sessions(user_id)


@app.get("/sessions/{session_id}/history", summary="Get chat history")
async def chat_history(session_id: str, limit: int = 50):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "messages": get_messages(session_id, limit=limit),
        "summary": session.get("summary"),
    }


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, summary="Send a message")
async def chat(body: ChatRequest):
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if session["user_id"] != body.user_id:
        raise HTTPException(403, "User does not own this session")

    # Load history
    history = get_messages(body.session_id, limit=settings.max_context_messages)
    summary = session.get("summary")

    # Run LangGraph pipeline
    result = await run_pipeline(
        session_id=body.session_id,
        user_id=body.user_id,
        query=body.query,
        chat_history=history,
        conversation_summary=summary,
    )

    # Persist messages
    add_message(body.session_id, "user", body.query)
    add_message(body.session_id, "assistant", result["response"])

    return ChatResponse(
        response=result["response"],
        sources=result["sources"],
        session_id=body.session_id,
        rewritten_query=result.get("rewritten_query"),
        retrieval_strategy=result.get("retrieval_strategy"),
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}
