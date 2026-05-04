import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from app.core.config import get_settings


settings = get_settings()


def get_connection() -> sqlite3.Connection:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- User preferences: keyed by user_id, persists across all sessions
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        """)


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(user_id: str) -> str:
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, NULL)",
            (session_id, user_id, now, now),
        )
    return session_id


def get_session(session_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def update_session_summary(session_id: str, summary: str):
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
            (summary, datetime.utcnow().isoformat(), session_id),
        )


def list_user_sessions(user_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── User Preferences ─────────────────────────────────────────────────────────

def get_user_preferences(user_id: str) -> dict:
    """Return stored preferences for a user, or empty dict if none."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT preferences FROM user_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row:
        try:
            return json.loads(row["preferences"])
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


def upsert_user_preferences(user_id: str, new_prefs: dict):
    """Merge new_prefs into existing preferences for the user (upsert)."""
    existing = get_user_preferences(user_id)
    merged = {**existing, **new_prefs}   # new values win on conflict
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_preferences (user_id, preferences, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                preferences = excluded.preferences,
                updated_at  = excluded.updated_at
            """,
            (user_id, json.dumps(merged), now),
        )


# ── Messages ──────────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.utcnow().isoformat()),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )


def get_messages(session_id: str, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return list(reversed([dict(r) for r in rows]))


def get_message_count(session_id: str) -> int:
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
