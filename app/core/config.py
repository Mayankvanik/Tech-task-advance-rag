from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    llama_cloud_api_key: str

    # OpenAI
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"

    # SQLite
    sqlite_db_path: str = "./data/chat_history.db"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_context_messages: int = 10
    summary_threshold: int = 20

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
