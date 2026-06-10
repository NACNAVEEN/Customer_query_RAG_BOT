"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Central configuration for InstaParkAI RAG chatbot."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "InstaParkAI RAG Chatbot"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # Paths
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    vector_store_dir: Path = Field(default=PROJECT_ROOT / "data" / "vector_store")
    bm25_store_dir: Path = Field(default=PROJECT_ROOT / "data" / "bm25_store")
    evaluation_reports_dir: Path = Field(default=PROJECT_ROOT / "data" / "evaluation")

    # Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings & Reranker
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_device: str = "cpu"

    # Chunking
    chunk_size: int = 700
    chunk_overlap: int = 100

    # Retrieval
    vector_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_top_k: int = 10
    rerank_top_k: int = 3
    similarity_threshold: float = 0.50
    rrf_k: int = 60

    # Semantic cache
    cache_similarity_threshold: float = 0.95
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    gptcache_enabled: bool = True

    # Context
    max_context_tokens: int = 6000

    # Memory
    memory_max_interactions: int = 5

    # LangSmith
    langsmith_enabled: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "instapark-rag"
    langchain_tracing_v2: bool = False

    # RAGAS
    ragas_enabled: bool = True

    @field_validator(
        "data_dir",
        "uploads_dir",
        "vector_store_dir",
        "bm25_store_dir",
        "evaluation_reports_dir",
        mode="after",
    )
    @classmethod
    def ensure_path(cls, value: Path) -> Path:
        return Path(value)

    def ensure_directories(self) -> None:
        """Create required data directories."""
        for directory in (
            self.data_dir,
            self.uploads_dir,
            self.vector_store_dir,
            self.bm25_store_dir,
            self.evaluation_reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured data directories exist under %s", self.data_dir)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
