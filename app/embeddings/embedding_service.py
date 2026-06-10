"""Embedding generation using BAAI/bge-m3."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton-style embedding service wrapping sentence-transformers."""

    _instance: EmbeddingService | None = None

    def __new__(cls) -> EmbeddingService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        settings = get_settings()
        self.model_name = settings.embedding_model
        self.device = settings.embedding_device
        logger.info("Loading embedding model %s on %s", self.model_name, self.device)

        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            raise RuntimeError(f"Failed to load embedding model: {self.model_name}") from exc

        self._initialized = True

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple document strings."""
        if not texts:
            return []
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    def cosine_similarity(self, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return cached embedding service instance."""
    return EmbeddingService()
