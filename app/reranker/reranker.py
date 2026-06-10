"""Cross-encoder reranker using BAAI/bge-reranker-base."""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config.settings import get_settings
from app.retrievers.faiss_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class RerankerService:
    """Rerank retrieved documents using a cross-encoder model."""

    _instance: RerankerService | None = None

    def __new__(cls) -> RerankerService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        settings = get_settings()
        self.model_name = settings.reranker_model
        logger.info("Loading reranker model %s", self.model_name)

        try:
            self._model = CrossEncoder(self.model_name, max_length=512)
        except Exception as exc:
            logger.exception("Failed to load reranker model")
            raise RuntimeError(f"Failed to load reranker: {self.model_name}") from exc

        self._initialized = True

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank documents and return top-k by relevance score."""
        settings = get_settings()
        k = top_k or settings.rerank_top_k

        if not documents:
            return []

        pairs = [[query, doc.content] for doc in documents]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.exception("Reranking failed, returning original order")
            return documents[:k]

        scored_docs: list[tuple[float, RetrievedDocument]] = []
        for score, doc in zip(scores, documents):
            scored_docs.append(
                (
                    float(score),
                    RetrievedDocument(
                        content=doc.content,
                        metadata=doc.metadata,
                        score=float(score),
                        source="reranker",
                    ),
                )
            )

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored_docs[:k]]
        logger.info("Reranked %d docs, returning top %d", len(documents), len(results))
        return results


@lru_cache
def get_reranker() -> RerankerService:
    return RerankerService()
