"""Builds LLM context from retrieved documents."""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

from app.config.settings import get_settings
from app.retrievers.faiss_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Manages context assembly under token limits."""

    def __init__(self, max_tokens: int | None = None) -> None:
        settings = get_settings()
        self.max_tokens = max_tokens or settings.max_context_tokens
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def _count_tokens(self, text: str) -> int:
        if self._encoder:
            return len(self._encoder.encode(text))
        return len(text) // 4  # Fallback rough approximation

    def _format_chunk(self, doc: RetrievedDocument, index: int) -> str:
        meta = doc.metadata
        header = (
            f"[Document {index + 1}]\n"
            f"[Source] {meta.get('source', 'unknown')}\n"
            f"[Page] {meta.get('page', 0)}\n"
            f"[Section] {meta.get('section', 'General')}\n"
            f"[Heading] {meta.get('heading', '')}\n"
            f"[Relevance Score] {doc.score:.4f}\n"
            f"[Content]\n"
        )
        return f"{header}{doc.content}\n"

    def build(self, documents: list[RetrievedDocument]) -> tuple[str, list[dict[str, Any]]]:
        """Construct context text and track individual chunk metadata."""
        if not documents:
            return "", []

        parts: list[str] = []
        chunk_meta: list[dict[str, Any]] = []
        total_tokens = 0

        for idx, doc in enumerate(documents):
            formatted = self._format_chunk(doc, idx)
            tokens = self._count_tokens(formatted)

            # Prevent exceeding token budget
            if total_tokens + tokens > self.max_tokens:
                logger.warning("Token budget exceeded during context build. Dropping trailing documents.")
                break

            parts.append(formatted)
            chunk_meta.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "section": doc.metadata.get("section", "General"),
                "heading": doc.metadata.get("heading", ""),
                "score": doc.score,
                "content_preview": doc.content[:150],
            })
            total_tokens += tokens

        context = "\n---\n".join(parts)
        logger.info("Context built with %d chunks, total tokens: %d", len(parts), total_tokens)
        return context, chunk_meta
