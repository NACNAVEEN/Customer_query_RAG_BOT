"""BM25 keyword retriever."""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.config.settings import get_settings
from app.ingestion.chunker import DocumentChunk
from app.retrievers.faiss_retriever import RetrievedDocument

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Simple whitespace/punctuation tokenizer."""
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class BM25RetrieverWrapper:
    """BM25-based sparse retriever with persistence."""

    STORE_FILE = "bm25_index.pkl"

    def __init__(self, store_dir: Path | None = None) -> None:
        settings = get_settings()
        self.store_dir = store_dir or settings.bm25_store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._bm25: BM25Okapi | None = None
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        store_path = self.store_dir / self.STORE_FILE
        if store_path.exists():
            try:
                with open(store_path, "rb") as f:
                    data = pickle.load(f)
                    self._documents = data["documents"]
                    self._metadatas = data["metadatas"]
                    tokenized = data["tokenized"]
                    self._bm25 = BM25Okapi(tokenized)
                logger.info("Loaded BM25 index with %d documents", len(self._documents))
            except Exception as exc:
                logger.warning("Failed to load BM25 index: %s", exc)

    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        """Add chunks to BM25 index."""
        if not chunks:
            return 0

        new_docs = [c.content for c in chunks]
        new_meta = [c.metadata for c in chunks]
        self._documents.extend(new_docs)
        self._metadatas.extend(new_meta)

        tokenized_corpus = [tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._persist(tokenized_corpus)
        logger.info("Added %d documents to BM25 index", len(chunks))
        return len(chunks)

    def _persist(self, tokenized_corpus: list[list[str]]) -> None:
        with open(self.store_dir / self.STORE_FILE, "wb") as f:
            pickle.dump(
                {
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "tokenized": tokenized_corpus,
                },
                f,
            )

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        """Retrieve top-k documents using BM25."""
        settings = get_settings()
        k = top_k or settings.bm25_top_k

        if self._bm25 is None or not self._documents:
            logger.warning("BM25 index is empty")
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results: list[RetrievedDocument] = []
        max_score = max(scores) if len(scores) > 0 else 1.0
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            normalized = float(scores[idx] / max_score) if max_score > 0 else 0.0
            results.append(
                RetrievedDocument(
                    content=self._documents[idx],
                    metadata=self._metadatas[idx],
                    score=normalized,
                    source="bm25",
                )
            )
        return results
