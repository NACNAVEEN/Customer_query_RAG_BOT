"""FAISS vector store retriever."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config.settings import get_settings
from app.embeddings.embedding_service import get_embedding_service
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A retrieved document with score and metadata."""

    content: str
    metadata: dict[str, Any]
    score: float
    source: str = "faiss"

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "source": self.source,
        }


class FAISSRetriever:
    """FAISS-based dense vector retriever."""

    INDEX_FILE = "index.faiss"
    DOCS_FILE = "documents.pkl"
    META_FILE = "metadata.json"

    def __init__(self, store_dir: Path | None = None) -> None:
        settings = get_settings()
        self.store_dir = store_dir or settings.vector_store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_service = get_embedding_service()
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        index_path = self.store_dir / self.INDEX_FILE
        docs_path = self.store_dir / self.DOCS_FILE

        if index_path.exists() and docs_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                with open(docs_path, "rb") as f:
                    data = pickle.load(f)
                    self._documents = data["documents"]
                    self._metadatas = data["metadatas"]
                logger.info("Loaded FAISS index with %d documents", len(self._documents))
            except Exception as exc:
                logger.warning("Failed to load FAISS index: %s", exc)
                self._index = None
                self._documents = []
                self._metadatas = []

    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the FAISS index."""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)
        vectors = np.array(embeddings, dtype=np.float32)

        if self._index is None:
            dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dimension)

        self._index.add(vectors)
        self._documents.extend(texts)
        self._metadatas.extend(metadatas)
        self._persist()
        logger.info("Added %d documents to FAISS index", len(chunks))
        return len(chunks)

    def _persist(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.store_dir / self.INDEX_FILE))
        with open(self.store_dir / self.DOCS_FILE, "wb") as f:
            pickle.dump(
                {"documents": self._documents, "metadatas": self._metadatas},
                f,
            )
        with open(self.store_dir / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump({"count": len(self._documents)}, f)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        """Retrieve top-k documents by vector similarity."""
        settings = get_settings()
        k = top_k or settings.vector_top_k

        if self._index is None or not self._documents:
            logger.warning("FAISS index is empty")
            return []

        query_vec = np.array(
            [self.embedding_service.embed_query(query)],
            dtype=np.float32,
        )
        k = min(k, len(self._documents))
        scores, indices = self._index.search(query_vec, k)

        results: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                RetrievedDocument(
                    content=self._documents[idx],
                    metadata=self._metadatas[idx],
                    score=float(score),
                    source="faiss",
                )
            )
        return results
