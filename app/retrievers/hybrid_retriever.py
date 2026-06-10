"""Hybrid retriever combining FAISS and BM25 with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from typing import Any

from app.config.settings import get_settings
from app.ingestion.chunker import DocumentChunk
from app.retrievers.bm25_retriever import BM25RetrieverWrapper
from app.retrievers.faiss_retriever import FAISSRetriever, RetrievedDocument

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense and sparse retrieval using RRF."""

    def __init__(
        self,
        faiss_retriever: FAISSRetriever | None = None,
        bm25_retriever: BM25RetrieverWrapper | None = None,
    ) -> None:
        self.faiss = faiss_retriever or FAISSRetriever()
        self.bm25 = bm25_retriever or BM25RetrieverWrapper()

    def index_documents(self, chunks: list[DocumentChunk]) -> dict[str, int]:
        """Index chunks in both FAISS and BM25."""
        faiss_count = self.faiss.add_documents(chunks)
        bm25_count = self.bm25.add_documents(chunks)
        return {"faiss": faiss_count, "bm25": bm25_count}

    @property
    def document_count(self) -> int:
        return max(self.faiss.document_count, self.bm25.document_count)

    def reciprocal_rank_fusion(
        self,
        result_lists: list[list[RetrievedDocument]],
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Fuse ranked lists using Reciprocal Rank Fusion."""
        settings = get_settings()
        rrf_k = k or settings.rrf_k
        scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}

        for results in result_lists:
            for rank, doc in enumerate(results, start=1):
                chunk_id = doc.metadata.get("chunk_id", doc.content[:50])
                key = str(chunk_id)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
                if key not in doc_map:
                    doc_map[key] = doc

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_docs: list[RetrievedDocument] = []
        for key, rrf_score in fused:
            doc = doc_map[key]
            fused_docs.append(
                RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=rrf_score,
                    source="hybrid_rrf",
                )
            )
        return fused_docs

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        """Run hybrid retrieval pipeline."""
        settings = get_settings()
        k = top_k or settings.hybrid_top_k

        vector_results = self.faiss.retrieve(query, top_k=settings.vector_top_k)
        bm25_results = self.bm25.retrieve(query, top_k=settings.bm25_top_k)

        fused = self.reciprocal_rank_fusion([vector_results, bm25_results])
        logger.info(
            "Hybrid retrieval: vector=%d, bm25=%d, fused=%d",
            len(vector_results),
            len(bm25_results),
            len(fused[:k]),
        )
        return fused[:k]
