"""Tests for hybrid retrieval and RRF fusion."""

from app.ingestion.chunker import DocumentChunk
from app.retrievers.faiss_retriever import RetrievedDocument
from app.retrievers.hybrid_retriever import HybridRetriever


def _make_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            content="InstaParkAI provides ANPR-based smart parking solutions.",
            metadata={
                "source": "test.pdf",
                "page": 1,
                "section": "Technology",
                "heading": "ANPR",
                "chunk_id": "chunk_001",
            },
        ),
        DocumentChunk(
            content="Pricing includes AMC annual maintenance contract models.",
            metadata={
                "source": "test.pdf",
                "page": 2,
                "section": "Pricing",
                "heading": "AMC",
                "chunk_id": "chunk_002",
            },
        ),
        DocumentChunk(
            content="RFID sensors enable contactless parking access control.",
            metadata={
                "source": "test.pdf",
                "page": 3,
                "section": "Technology",
                "heading": "RFID",
                "chunk_id": "chunk_003",
            },
        ),
    ]


class TestHybridRetriever:
    def test_index_and_retrieve(self) -> None:
        retriever = HybridRetriever()
        chunks = _make_chunks()
        retriever.index_documents(chunks)

        results = retriever.retrieve("ANPR parking technology")
        assert len(results) > 0
        assert any("ANPR" in r.content for r in results)

    def test_rrf_fusion(self) -> None:
        retriever = HybridRetriever()
        list_a = [
            RetrievedDocument("doc a", {"chunk_id": "1"}, 0.9, "faiss"),
            RetrievedDocument("doc b", {"chunk_id": "2"}, 0.8, "faiss"),
        ]
        list_b = [
            RetrievedDocument("doc b", {"chunk_id": "2"}, 0.95, "bm25"),
            RetrievedDocument("doc c", {"chunk_id": "3"}, 0.7, "bm25"),
        ]
        fused = retriever.reciprocal_rank_fusion([list_a, list_b])
        assert len(fused) == 3
        assert fused[0].metadata["chunk_id"] == "2"
