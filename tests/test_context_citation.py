"""Tests for context and citation builders."""

from app.pipeline.citation_builder import CitationBuilder
from app.pipeline.context_builder import ContextBuilder
from app.retrievers.faiss_retriever import RetrievedDocument


class TestContextBuilder:
    def test_build_context_with_metadata(self) -> None:
        docs = [
            RetrievedDocument(
                content="InstaParkAI smart parking platform.",
                metadata={
                    "source": "brochure.pdf",
                    "page": 7,
                    "section": "Technology Stack",
                    "heading": "Overview",
                },
                score=0.85,
            )
        ]
        builder = ContextBuilder()
        context, meta = builder.build(docs)
        assert "[Source]" in context
        assert "brochure.pdf" in context
        assert "Technology Stack" in context
        assert len(meta) == 1


class TestCitationBuilder:
    def test_deduplicated_citations(self) -> None:
        builder = CitationBuilder()
        meta = [
            {"source": "doc.pdf", "page": 7, "section": "Technology Stack", "score": 0.9},
            {"source": "doc.pdf", "page": 7, "section": "Technology Stack", "score": 0.8},
            {"source": "doc.pdf", "page": 9, "section": "Business Benefits", "score": 0.7},
        ]
        citations = builder.build(meta)
        assert len(citations) == 2
        assert citations[0]["display"] == "Page 7 | Technology Stack"

    def test_format_citations_text(self) -> None:
        builder = CitationBuilder()
        citations = [{"display": "Page 7 | Technology Stack"}]
        text = builder.format_citations_text(citations)
        assert "Sources" in text
        assert "Page 7" in text
