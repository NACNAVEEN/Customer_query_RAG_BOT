"""Tests for metadata extraction and chunking."""

from app.ingestion.metadata import (
    ChunkMetadata,
    detect_headings,
    enrich_chunk_metadata,
    generate_chunk_id,
)


class TestMetadata:
    def test_generate_chunk_id_deterministic(self) -> None:
        id1 = generate_chunk_id("doc.pdf", 1, "sample content", 0)
        id2 = generate_chunk_id("doc.pdf", 1, "sample content", 0)
        assert id1 == id2
        assert len(id1) == 16

    def test_detect_headings(self) -> None:
        text = "Technology Stack\n\nANPR uses cameras.\n\n1. Introduction"
        headings = detect_headings(text)
        assert len(headings) >= 1

    def test_enrich_chunk_metadata(self) -> None:
        meta = enrich_chunk_metadata(
            source="instapark.pdf",
            page=7,
            content="InstaParkAI uses AI for parking.",
            index=0,
            active_heading="Technology Stack",
        )
        assert meta.source == "instapark.pdf"
        assert meta.page == 7
        assert meta.section == "Technology Stack"
        assert meta.chunk_id

    def test_chunk_metadata_to_dict(self) -> None:
        meta = ChunkMetadata(
            source="test.pdf",
            page=1,
            section="FAQ",
            heading="Pricing",
            chunk_id="abc123",
        )
        d = meta.to_dict()
        assert d["source"] == "test.pdf"
        assert d["page"] == 1
