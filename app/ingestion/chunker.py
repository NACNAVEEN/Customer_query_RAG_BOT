"""Recursive text chunking with metadata preservation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings
from app.ingestion.loader import PageDocument
from app.ingestion.metadata import enrich_chunk_metadata

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A text chunk with associated metadata."""

    content: str
    metadata: dict[str, Any]

    def to_langchain_dict(self) -> dict[str, Any]:
        return {"page_content": self.content, "metadata": self.metadata}


class RecursiveChunker:
    """Split documents using recursive character splitting."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_page(self, page: PageDocument) -> list[DocumentChunk]:
        """Chunk a single page document."""
        if not page.text.strip():
            return []

        active_heading = page.headings[-1] if page.headings else ""
        text_chunks = self._splitter.split_text(page.text)
        chunks: list[DocumentChunk] = []

        for idx, text in enumerate(text_chunks):
            meta = enrich_chunk_metadata(
                source=page.source,
                page=page.page_number,
                content=text,
                index=idx,
                headings_on_page=page.headings,
                active_heading=active_heading,
            )
            chunks.append(DocumentChunk(content=text, metadata=meta.to_dict()))

        logger.debug(
            "Created %d chunks from %s page %d",
            len(chunks),
            page.source,
            page.page_number,
        )
        return chunks

    def chunk_pages(self, pages: list[PageDocument]) -> list[DocumentChunk]:
        """Chunk multiple pages."""
        all_chunks: list[DocumentChunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_page(page))
        logger.info("Created %d total chunks from %d pages", len(all_chunks), len(pages))
        return all_chunks
