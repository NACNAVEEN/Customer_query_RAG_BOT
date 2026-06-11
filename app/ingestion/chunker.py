"""Recursive text chunking with metadata preservation and FAQ-aware splitting."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings
from app.ingestion.loader import PageDocument
from app.ingestion.metadata import enrich_chunk_metadata

logger = logging.getLogger(__name__)

# Pattern to detect FAQ question-answer pairs (e.g., "Q1:", "Q2:", "Q3:" etc.)
FAQ_PATTERN = re.compile(
    r"(?=(?:^|\n)\s*Q\d+[\.:]\s*)",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class DocumentChunk:
    """A text chunk with associated metadata."""

    content: str
    metadata: dict[str, Any]

    def to_langchain_dict(self) -> dict[str, Any]:
        return {"page_content": self.content, "metadata": self.metadata}


class RecursiveChunker:
    """Split documents using recursive character splitting with FAQ awareness."""

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

    def _is_faq_section(self, text: str) -> bool:
        """Detect if the text contains FAQ-style Q&A pairs."""
        matches = FAQ_PATTERN.findall(text)
        return len(matches) >= 2

    def _split_faq(self, text: str) -> list[str]:
        """Split FAQ text into individual Q&A pairs.

        Each chunk will contain exactly one question-answer pair.
        """
        parts = FAQ_PATTERN.split(text)
        qa_chunks: list[str] = []

        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            # If the part is a preamble (before the first Q), keep it as a chunk
            # If the part is a Q&A pair, keep it as its own chunk
            qa_chunks.append(stripped)

        return qa_chunks

    def chunk_page(self, page: PageDocument) -> list[DocumentChunk]:
        """Chunk a single page document with FAQ-aware splitting."""
        if not page.text.strip():
            return []

        active_heading = page.headings[-1] if page.headings else ""
        chunks: list[DocumentChunk] = []

        # Check if this page contains FAQ content
        if self._is_faq_section(page.text):
            logger.debug("FAQ section detected on page %d of %s", page.page_number, page.source)
            faq_chunks = self._split_faq(page.text)

            for idx, text in enumerate(faq_chunks):
                # Extract the question as heading for better metadata
                q_match = re.match(r"(Q\d+[\.:]\s*.+?)(?:\n|$)", text, re.IGNORECASE)
                chunk_heading = q_match.group(1).strip() if q_match else active_heading

                meta = enrich_chunk_metadata(
                    source=page.source,
                    page=page.page_number,
                    content=text,
                    index=idx,
                    headings_on_page=page.headings,
                    active_heading=chunk_heading,
                )
                chunks.append(DocumentChunk(content=text, metadata=meta.to_dict()))

            # If any FAQ chunk is too large, further split it
            final_chunks: list[DocumentChunk] = []
            for chunk in chunks:
                if len(chunk.content) > self.chunk_size:
                    sub_texts = self._splitter.split_text(chunk.content)
                    for sub_idx, sub_text in enumerate(sub_texts):
                        meta = enrich_chunk_metadata(
                            source=page.source,
                            page=page.page_number,
                            content=sub_text,
                            index=sub_idx,
                            headings_on_page=page.headings,
                            active_heading=chunk.metadata.get("heading", active_heading),
                        )
                        final_chunks.append(DocumentChunk(content=sub_text, metadata=meta.to_dict()))
                else:
                    final_chunks.append(chunk)
            chunks = final_chunks
        else:
            # Standard recursive splitting for non-FAQ content
            text_chunks = self._splitter.split_text(page.text)
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
