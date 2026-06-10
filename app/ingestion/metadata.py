"""Metadata extraction and enrichment for document chunks."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:Chapter|Section|Part)\s+\d+[\.\:]?\s*.+|"
    r"\d+[\.\)]\s+[A-Z].+|"
    r"[A-Z][A-Z\s]{2,}$|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"
    r")$",
    re.MULTILINE,
)


@dataclass
class ChunkMetadata:
    """Structured metadata for a document chunk."""

    source: str
    page: int
    section: str = "General"
    heading: str = ""
    chunk_id: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "heading": self.heading,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkMetadata:
        return cls(
            source=str(data.get("source", "unknown")),
            page=int(data.get("page", 0)),
            section=str(data.get("section", "General")),
            heading=str(data.get("heading", "")),
            chunk_id=str(data.get("chunk_id", "")),
        )


def generate_chunk_id(source: str, page: int, content: str, index: int) -> str:
    """Generate a deterministic unique chunk identifier."""
    payload = f"{source}:{page}:{index}:{content[:200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def detect_headings(text: str) -> list[str]:
    """Detect potential headings within page text."""
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            continue
        if HEADING_PATTERN.match(stripped):
            headings.append(stripped)
    return headings


def extract_section_from_headings(headings: list[str], current_heading: str) -> str:
    """Map heading hierarchy to a section name."""
    if current_heading:
        return current_heading
    if headings:
        return headings[0]
    return "General"


def enrich_chunk_metadata(
    source: str,
    page: int,
    content: str,
    index: int,
    headings_on_page: list[str] | None = None,
    active_heading: str = "",
) -> ChunkMetadata:
    """Build complete metadata for a chunk."""
    headings = headings_on_page or detect_headings(content)
    heading = active_heading or (headings[-1] if headings else "")
    section = extract_section_from_headings(headings, heading)
    chunk_id = generate_chunk_id(source, page, content, index)

    metadata = ChunkMetadata(
        source=source,
        page=page,
        section=section,
        heading=heading,
        chunk_id=chunk_id,
    )
    logger.debug("Enriched metadata for chunk %s from %s page %d", chunk_id, source, page)
    return metadata
