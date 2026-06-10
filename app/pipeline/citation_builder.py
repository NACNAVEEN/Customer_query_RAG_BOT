"""Generates structured citation entries for responses."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CitationBuilder:
    """Consolidates document metadata to generate unique citation links."""

    def build(self, chunk_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate and format citations from chunk metadata list."""
        seen: set[str] = set()
        citations: list[dict[str, Any]] = []

        for meta in chunk_metadata:
            source = meta.get("source", "unknown")
            page = meta.get("page", 0)
            section = meta.get("section", "General")
            heading = meta.get("heading", "")

            # Deduplicate by document and page/heading
            key = f"{source}:{page}:{heading}"
            if key in seen:
                continue
            seen.add(key)

            citations.append({
                "source": source,
                "page": page,
                "section": section,
                "heading": heading,
                "display": f"Page {page} | {section}" if page > 0 else section,
            })

        logger.debug("Generated %d citations from retrieved metadata", len(citations))
        return citations

    def format_citations_text(self, citations: list[dict[str, Any]]) -> str:
        """Create a plain text representation of citations for chat history."""
        if not citations:
            return ""

        lines = ["\n\nSources:"]
        for idx, cite in enumerate(citations, 1):
            lines.append(f"[{idx}] {cite['display']}")
        return "\n".join(lines)
