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

            # Build clean display string: 📖 Section > Heading | Page N
            display = self._format_display(source, page, section, heading)

            citations.append({
                "source": source,
                "page": page,
                "section": section,
                "heading": heading,
                "display": display,
            })

        logger.debug("Generated %d citations from retrieved metadata", len(citations))
        return citations

    def _format_display(
        self,
        source: str,
        page: int,
        section: str,
        heading: str,
    ) -> str:
        """Format a clean, readable citation display string.

        Examples:
            📖 FAQ > What technologies are used? | Page 15
            📖 Technology Stack | Page 7
            📖 Company Overview | Page 3
        """
        parts: list[str] = []

        # Use section as primary label
        if section and section != "General":
            parts.append(section)

        # Add heading as sub-label if different from section
        if heading and heading != section:
            # Truncate very long headings
            display_heading = heading if len(heading) <= 80 else heading[:77] + "..."
            if parts:
                parts.append(f" > {display_heading}")
            else:
                parts.append(display_heading)

        # Fallback to source name if no section/heading
        if not parts:
            # Remove .pdf extension for cleaner display
            clean_source = source.rsplit(".", 1)[0] if "." in source else source
            parts.append(clean_source)

        display = "".join(parts)

        # Add page number
        if page > 0:
            display += f" | Page {page}"

        return display

    def format_citations_text(self, citations: list[dict[str, Any]]) -> str:
        """Create a plain text representation of citations for chat history."""
        if not citations:
            return ""

        lines = ["\n\nSources:"]
        for idx, cite in enumerate(citations, 1):
            lines.append(f"[{idx}] {cite['display']}")
        return "\n".join(lines)
