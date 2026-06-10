"""PDF document loading with page-level text extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

from app.ingestion.metadata import detect_headings

logger = logging.getLogger(__name__)


@dataclass
class PageDocument:
    """Represents a single PDF page with extracted text."""

    source: str
    page_number: int
    text: str
    headings: list[str]

    @property
    def metadata(self) -> dict[str, str | int | list[str]]:
        return {
            "source": self.source,
            "page": self.page_number,
            "headings": self.headings,
        }


class PDFLoader:
    """Load PDF files and extract text per page with heading detection."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {self.file_path.suffix}")

    def load_pages(self) -> list[PageDocument]:
        """Load all pages from the PDF."""
        try:
            reader = PdfReader(str(self.file_path))
        except Exception as exc:
            logger.exception("Failed to read PDF: %s", self.file_path)
            raise RuntimeError(f"Failed to read PDF: {self.file_path}") from exc

        source_name = self.file_path.name
        pages: list[PageDocument] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Failed to extract page %d from %s: %s", page_idx, source_name, exc)
                text = ""

            text = text.strip()
            if not text:
                logger.debug("Skipping empty page %d in %s", page_idx, source_name)
                continue

            headings = detect_headings(text)
            pages.append(
                PageDocument(
                    source=source_name,
                    page_number=page_idx,
                    text=text,
                    headings=headings,
                )
            )

        logger.info("Loaded %d pages from %s", len(pages), source_name)
        return pages

    def lazy_load(self) -> Iterator[PageDocument]:
        """Lazily yield pages from the PDF."""
        for page in self.load_pages():
            yield page
