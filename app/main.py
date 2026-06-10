"""InstaParkAI RAG Chatbot - Main entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config.settings import get_settings
from app.pipeline.rag_graph import RAGPipeline

NOT_FOUND_RESPONSE = "I could not find this information in the provided knowledge base."


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_cli_query(pipeline: RAGPipeline, query: str) -> None:
    """Run a single query from CLI."""
    result = pipeline.query(query)
    print("\n" + "=" * 60)
    print("InstaParkAI RAG Chatbot Response")
    print("=" * 60)
    print(result["answer"])
    if result.get("citations"):
        print("\nCitations:")
        for cite in result["citations"]:
            print(f"  - {cite.get('display', cite)}")
    if result.get("metrics"):
        print(f"\nLatency: {result['metrics'].get('total_latency_ms', 0):.0f} ms")
    print("=" * 60)


def run_cli_ingest(pipeline: RAGPipeline, pdf_path: str) -> None:
    """Ingest a PDF from CLI."""
    result = pipeline.ingest_pdf(pdf_path)
    print(f"Ingested {result['pages']} pages, {result['chunks']} chunks")
    print(f"Index counts: {result['indexed']}")


def main() -> None:
    """Main CLI entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="InstaParkAI RAG Chatbot")
    parser.add_argument("--query", "-q", type=str, help="Run a single query")
    parser.add_argument("--ingest", "-i", type=str, help="Ingest a PDF file")
    args = parser.parse_args()

    settings = get_settings()
    logger.info("Starting %s [%s]", settings.app_name, settings.environment)

    pipeline = RAGPipeline()

    if args.ingest:
        pdf = Path(args.ingest)
        if not pdf.exists():
            logger.error("PDF not found: %s", pdf)
            sys.exit(1)
        run_cli_ingest(pipeline, str(pdf))
    elif args.query:
        run_cli_query(pipeline, args.query)
    else:
        print("InstaParkAI RAG Chatbot")
        print("Usage:")
        print("  python -m app.main --ingest path/to/document.pdf")
        print("  python -m app.main --query 'What is ANPR technology?'")
        print("  uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()

