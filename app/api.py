"""FastAPI REST API backend for InstaParkAI RAG Chatbot.

Exposes the RAG pipeline as HTTP endpoints so any frontend (React, Vue,
Angular, mobile apps, etc.) can consume the chatbot service.

Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.main import setup_logging
from app.pipeline.rag_graph import RAGPipeline

logger = logging.getLogger(__name__)

# ─── Pipeline Singleton ──────────────────────────────────────────────────────
# Shared across all requests. Initialised once during application startup
# via the lifespan context manager below.

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the global pipeline instance (must be called after startup)."""
    assert _pipeline is not None, "Pipeline not initialised – server not started?"
    return _pipeline


@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001 – required by FastAPI
    """Startup / shutdown lifecycle hook.

    Heavy-weight resources (embeddings model, FAISS index, reranker) are
    loaded once here, not on every request.
    """
    global _pipeline  # noqa: PLW0603
    setup_logging()
    logger.info("Initialising RAG pipeline …")
    _pipeline = RAGPipeline()
    logger.info(
        "Pipeline ready – %d chunks indexed", _pipeline.retriever.document_count
    )
    yield
    logger.info("Shutting down …")


# ─── FastAPI Application ─────────────────────────────────────────────────────

app = FastAPI(
    title="InstaParkAI RAG API",
    description=(
        "REST API exposing the InstaParkAI Smart Parking RAG chatbot. "
        "Supports querying the knowledge base, ingesting PDF documents, "
        "managing chat sessions, and retrieving system health."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow React dev server (Vite 5173, CRA 3000) and any other local frontend.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming chat query from the frontend."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for conversation continuity",
    )


class Citation(BaseModel):
    """A single citation reference."""
    source: str = ""
    page: int | None = None
    chunk_id: str = ""
    display: str = ""


class QueryResponse(BaseModel):
    """Structured response returned to the frontend."""
    answer: str
    citations: list[dict[str, Any]] = []
    intent: str = ""
    intent_confidence: float = 0.0
    cache_hit: bool = False
    latency_ms: float = 0.0
    session_id: str = ""


class IngestResponse(BaseModel):
    """Result of document ingestion."""
    filename: str
    pages: int
    chunks: int
    indexed: dict[str, int] = {}


class HealthResponse(BaseModel):
    """System health / readiness check."""
    status: str
    indexed_chunks: int
    cache_enabled: bool
    llm_model: str
    embedding_model: str


class SessionClearResponse(BaseModel):
    """Confirmation of session clear."""
    message: str
    session_id: str


class CacheClearResponse(BaseModel):
    """Confirmation of cache clear."""
    message: str


# ─── In-Memory Session Store ─────────────────────────────────────────────────
# Maps session_id → RAGPipeline (each session has its own conversation memory).
# For a production deployment swap this for Redis-backed sessions.

_sessions: dict[str, RAGPipeline] = {}


def _get_session_pipeline(session_id: str | None) -> tuple[str, RAGPipeline]:
    """Return (session_id, pipeline) – creates a new session if needed.

    All sessions share the *same* retriever, reranker, and cache, but each
    gets its own ConversationBufferMemory so that chat history is isolated.
    """
    global_pipeline = get_pipeline()

    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]

    # Create a new session with its own memory but shared heavy components
    sid = session_id or str(uuid.uuid4())
    from app.memory.conversation_memory import ConversationBufferMemory

    session_pipeline = RAGPipeline(
        retriever=global_pipeline.retriever,
        reranker=global_pipeline.reranker,
        cache=global_pipeline.cache,
        memory=ConversationBufferMemory(),
    )
    _sessions[sid] = session_pipeline
    logger.info("Created new chat session: %s", sid)
    return sid, session_pipeline


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/api/query", response_model=QueryResponse, tags=["Chat"])
async def query_chatbot(request: QueryRequest):
    """Send a question to the RAG chatbot and receive an answer.

    This is the primary endpoint your React frontend should call.
    Pass an optional `session_id` to maintain conversation history
    across multiple turns.
    """
    try:
        session_id, pipeline = _get_session_pipeline(request.session_id)
        result = pipeline.query(request.query)

        return QueryResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            intent=result.get("intent", ""),
            intent_confidence=result.get("intent_confidence", 0.0),
            cache_hit=result.get("cache_hit", False),
            latency_ms=result.get("metrics", {}).get("total_latency_ms", 0.0),
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document into the knowledge base.

    The file is saved to disk, chunked, embedded, and indexed into
    both the FAISS vector store and BM25 keyword index.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = get_settings()
    save_path = settings.uploads_dir / file.filename

    try:
        content = await file.read()
        save_path.write_bytes(content)
        logger.info("Saved uploaded file: %s (%d bytes)", save_path, len(content))

        pipeline = get_pipeline()
        result = pipeline.ingest_pdf(str(save_path))

        return IngestResponse(
            filename=file.filename,
            pages=result["pages"],
            chunks=result["chunks"],
            indexed=result.get("indexed", {}),
        )
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Lightweight readiness probe.

    Returns the number of indexed chunks, which model is in use, and
    whether the semantic cache is active. Use this in your React app
    to show a connection status indicator.
    """
    settings = get_settings()
    pipeline = get_pipeline()
    model_name = (
        settings.groq_model if settings.groq_api_key else settings.gemini_model
    )
    return HealthResponse(
        status="healthy",
        indexed_chunks=pipeline.retriever.document_count,
        cache_enabled=settings.gptcache_enabled,
        llm_model=model_name,
        embedding_model=settings.embedding_model,
    )


@app.post(
    "/api/sessions/{session_id}/clear",
    response_model=SessionClearResponse,
    tags=["Sessions"],
)
async def clear_session(session_id: str):
    """Clear conversation history for a specific session.

    The knowledge base and cache remain intact.
    """
    if session_id in _sessions:
        _sessions[session_id].memory.clear()
        return SessionClearResponse(
            message="Chat history cleared.", session_id=session_id
        )
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


@app.post("/api/cache/clear", response_model=CacheClearResponse, tags=["System"])
async def clear_cache():
    """Flush the semantic cache (all sessions)."""
    pipeline = get_pipeline()
    pipeline.cache.clear()
    return CacheClearResponse(message="Semantic cache cleared.")

# ─── Static Frontend (Production) ────────────────────────────────────────────
# Serve the web/ frontend from the same process so a single Render service
# handles both API and UI.  API routes are registered first, so /api/* always
# takes priority over static file matching.

WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static-root")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_spa(path: str):
        file = WEB_DIR / path
        if file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(WEB_DIR / "index.html"))

# ─── Dev Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
