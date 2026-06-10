"""LangGraph orchestrator for the RAG pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.cache.semantic_cache import SemanticCache
from app.config.settings import get_settings
from app.llm.gemini_client import GeminiClient
from app.memory.conversation_memory import ConversationBufferMemory
from app.observability.langsmith import LangSmithObserver, PipelineMetrics
from app.pipeline.citation_builder import CitationBuilder
from app.pipeline.context_builder import ContextBuilder
from app.preprocessing.intent_classifier import Intent, IntentClassifier
from app.preprocessing.query_preprocessor import QueryPreprocessor
from app.preprocessing.query_rewriter import QueryRewriter
from app.reranker.reranker import RerankerService
from app.retrievers.faiss_retriever import FAISSRetriever, RetrievedDocument
from app.retrievers.hybrid_retriever import HybridRetriever
from app.validators.hallucination_guard import NOT_FOUND_PHRASE
from app.validators.response_validator import ResponseValidator

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    """Execution state within the RAG pipeline."""

    original_query: str
    preprocessed_query: str
    rewritten_query: str
    intent: str
    intent_confidence: float
    cache_hit: bool
    answer: str
    citations: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    context: str
    chunk_metadata: list[dict[str, Any]]
    metrics: dict[str, Any]
    error: str
    skip_llm: bool


class RAGPipeline:
    """Orchestrates RAG workflow using modular LangGraph nodes."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: RerankerService | None = None,
        cache: SemanticCache | None = None,
        memory: ConversationBufferMemory | None = None,
    ) -> None:
        self.settings = get_settings()
        self.preprocessor = QueryPreprocessor()
        self.intent_classifier = IntentClassifier()
        self.query_rewriter = QueryRewriter()
        self.cache = cache or SemanticCache()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or RerankerService()
        self.context_builder = ContextBuilder()
        self.citation_builder = CitationBuilder()
        self.llm = GeminiClient()
        self.validator = ResponseValidator(llm_client=self.llm)
        self.observer = LangSmithObserver()
        self.memory = memory or ConversationBufferMemory()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(RAGState)

        # 1. Define nodes
        graph.add_node("preprocess", self._node_preprocess)
        graph.add_node("classify_intent", self._node_classify_intent)
        graph.add_node("check_cache", self._node_check_cache)
        graph.add_node("rewrite_query", self._node_rewrite_query)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("rerank", self._node_rerank)
        graph.add_node("confidence_gate", self._node_confidence_gate)
        graph.add_node("build_context", self._node_build_context)
        graph.add_node("generate", self._node_generate)
        graph.add_node("validate", self._node_validate)
        graph.add_node("finalize", self._node_finalize)

        # 2. Define edges
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "classify_intent")
        graph.add_edge("classify_intent", "check_cache")

        # Cache decision routing
        graph.add_conditional_edges(
            "check_cache",
            self._route_cache,
            {"cached": "finalize", "continue": "rewrite_query"},
        )

        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "confidence_gate")

        # Confidence gate routing
        graph.add_conditional_edges(
            "confidence_gate",
            self._route_confidence,
            {"reject": "finalize", "accept": "build_context"},
        )

        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _route_cache(self, state: RAGState) -> str:
        return "cached" if state.get("cache_hit") else "continue"

    def _route_confidence(self, state: RAGState) -> str:
        return "reject" if state.get("skip_llm") else "accept"

    # Nodes implementation
    def _node_preprocess(self, state: RAGState) -> RAGState:
        query = state.get("original_query", "")
        res = self.preprocessor.preprocess(query)
        return {"preprocessed_query": res.normalized}

    def _node_classify_intent(self, state: RAGState) -> RAGState:
        query = state.get("preprocessed_query", "")
        res = self.intent_classifier.classify(query)
        return {
            "intent": res.intent.value,
            "intent_confidence": res.confidence,
        }

    def _node_check_cache(self, state: RAGState) -> RAGState:
        query = state.get("preprocessed_query", "")
        cache_res = self.cache.lookup(query)

        if cache_res.hit:
            logger.info("Semantic cache hit for: %s", query)
            return {
                "cache_hit": True,
                "answer": cache_res.response,
                "citations": cache_res.citations or [],
                "metrics": {"cache_similarity": cache_res.similarity},
            }
        return {"cache_hit": False}

    def _node_rewrite_query(self, state: RAGState) -> RAGState:
        query = state.get("preprocessed_query", "")
        intent_str = state.get("intent", "")
        from app.preprocessing.intent_classifier import IntentResult

        try:
            intent_enum = Intent(intent_str)
        except ValueError:
            intent_enum = Intent.FAQ

        intent_res = IntentResult(
            intent=intent_enum,
            confidence=state.get("intent_confidence", 0.0),
            scores={},
        )
        res = self.query_rewriter.rewrite(query, intent_res)
        return {"rewritten_query": res.rewritten}

    def _node_retrieve(self, state: RAGState) -> RAGState:
        query = state.get("rewritten_query", "")
        start_time = time.perf_counter()

        with self.observer.trace_step("retrieval"):
            docs = self.retriever.retrieve(query)

        latency = (time.perf_counter() - start_time) * 1000
        return {
            "retrieved_docs": [d.to_dict() for d in docs],
            "metrics": {
                **state.get("metrics", {}),
                "retrieval_latency_ms": latency,
            },
        }

    def _node_rerank(self, state: RAGState) -> RAGState:
        query = state.get("rewritten_query", "")
        retrieved_dicts = state.get("retrieved_docs", [])
        start_time = time.perf_counter()

        docs = [
            RetrievedDocument(
                content=d["content"],
                metadata=d["metadata"],
                score=d["score"],
            )
            for d in retrieved_dicts
        ]

        with self.observer.trace_step("rerank"):
            reranked = self.reranker.rerank(query, docs)

        latency = (time.perf_counter() - start_time) * 1000
        return {
            "retrieved_docs": [d.to_dict() for d in reranked],
            "metrics": {
                **state.get("metrics", {}),
                "rerank_latency_ms": latency,
            },
        }

    def _node_confidence_gate(self, state: RAGState) -> RAGState:
        """Similarity threshold gate."""
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {
                "skip_llm": True,
                "answer": NOT_FOUND_PHRASE,
                "citations": [],
            }

        top_score = docs[0].get("score", 0.0)
        # Check if similarity meets configured threshold
        if top_score < self.settings.similarity_threshold:
            logger.info(
                "Gate rejected query. Top score (%.4f) < threshold (%.4f)",
                top_score,
                self.settings.similarity_threshold,
            )
            return {
                "skip_llm": True,
                "answer": NOT_FOUND_PHRASE,
                "citations": [],
                "metrics": {
                    **state.get("metrics", {}),
                    "top_retrieval_score": top_score,
                },
            }

        return {
            "skip_llm": False,
            "metrics": {
                **state.get("metrics", {}),
                "top_retrieval_score": top_score,
            },
        }

    def _node_build_context(self, state: RAGState) -> RAGState:
        retrieved_dicts = state.get("retrieved_docs", [])
        docs = [
            RetrievedDocument(
                content=d["content"],
                metadata=d["metadata"],
                score=d["score"],
            )
            for d in retrieved_dicts
        ]
        context_str, chunk_meta = self.context_builder.build(docs)
        citations = self.citation_builder.build(chunk_meta)
        return {
            "context": context_str,
            "chunk_metadata": chunk_meta,
            "citations": citations,
        }

    def _node_generate(self, state: RAGState) -> RAGState:
        query = state.get("rewritten_query", "")
        context = state.get("context", "")
        chat_history = self.memory.get_history()

        start_time = time.perf_counter()
        with self.observer.trace_step("llm_generation"):
            answer = self.llm.generate(
                query=query,
                context=context,
                chat_history=chat_history,
            )

        latency = (time.perf_counter() - start_time) * 1000
        return {
            "answer": answer,
            "metrics": {
                **state.get("metrics", {}),
                "llm_latency_ms": latency,
            },
        }

    def _node_validate(self, state: RAGState) -> RAGState:
        answer = state.get("answer", "")
        context = state.get("context", "")
        query = state.get("rewritten_query", "")
        citations = state.get("citations", [])
        scores = [d.get("score", 0.0) for d in state.get("retrieved_docs", [])]

        validated_answer, val_res = self.validator.validate_with_regeneration(
            query=query,
            context=context,
            initial_answer=answer,
            citations=citations,
            retrieved_scores=scores,
        )

        return {
            "answer": validated_answer,
            "metrics": {
                **state.get("metrics", {}),
                "validation_passed": val_res.valid,
                "hallucination_detected": not val_res.valid,
            },
        }

    def _node_finalize(self, state: RAGState) -> RAGState:
        answer = state.get("answer", NOT_FOUND_PHRASE)
        citations = state.get("citations", [])


        # Write to semantic cache
        if not state.get("cache_hit"):
            self.cache.store(
                query=state.get("preprocessed_query", ""),
                response=answer,
                citations=citations,
                metadata=state.get("metrics", {}),
            )

        # Save to memory
        self.memory.add(
            user=state.get("original_query", ""),
            assistant=answer,
            metadata={"intent": state.get("intent", "")},
        )

        # Log pipeline performance metrics
        metrics = PipelineMetrics(
            retrieval_latency_ms=state.get("metrics", {}).get("retrieval_latency_ms", 0.0),
            rerank_latency_ms=state.get("metrics", {}).get("rerank_latency_ms", 0.0),
            llm_latency_ms=state.get("metrics", {}).get("llm_latency_ms", 0.0),
            cache_hit=state.get("cache_hit", False),
            cache_hit_rate=self.cache.hit_rate,
            hallucination_detected=state.get("metrics", {}).get("hallucination_detected", False),
            top_retrieval_score=state.get("metrics", {}).get("top_retrieval_score", 0.0),
            intent=state.get("intent", ""),
            intent_confidence=state.get("intent_confidence", 0.0),
        )
        self.observer.record_metrics(metrics)

        return {
            "answer": answer,
            "citations": citations,
            "metrics": {**state.get("metrics", {}), **metrics.to_dict()},
        }

    def query(self, user_query: str) -> dict[str, Any]:
        """Execute query pipeline."""
        start_time = time.perf_counter()
        initial_state: RAGState = {"original_query": user_query}

        try:
            result = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.exception("Pipeline invocation failed.")
            return {
                "answer": NOT_FOUND_PHRASE,
                "citations": [],
                "error": str(exc),
                "metrics": {},
            }

        total_latency = (time.perf_counter() - start_time) * 1000
        result.setdefault("metrics", {})["total_latency_ms"] = total_latency

        return {
            "answer": result.get("answer", NOT_FOUND_PHRASE),
            "citations": result.get("citations", []),
            "retrieved_docs": result.get("retrieved_docs", []),
            "intent": result.get("intent", ""),
            "intent_confidence": result.get("intent_confidence", 0.0),
            "cache_hit": result.get("cache_hit", False),
            "metrics": result.get("metrics", {}),
            "error": result.get("error", ""),
        }

    def ingest_pdf(self, file_path: str) -> dict[str, Any]:
        """Load and index PDF document."""
        from app.ingestion.chunker import RecursiveChunker
        from app.ingestion.loader import PDFLoader

        loader = PDFLoader(file_path)
        pages = loader.load_pages()

        chunker = RecursiveChunker()
        chunks = chunker.chunk_pages(pages)

        # Add to both index components
        counts = self.retriever.index_documents(chunks)
        faiss_count = counts["faiss"]
        bm25_count = counts["bm25"]

        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "indexed": {"faiss": faiss_count, "bm25": bm25_count},
        }
