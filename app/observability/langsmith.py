"""LangSmith observability integration."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Collected pipeline latency and quality metrics."""

    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    cache_hit: bool = False
    cache_hit_rate: float = 0.0
    hallucination_detected: bool = False
    top_retrieval_score: float = 0.0
    intent: str = ""
    intent_confidence: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "rerank_latency_ms": self.rerank_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "cache_hit": self.cache_hit,
            "cache_hit_rate": self.cache_hit_rate,
            "hallucination_detected": self.hallucination_detected,
            "top_retrieval_score": self.top_retrieval_score,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            **self.extra,
        }


class LangSmithObserver:
    """Configure and track LangSmith tracing."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.langsmith_enabled
        self.project = settings.langsmith_project
        self._configured = False
        self._metrics_history: list[PipelineMetrics] = []

        if self.enabled and settings.langsmith_api_key:
            self._configure_langsmith(settings)

    def _configure_langsmith(self, settings: Any) -> None:
        try:
            os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
            self._configured = True
            logger.info("LangSmith tracing enabled for project: %s", self.project)
        except Exception as exc:
            logger.warning("Failed to configure LangSmith: %s", exc)

    @contextmanager
    def trace_step(self, step_name: str) -> Generator[dict[str, float], None, None]:
        """Context manager to time a pipeline step."""
        timing: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield timing
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            timing["elapsed_ms"] = elapsed_ms
            logger.debug("Step '%s' completed in %.2f ms", step_name, elapsed_ms)

    def record_metrics(self, metrics: PipelineMetrics) -> None:
        """Record pipeline metrics for reporting."""
        self._metrics_history.append(metrics)
        logger.info("Pipeline metrics: %s", metrics.to_dict())

        if self.enabled and self._configured:
            try:
                from langsmith import Client

                client = Client()
                client.create_run(
                    name="instapark-rag-query",
                    run_type="chain",
                    inputs={"metrics": metrics.to_dict()},
                    project_name=self.project,
                )
            except Exception as exc:
                logger.debug("LangSmith run logging skipped: %s", exc)

    @property
    def average_cache_hit_rate(self) -> float:
        if not self._metrics_history:
            return 0.0
        hits = sum(1 for m in self._metrics_history if m.cache_hit)
        return hits / len(self._metrics_history)

    @property
    def hallucination_rate(self) -> float:
        if not self._metrics_history:
            return 0.0
        detected = sum(1 for m in self._metrics_history if m.hallucination_detected)
        return detected / len(self._metrics_history)
