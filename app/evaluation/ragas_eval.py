"""RAGAS evaluation for RAG pipeline quality."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Single evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


@dataclass
class EvaluationReport:
    """RAGAS evaluation report."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    samples_evaluated: int
    timestamp: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "samples_evaluated": self.samples_evaluated,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class RAGASEvaluator:
    """Run RAGAS metrics on evaluation samples."""

    def __init__(self) -> None:
        settings = get_settings()
        self.enabled = settings.ragas_enabled
        self.reports_dir = settings.evaluation_reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, samples: list[EvaluationSample]) -> EvaluationReport:
        """Run RAGAS evaluation and generate report."""
        if not samples:
            return EvaluationReport(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                samples_evaluated=0,
                timestamp=datetime.utcnow().isoformat(),
                details={"error": "No samples provided"},
            )

        if not self.enabled:
            return self._fallback_evaluation(samples)

        try:
            return self._run_ragas(samples)
        except Exception as exc:
            logger.warning("RAGAS evaluation failed, using fallback: %s", exc)
            return self._fallback_evaluation(samples)

    def _run_ragas(self, samples: list[EvaluationSample]) -> EvaluationReport:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }
        if all(s.ground_truth for s in samples):
            data["ground_truth"] = [s.ground_truth or "" for s in samples]

        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        result = evaluate(dataset, metrics=metrics)

        report = EvaluationReport(
            faithfulness=float(result.get("faithfulness", 0.0) or 0.0),
            answer_relevancy=float(result.get("answer_relevancy", 0.0) or 0.0),
            context_precision=float(result.get("context_precision", 0.0) or 0.0),
            context_recall=float(result.get("context_recall", 0.0) or 0.0),
            samples_evaluated=len(samples),
            timestamp=datetime.utcnow().isoformat(),
            details=dict(result),
        )
        self._save_report(report)
        return report

    def _fallback_evaluation(self, samples: list[EvaluationSample]) -> EvaluationReport:
        """Heuristic evaluation when RAGAS is unavailable."""
        faithfulness_scores: list[float] = []
        relevancy_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []

        for sample in samples:
            answer_words = set(sample.answer.lower().split())
            context_words: set[str] = set()
            for ctx in sample.contexts:
                context_words.update(ctx.lower().split())

            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                faithfulness_scores.append(min(overlap * 1.2, 1.0))

            q_words = set(sample.question.lower().split())
            if q_words:
                relevancy = len(q_words & answer_words) / len(q_words)
                relevancy_scores.append(min(relevancy * 1.5, 1.0))

            if context_words and q_words:
                precision_scores.append(len(q_words & context_words) / max(len(q_words), 1))
                recall_scores.append(len(q_words & context_words) / max(len(context_words), 1))

        report = EvaluationReport(
            faithfulness=sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            answer_relevancy=sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
            context_precision=sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
            context_recall=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
            samples_evaluated=len(samples),
            timestamp=datetime.utcnow().isoformat(),
            details={"method": "fallback_heuristic"},
        )
        self._save_report(report)
        return report

    def _save_report(self, report: EvaluationReport) -> Path:
        filename = f"ragas_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.reports_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Saved evaluation report to %s", path)
        return path
