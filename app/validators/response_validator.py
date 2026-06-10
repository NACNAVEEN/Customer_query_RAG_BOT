"""Validates the generated response for alignment and citation grounding."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.llm.gemini_client import GeminiClient
from app.validators.hallucination_guard import HallucinationGuard, NOT_FOUND_PHRASE

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Detailed response validation result."""

    valid: bool
    reason: str = ""
    regenerated: bool = False


class ResponseValidator:
    """Orchestrates response validation checks and handles regeneration."""

    def __init__(
        self,
        llm_client: GeminiClient | None = None,
        hallucination_guard: HallucinationGuard | None = None,
        max_retries: int = 2,
    ) -> None:
        self.llm = llm_client or GeminiClient()
        self.guard = hallucination_guard or HallucinationGuard()
        self.max_retries = max_retries

    def validate(
        self,
        answer: str,
        context: str,
        query: str,
        citations: list[dict] | None = None,
        retrieved_scores: list[float] | None = None,
    ) -> ValidationResult:
        """Run validation rules on the output."""
        if NOT_FOUND_PHRASE in answer:
            return ValidationResult(valid=True, reason="Info not found response is default valid.")

        # 1. Heuristic check
        guard_res = self.guard.check(answer, context, retrieved_scores)
        if not guard_res.passed:
            return ValidationResult(valid=False, reason=guard_res.reason)

        # 2. Check for citation grounding (ensure answer has sources if not empty)
        if not citations and context.strip():
            return ValidationResult(valid=False, reason="Response lacks appropriate citation links.")

        # 3. Simple keyword overlap validation between answer and context
        answer_tokens = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        context_tokens = set(re.findall(r"\b\w{4,}\b", context.lower()))
        if answer_tokens and context_tokens:
            overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
            if overlap < 0.10:  # Allow relatively low overlap but must have some shared keywords
                return ValidationResult(
                    valid=False,
                    reason=f"Insufficient terminology overlap ({overlap:.2f}) with context.",
                )

        # 4. LLM validator check (semantic grounding verification)
        try:
            llm_check = self.llm.generate_validation_check(answer, context, query)
            if "INVALID" in llm_check:
                return ValidationResult(
                    valid=False,
                    reason="LLM validation check failed; answer contains ungrounded claims.",
                )
        except Exception as exc:
            logger.warning("LLM-based validation check skipped due to exception: %s", exc)

        return ValidationResult(valid=True, reason="Validation succeeded.")

    def validate_with_regeneration(
        self,
        query: str,
        context: str,
        initial_answer: str,
        citations: list[dict] | None = None,
        retrieved_scores: list[float] | None = None,
    ) -> tuple[str, ValidationResult]:
        """Validate answer and attempt to regenerate if invalid, up to max_retries."""
        answer = initial_answer
        for attempt in range(self.max_retries + 1):
            val_result = self.validate(
                answer=answer,
                context=context,
                query=query,
                citations=citations,
                retrieved_scores=retrieved_scores,
            )
            if val_result.valid:
                val_result.regenerated = attempt > 0
                return answer, val_result

            logger.info("Validation failed on attempt %d: %s", attempt + 1, val_result.reason)

            # Re-generate only if retries left
            if attempt < self.max_retries:
                try:
                    # Request regeneration with explicit correction notice
                    reg_query = f"{query} (Note: Please ensure the response relies strictly on the provided context without adding outside facts.)"
                    answer = self.llm.generate(reg_query, context)
                except Exception as exc:
                    logger.error("Failed to regenerate answer during validation loop: %s", exc)
                    break

        # If all retries fail, return the default NOT_FOUND phrase to avoid hallucination
        return NOT_FOUND_PHRASE, ValidationResult(
            valid=True,
            reason="Failed validation checks, returned default not found response.",
            regenerated=True,
        )
