"""Heuristic checks to detect potential hallucinations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NOT_FOUND_PHRASE = "I could not find this information in the provided knowledge base."

SUSPICIOUS_PATTERNS = [
    re.compile(r"\b(I think|probably|maybe|likely|perhaps)\b", re.I),
    re.compile(r"\b(as an AI|in general|typically|usually)\b", re.I),
    re.compile(r"\b(Wikipedia|Google|according to studies)\b", re.I),
]


@dataclass
class GuardResult:
    """Result of the hallucination checks."""

    passed: bool
    reason: str = ""
    hallucination_detected: bool = False


class HallucinationGuard:
    """Applies rule-based validation on answers before returning them."""

    def check(
        self,
        answer: str,
        context: str,
        retrieved_scores: list[float] | None = None,
    ) -> GuardResult:
        """Verify the response does not contain obvious hallucination signals."""
        if NOT_FOUND_PHRASE in answer:
            return GuardResult(passed=True, reason="Response indicates info not found.")

        # Empty context but positive response is highly suspicious
        if not context.strip():
            if NOT_FOUND_PHRASE not in answer:
                return GuardResult(
                    passed=False,
                    reason="Answer generated but context is empty.",
                    hallucination_detected=True,
                )
            return GuardResult(passed=True)

        # Hedging or general phrasing
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern.search(answer):
                return GuardResult(
                    passed=False,
                    reason=f"Suspicious hedging or general phrasing detected: '{pattern.pattern}'",
                    hallucination_detected=True,
                )

        # Number mismatch: check if numbers mentioned in answer are present in the context
        numbers_in_answer = set(re.findall(r"\d+(?:\.\d+)?%?", answer))
        if numbers_in_answer:
            unsupported_numbers = [
                n for n in numbers_in_answer if n not in context and f"{n}%" not in context
            ]
            # If more than 50% of the numbers or at least 2 numbers in the answer are not in context, flag it
            if len(unsupported_numbers) > len(numbers_in_answer) * 0.5 and len(unsupported_numbers) >= 2:
                return GuardResult(
                    passed=False,
                    reason=f"Numerical claims {unsupported_numbers} in answer are not supported by the context.",
                    hallucination_detected=True,
                )

        return GuardResult(passed=True, reason="Heuristic checks passed.")
