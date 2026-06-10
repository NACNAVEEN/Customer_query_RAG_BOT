"""Tests for response validation and hallucination guard."""

from app.validators.hallucination_guard import HallucinationGuard, NOT_FOUND_PHRASE
from app.validators.response_validator import ResponseValidator


class TestHallucinationGuard:
    def test_not_found_passes(self) -> None:
        guard = HallucinationGuard()
        result = guard.check(NOT_FOUND_PHRASE, "")
        assert result.passed

    def test_suspicious_phrasing_fails(self) -> None:
        guard = HallucinationGuard()
        result = guard.check(
            "I think InstaParkAI probably has many features.",
            "InstaParkAI has smart parking.",
        )
        assert not result.passed
        assert result.hallucination_detected


class TestResponseValidator:
    def test_not_found_valid(self) -> None:
        validator = ResponseValidator()
        result = validator.validate(
            answer=NOT_FOUND_PHRASE,
            context="",
            query="random question",
        )
        assert result.valid

    def test_missing_citations_invalid(self) -> None:
        validator = ResponseValidator()
        result = validator.validate(
            answer="InstaParkAI offers ANPR solutions.",
            context="InstaParkAI offers ANPR solutions for parking.",
            query="What is ANPR?",
            citations=None,
        )
        assert not result.valid
