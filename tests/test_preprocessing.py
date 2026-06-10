"""Tests for query preprocessing and intent classification."""

from app.preprocessing.intent_classifier import Intent, IntentClassifier
from app.preprocessing.query_preprocessor import QueryPreprocessor
from app.preprocessing.query_rewriter import QueryRewriter


class TestQueryPreprocessor:
    def test_lowercase_normalization(self) -> None:
        preprocessor = QueryPreprocessor()
        result = preprocessor.preprocess("What Is ANPR?")
        assert result.normalized == result.normalized.lower()

    def test_acronym_expansion(self) -> None:
        preprocessor = QueryPreprocessor()
        result = preprocessor.preprocess("explain anpr and rfid")
        assert "automatic number plate recognition" in result.normalized
        assert "radio frequency identification" in result.normalized
        assert len(result.expanded_acronyms) >= 2

    def test_whitespace_cleanup(self) -> None:
        preprocessor = QueryPreprocessor()
        result = preprocessor.preprocess("  what   is   amc  ")
        assert "  " not in result.normalized


class TestIntentClassifier:
    def test_pricing_intent(self) -> None:
        classifier = IntentClassifier()
        result = classifier.classify("What is the pricing model?")
        assert result.intent == Intent.PRICING
        assert result.confidence > 0.3

    def test_technology_intent(self) -> None:
        classifier = IntentClassifier()
        result = classifier.classify("How does ANPR technology work?")
        assert result.intent == Intent.TECHNOLOGY

    def test_out_of_scope(self) -> None:
        classifier = IntentClassifier()
        result = classifier.classify("What is the weather today?")
        assert result.intent == Intent.OUT_OF_SCOPE


class TestQueryRewriter:
    def test_ambiguous_rewrite(self) -> None:
        rewriter = QueryRewriter()
        result = rewriter.rewrite("How does it work?")
        assert result.was_rewritten
        assert "InstaParkAI" in result.rewritten

    def test_pricing_rewrite(self) -> None:
        rewriter = QueryRewriter()
        result = rewriter.rewrite("Tell me pricing")
        assert result.was_rewritten
        assert "pricing" in result.rewritten.lower()
