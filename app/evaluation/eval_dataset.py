"""Evaluation dataset for InstaParkAI RAG Chatbot.

Contains positive, multilingual, and trap test cases per spec.
"""

from __future__ import annotations

from dataclasses import dataclass

NOT_FOUND_RESPONSE = "I could not find this information in the provided knowledge base."


@dataclass
class EvalCase:
    """Single evaluation test case."""

    id: int
    query: str
    category: str  # positive | multilingual | trap
    expected_intent: str
    expected_contains: list[str]  # phrases the answer should contain
    should_not_find: bool  # True if expected to return NOT_FOUND


# ============================================================
# Evaluation Dataset
# ============================================================

EVAL_DATASET: list[EvalCase] = [
    # --- Positive Cases ---
    EvalCase(
        id=1,
        query="What is ANPR technology?",
        category="positive",
        expected_intent="technology",
        expected_contains=["ANPR", "Automatic Number Plate Recognition"],
        should_not_find=False,
    ),
    EvalCase(
        id=2,
        query="Tell me about the pricing models",
        category="positive",
        expected_intent="pricing",
        expected_contains=["pricing", "model"],
        should_not_find=False,
    ),
    EvalCase(
        id=3,
        query="How do IoT sensors help in parking?",
        category="positive",
        expected_intent="technology",
        expected_contains=["IoT", "sensor"],
        should_not_find=False,
    ),
    EvalCase(
        id=4,
        query="What is the AMC service contract?",
        category="positive",
        expected_intent="faq",
        expected_contains=["AMC", "Annual Maintenance"],
        should_not_find=False,
    ),
    EvalCase(
        id=5,
        query="How does RFID integration work?",
        category="positive",
        expected_intent="technology",
        expected_contains=["RFID", "Radio Frequency Identification"],
        should_not_find=False,
    ),

    # --- Multilingual Cases ---
    EvalCase(
        id=6,
        query="ANPR technology kya hai?",
        category="multilingual",
        expected_intent="technology",
        expected_contains=["ANPR", "Automatic Number Plate Recognition"],
        should_not_find=False,
    ),
    EvalCase(
        id=7,
        query="Pricing model kya hai?",
        category="multilingual",
        expected_intent="pricing",
        expected_contains=["model"],
        should_not_find=False,
    ),
    EvalCase(
        id=8,
        query="क्या स्मार्ट पार्किंग में IoT सेंसर हैं?",
        category="multilingual",
        expected_intent="technology",
        expected_contains=["IoT", "sensor"],
        should_not_find=False,
    ),

    # --- Trap Cases (must NOT hallucinate) ---
    EvalCase(
        id=9,
        query="What is Deluxe Room price?",
        category="trap",
        expected_intent="other",
        expected_contains=["could not find", "knowledge base"],
        should_not_find=True,
    ),
    EvalCase(
        id=10,
        query="What is the weather today?",
        category="trap",
        expected_intent="other",
        expected_contains=["could not find", "knowledge base"],
        should_not_find=True,
    ),
]


def get_eval_dataset() -> list[EvalCase]:
    """Return the full evaluation dataset."""
    return EVAL_DATASET


def get_positive_cases() -> list[EvalCase]:
    return [c for c in EVAL_DATASET if c.category == "positive"]


def get_multilingual_cases() -> list[EvalCase]:
    return [c for c in EVAL_DATASET if c.category == "multilingual"]


def get_trap_cases() -> list[EvalCase]:
    return [c for c in EVAL_DATASET if c.category == "trap"]
