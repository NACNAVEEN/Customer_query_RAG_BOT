"""Query rewriting to disambiguate short or abstract queries."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.preprocessing.intent_classifier import Intent, IntentResult

logger = logging.getLogger(__name__)

AMBIGUOUS_PATTERNS = [
    (re.compile(r"^how does it work\??$", re.I), "How does InstaParkAI's parking system work?"),
    (re.compile(r"^tell me pricing\??$", re.I), "What are InstaParkAI pricing and contract models?"),
    (re.compile(r"^(what are the )?charges\??$", re.I), "What are the pricing plans and subscription charges for InstaParkAI?"),
    (re.compile(r"^how to set up\??$", re.I), "What are the installation steps and setup flow for InstaParkAI hardware and software?"),
    (re.compile(r"^architecture\??$", re.I), "What is the technical system architecture and data flow of InstaParkAI?"),
    (re.compile(r"^anpr\??$", re.I), "How does the ANPR (Automatic Number Plate Recognition) system work in InstaParkAI?"),
    (re.compile(r"^rfid\??$", re.I), "What is the role and process flow of RFID integration in InstaParkAI?"),
]

INTENT_REWRITE_PREFIX: dict[Intent, str] = {
    Intent.FAQ: "Frequently asked question about",
    Intent.COMPANY_INFORMATION: "Information regarding the company",
    Intent.PRODUCT_INFORMATION: "Product features and options for",
    Intent.PRICING: "Cost, pricing plans, and subscription details for",
    Intent.TECHNOLOGY: "Technical details, API, or algorithm for",
    Intent.PROCESS_FLOW: "Workflow, steps, or installation process for",
}


@dataclass
class RewrittenQuery:
    """Rewritten query data."""

    original: str
    rewritten: str
    was_rewritten: bool


class QueryRewriter:
    """Enriches query details based on intent and ambiguity rules."""

    def rewrite(self, query: str, intent_result: IntentResult | None = None) -> RewrittenQuery:
        """Rewrite query if it is overly short or ambiguous."""
        stripped = query.strip()

        # Check explicit ambiguous patterns
        for pattern, replacement in AMBIGUOUS_PATTERNS:
            if pattern.match(stripped):
                logger.info("Rewrote ambiguous query: %s -> %s", query, replacement)
                return RewrittenQuery(
                    original=query,
                    rewritten=replacement,
                    was_rewritten=True,
                )

        # Contextual prefix appending for very short queries
        if intent_result and len(stripped.split()) <= 2:
            intent = intent_result.intent
            if intent in INTENT_REWRITE_PREFIX and intent_result.confidence >= 0.5:
                prefix = INTENT_REWRITE_PREFIX[intent]
                rewritten = f"{prefix} {stripped}"
                logger.debug("Rewrote short query based on intent: %s", rewritten)
                return RewrittenQuery(
                    original=query,
                    rewritten=rewritten,
                    was_rewritten=True,
                )

        return RewrittenQuery(original=query, rewritten=stripped, was_rewritten=False)
