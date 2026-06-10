"""Intent classification for user queries."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Supported intent classes."""

    FAQ = "FAQ"
    COMPANY_INFORMATION = "Company Information"
    PRODUCT_INFORMATION = "Product Information"
    PRICING = "Pricing"
    TECHNOLOGY = "Technology"
    PROCESS_FLOW = "Process Flow"
    GENERAL_CONVERSATION = "General Conversation"
    OUT_OF_SCOPE = "Out Of Scope"


INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.FAQ: [
        r"\bfaq\b",
        r"\bquestions?\b",
        r"\bhow to\b",
        r"\bwhat is the process\b",
        r"\bcan i\b",
        r"\bis it possible\b",
    ],
    Intent.COMPANY_INFORMATION: [
        r"\babout\b",
        r"\bwho is\b",
        r"\binstapark\b",
        r"\bcompany\b",
        r"\bbackground\b",
        r"\bfounder\b",
        r"\bteam\b",
        r"\bhistory\b",
        r"\boffice\b",
        r"\blocation\b",
    ],
    Intent.PRODUCT_INFORMATION: [
        r"\bproduct\b",
        r"\bfeature\b",
        r"\bhow does it work\b",
        r"\banpr\b",
        r"\brfid\b",
        r"\bhardware\b",
        r"\bsoftware\b",
        r"\bsolution\b",
        r"\bapp\b",
        r"\bdashboard\b",
    ],
    Intent.PRICING: [
        r"\bprice\b",
        r"\bcost\b",
        r"\bcharge\b",
        r"\bfee\b",
        r"\bpricing\b",
        r"\bsubscription\b",
        r"\bplan\b",
        r"\bfree trial\b",
        r"\bquote\b",
    ],
    Intent.TECHNOLOGY: [
        r"\btech\b",
        r"\btechnolog(y|ies)\b",
        r"\barchitecture\b",
        r"\balgorithm\b",
        r"\bcamera\b",
        r"\bsensor\b",
        r"\bintegration\b",
        r"\bapi\b",
        r"\bsdk\b",
        r"\bdatabase\b",
        r"\bsecurity\b",
        r"\bencryption\b",
    ],
    Intent.PROCESS_FLOW: [
        r"\bflow\b",
        r"\bprocess\b",
        r"\bstep\b",
        r"\bworkflow\b",
        r"\bsequence\b",
        r"\bprocedure\b",
        r"\bsetup\b",
        r"\binstall\b",
        r"\bdeploy\b",
    ],
    Intent.GENERAL_CONVERSATION: [
        r"\bhello\b",
        r"\bhi\b",
        r"\bgreetings\b",
        r"\bthanks\b",
        r"\bthank you\b",
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\bhelp\b",
    ],
    Intent.OUT_OF_SCOPE: [],
}

CLASSIFICATION_PROMPT = """Classify the user query into exactly one category:
FAQ: general user questions about rules/policies
Company Information: questions about the company's background, team, or location
Product Information: details on features, app, dashboard, hardware, or software
Pricing: prices, subscription plans, costs, or quotes
Technology: API, SDK, database, cameras, sensors, ANPR, or security
Process Flow: installation, setup, or deployment workflows
General Conversation: greetings, thanks, or simple chat
Out Of Scope: questions completely unrelated to smart parking or the company

Return ONLY the category name. Do not include any other text."""


@dataclass
class IntentResult:
    """Intent classification result."""

    intent: Intent
    confidence: float
    scores: dict[str, float]


class IntentClassifier:
    """Classifies user queries to route them effectively."""

    def classify(self, query: str) -> IntentResult:
        """Classify a query using regex patterns."""
        query_lower = query.lower()
        scores: dict[Intent, float] = {intent: 0.0 for intent in Intent}

        for intent, patterns in INTENT_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
            if matches:
                scores[intent] = min(0.5 + matches * 0.15, 0.95)

        # Boost Technology if keyword 'technology' or 'tech' is explicitly present
        if "technology" in query_lower or "tech" in query_lower:
            if scores[Intent.TECHNOLOGY] > 0:
                scores[Intent.TECHNOLOGY] += 0.1
            else:
                scores[Intent.TECHNOLOGY] = 0.6

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]

        has_any_match = any(v > 0.0 for v in scores.values())
        out_of_scope_indicators = [
            r"\bweather\b", r"\bsports?\b", r"\bcricket\b", r"\bfootball\b", 
            r"\brecipes?\b", r"\bpolitics?\b", r"\bpresident\b", r"\bcapital\b",
            r"\bmovies?\b", r"\bsongs?\b", r"\bgames?\b", r"\bnews\b", r"\bjokes?\b"
        ]
        is_explicit_out_of_scope = any(re.search(pat, query_lower) for pat in out_of_scope_indicators)

        if is_explicit_out_of_scope or not has_any_match:
            best_intent = Intent.OUT_OF_SCOPE
            best_score = 0.8 if is_explicit_out_of_scope else 0.5
        elif best_score < 0.3:
            best_intent = Intent.FAQ
            best_score = 0.35

        logger.debug("Regex classified intent: %s (%.2f)", best_intent.value, best_score)
        return IntentResult(
            intent=best_intent,
            confidence=best_score,
            scores={k.value: v for k, v in scores.items()},
        )
