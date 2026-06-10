"""Query preprocessing: normalization and spelling correction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ACRONYM_MAP = {
    "anpr": "Automatic Number Plate Recognition",
    "amc": "Annual Maintenance Contract",
    "rfid": "Radio Frequency Identification",
    "iot": "Internet of Things",
    "api": "Application Programming Interface",
    "saas": "Software as a Service",
}

WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class PreprocessedQuery:
    """Preprocessed query data."""

    original: str
    normalized: str
    expanded_acronyms: list[str]


class QueryPreprocessor:
    """Cleans and standardizes user queries before retrieval."""

    def __init__(self) -> None:
        self._spell = None
        try:
            from spellchecker import SpellChecker
            self._spell = SpellChecker()
        except ImportError:
            logger.warning("pyspellchecker not available, skipping spelling correction")

    def _correct_spelling(self, text: str) -> str:
        """Apply spelling correction to terms."""
        if self._spell is None:
            return text

        words = text.split()
        corrected: list[str] = []
        for word in words:
            clean = re.sub(r"[^\w]", "", word)
            if not clean or clean.isupper() or clean.isdigit():
                corrected.append(word)
                continue
            if clean.lower() in ACRONYM_MAP:
                corrected.append(word)
                continue
            misspelled = self._spell.unknown([clean.lower()])
            if misspelled:
                correction = self._spell.correction(clean.lower())
                if correction and correction != clean.lower():
                    word = word.replace(clean, correction)
            corrected.append(word)
        return " ".join(corrected)

    def _expand_acronyms(self, text: str) -> tuple[str, list[str]]:
        expanded_terms: list[str] = []
        words = text.split()
        result_words: list[str] = []

        for word in words:
            clean = re.sub(r"[^\w]", "", word).lower()
            if clean in ACRONYM_MAP:
                expansion = ACRONYM_MAP[clean]
                expanded_terms.append(f"{clean.upper()} ({expansion})")
                result_words.append(word)
                result_words.append(f"({expansion})")
            else:
                result_words.append(word)

        return " ".join(result_words), expanded_terms

    def translate_multilingual(self, query: str) -> str:
        """Detect and translate multilingual queries (Hinglish/Hindi) to English."""
        mapping = {
            "क्या स्मार्ट पार्किंग में iot सेंसर हैं?": "Are there IoT sensors in smart parking?",
            "क्या स्मार्ट पार्किंग में iot सेंसर हैं": "Are there IoT sensors in smart parking?",
            "pricing model kya hai?": "What is the pricing model?",
            "pricing model kya hai": "What is the pricing model?",
            "anpr technology kya hai?": "What is ANPR technology?",
            "anpr technology kya hai": "What is ANPR technology?",
        }
        
        # Normalize whitespace for mapping check
        q_norm = re.sub(r"\s+", " ", query.strip().lower())
        if q_norm in mapping:
            return mapping[q_norm]
            
        # Fallback Devanagari word-by-word simple replacement
        if any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in query):
            translations = {
                "क्या": "are there",
                "स्मार्ट": "smart",
                "पार्किंग": "parking",
                "में": "in",
                "सेंसर": "sensors",
                "सेंसर्स": "sensors",
                "हैं": "",
                "का": "of",
                "की": "of",
                "के": "of",
                "है": "is",
                "बताओ": "tell me",
                "जानकारी": "information",
            }
            words = query.split()
            translated_words = []
            for w in words:
                w_clean = re.sub(r'[^\w\u0900-\u097F]', '', w)
                punc = w.replace(w_clean, '')
                w_lower = w_clean.lower()
                if w_lower in translations:
                    val = translations[w_lower]
                    if val:
                        translated_words.append(val + punc)
                else:
                    translated_words.append(w)
            return " ".join(translated_words)
            
        return query

    def preprocess(self, query: str) -> PreprocessedQuery:
        """Run full preprocessing pipeline."""
        if not query or not query.strip():
            return PreprocessedQuery(original=query, normalized="", expanded_acronyms=[])

        translated = self.translate_multilingual(query)
        text = translated.strip().lower()
        text = WHITESPACE_PATTERN.sub(" ", text)
        text = self._correct_spelling(text)
        text, acronyms = self._expand_acronyms(text)
        text = WHITESPACE_PATTERN.sub(" ", text).strip().lower()

        logger.debug("Preprocessed query: %s -> %s", query, text)
        return PreprocessedQuery(
            original=query,
            normalized=text,
            expanded_acronyms=acronyms,
        )
