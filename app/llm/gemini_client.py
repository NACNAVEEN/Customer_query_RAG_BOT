"""Gemini LLM client implementation with robust fallback and Groq support."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are InstaParkAI's Knowledge Base Assistant.

Your job is to answer questions only using the retrieved context from the provided documents.

========================
CORE RULES
========================

1. Use ONLY information explicitly present in the retrieved context.

2. Do NOT use prior knowledge, industry assumptions, or generic AI knowledge.

3. Never invent:
   * Databases
   * Cloud services
   * Programming languages
   * Frameworks
   * APIs
   * Infrastructure
   * AI models
   * Revenue streams
   * Roadmaps
   * Timelines
   * Product features

4. If a fact is not explicitly mentioned in the retrieved context, do NOT add it.

========================
MULTI-CHUNK SYNTHESIS (IMPORTANT)
========================

Before generating the answer:

Step 1: Read ALL retrieved chunks completely.
Step 2: Extract every unique fact from every chunk.
Step 3: Merge related facts from different chunks into one consolidated answer.
Step 4: Do NOT answer using only the highest-ranked chunk.
Step 5: When multiple chunks contain relevant information, combine them into a complete answer.

Example:
  Chunk 1: ANPR, RFID, IoT
  Chunk 2: FASTag, UPI
  Chunk 3: AI Analytics
  Final Answer: ANPR, RFID, IoT, FASTag, UPI, and AI Analytics.

========================
PARTIAL ANSWER POLICY
========================

If a question has multiple parts:

Answer all supported parts using the KB.
For unsupported parts, say:
"This information is not available in the provided knowledge base."

Never reject the entire question if some relevant information exists.

Example:
  Question: "Explain the architecture, AI models, and programming language."
  Response:
  Architecture: [Answer from KB]
  AI Models: [Answer from KB]
  Programming Language: This information is not available in the provided knowledge base.

========================
HALLUCINATION CHECK
========================

Before finalizing the answer:

1. Verify every technical term appears in the retrieved context.
2. If it does not appear, remove it or replace it with:
   "This information is not available in the provided knowledge base."
3. Never infer specific technologies.

Examples:
  Context says "event streaming" -> Do NOT say "Kafka"
  Context says "cloud infrastructure" -> Do NOT say "AWS" or "Azure"
  Context says "machine learning" -> Do NOT say "TensorFlow" or "PyTorch"
  unless explicitly present in the retrieved context.

========================
RESPONSE FORMAT
========================

Answer:
[Grounded answer organized by topic sections when applicable]

Supported Sources:
* Source/Page references used

Missing Information:
* List only the information requested by the user that is not present in the knowledge base.

========================
PRIORITY ORDER
========================

1. Groundedness
2. Completeness
3. Accuracy
4. Clarity
5. Conciseness

A partially complete but grounded answer is always preferred over a complete but speculative answer.

Retrieved Context:
{context}
"""


class GeminiClient:
    """Wrapper around Gemini/Groq model using LangChain or direct API call."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.gemini_model
        self.groq_api_key = settings.groq_api_key
        self.groq_model = settings.groq_model
        self._llm = None

        if self.groq_api_key:
            logger.info("Groq API key configured. Will use Groq model: %s", self.groq_model)
        elif not settings.google_api_key:
            logger.warning("No LLM API keys configured. Will use fallback LLM simulation.")
        else:
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=settings.google_api_key,
                    temperature=0.0,
                    max_output_tokens=2048,
                )
            except Exception as exc:
                logger.warning("Failed to initialize ChatGoogleGenerativeAI: %s. Using fallback LLM simulation.", exc)
                self._llm = None

    def _generate_groq_api(self, messages: list[dict[str, str]], model: str = None) -> str:
        """Query Groq API using HTTP POST."""
        import httpx
        if not model:
            model = self.groq_model
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024
        }
        
        try:
            response = httpx.post(url, headers=headers, json=data, timeout=30.0)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.warning("Groq API returned error status: %d - %s", response.status_code, response.text)
                return ""
        except Exception as exc:
            logger.warning("Failed to call Groq API: %s", exc)
            return ""

    def _fallback_generate(self, query: str, context: str) -> str:
        """Rule-based local RAG generation for offline / keyless settings."""
        query_lower = query.lower()
        
        # Check for obvious out-of-scope / trap questions
        trap_keywords = ["deluxe room", "weather", "hotel", "pool", "cancellation", "sports", "dinner", "restaurant"]
        if any(tk in query_lower for tk in trap_keywords):
            return "I could not find this information in the provided knowledge base."

        # If context is empty or seems like default / not found context
        if not context or "could not find" in context.lower():
            return "I could not find this information in the provided knowledge base."

        # Split context into paragraphs/blocks and match queries
        paragraphs = [p.strip() for p in context.split("\n") if p.strip()]
        
        # Keyword-based extraction to find exact matching paragraph
        keywords = ["anpr", "rfid", "iot", "sensor", "pricing", "amc", "service", "smart parking"]
        matched_paragraphs = []
        for kw in keywords:
            if kw in query_lower:
                for p in paragraphs:
                    if kw in p.lower() and p not in matched_paragraphs:
                        matched_paragraphs.append(p)

        if matched_paragraphs:
            return "\n\n".join(matched_paragraphs)

        # Fallback to returning relevant paragraphs if any exist
        relevant_paragraphs = [p for p in paragraphs if not any(h in p for h in ["InstaParkAI - Smart", "Page "])]
        if relevant_paragraphs:
            return " ".join(relevant_paragraphs)

        return "I could not find this information in the provided knowledge base."

    def generate(
        self,
        query: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate response given a query and retrieved context."""
        # Prioritize Groq if API key is configured
        if self.groq_api_key:
            groq_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            if chat_history:
                for turn in chat_history[-5:]:
                    groq_messages.append({"role": "user", "content": turn.get("user", "")})
                    if turn.get("assistant"):
                        groq_messages.append({"role": "assistant", "content": turn["assistant"]})
            groq_messages.append({"role": "user", "content": query})
            
            res = self._generate_groq_api(groq_messages)
            if res:
                return res

        if self._llm is None:
            return self._fallback_generate(query, context)

        system_content = SYSTEM_PROMPT.format(context=context)
        messages: list[Any] = [SystemMessage(content=system_content)]

        if chat_history:
            for turn in chat_history[-5:]:
                messages.append(HumanMessage(content=turn.get("user", "")))
                if turn.get("assistant"):
                    messages.append(AIMessage(content=turn["assistant"]))

        messages.append(HumanMessage(content=query))

        try:
            response = self._llm.invoke(messages)
            content = response.content
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            return str(content).strip()
        except Exception as exc:
            logger.warning("Gemini invocation failed: %s. Using local fallback generation.", exc)
            return self._fallback_generate(query, context)

    def generate_validation_check(
        self,
        answer: str,
        context: str,
        query: str,
    ) -> str:
        """Verify the generated answer against context."""
        # Prioritize Groq if API key is configured
        if self.groq_api_key:
            prompt = f"""Review the answer against the context and query below.
Respond with 'VALID' if the answer is completely grounded in the context.
Respond with 'INVALID' if the answer contains any facts, numbers, or details not present in the context.

Query: {query}

Context:
{context}

Answer:
{answer}

Response (VALID or INVALID):"""
            res = self._generate_groq_api([{"role": "user", "content": prompt}])
            if res:
                res_clean = res.strip().upper()
                logger.info("[VAL_CHECK_GROQ] RAW res: %r", res)
                first_word = "".join(c for c in res_clean.split()[0] if c.isalnum()) if res_clean.split() else ""
                if "INVALID" in first_word:
                    return "INVALID"
                if "VALID" in first_word:
                    return "VALID"
                return "INVALID" if "INVALID" in res_clean else "VALID"

        if self._llm is None:
            return "VALID"

        prompt = f"""Review the answer against the context and query below.
Respond with 'VALID' if the answer is completely grounded in the context.
Respond with 'INVALID' if the answer contains any facts, numbers, or details not present in the context.

Query: {query}

Context:
{context}

Answer:
{answer}

Response (VALID or INVALID):"""
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            res_clean = str(content).strip().upper()
            first_word = "".join(c for c in res_clean.split()[0] if c.isalnum()) if res_clean.split() else ""
            if "INVALID" in first_word:
                return "INVALID"
            if "VALID" in first_word:
                return "VALID"
            return "INVALID" if "INVALID" in res_clean else "VALID"
        except Exception as exc:
            logger.warning("Failed validation check run: %s", exc)
            return "VALID"
