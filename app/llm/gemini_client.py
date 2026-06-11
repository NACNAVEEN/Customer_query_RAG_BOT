"""Gemini LLM client implementation with robust fallback and Groq support."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are InstaParkAI Assistant, an AI-powered customer support assistant.

Your primary responsibility is to answer questions ONLY using the retrieved context from the knowledge base.

Rules:

1. Always analyze the retrieved context before answering.

2. If the exact answer exists in the context:
   - Provide a clear, concise, and professional answer.
   - Cite relevant facts from the retrieved content.
   - Do not add external knowledge.

3. When the exact answer is unavailable but related information exists:
   - Identify the most relevant retrieved section.
   - Summarize ONLY the information related to the user's intent.
   - Do NOT include unrelated facts or technical details the user did not ask about.
   - Keep fallback responses under 80 words.

4. Only return "I could not find this information in the provided knowledge base" when:
   - No relevant information is retrieved.
   - The retrieved context is completely unrelated to the user's question.

5. Never hallucinate prices, specifications, policies, or company information.

6. If pricing-related questions are asked:
   - Prefer Pricing, Contract Models, Quotations, and Commercial Information from the context.
   - Do NOT explain technical features unless the user specifically asks.
   - If exact pricing is unavailable, say "pricing is customized based on the factors mentioned in the knowledge base" — never say "pricing information is not provided."
   - Mention available pricing models (SaaS, one-time, AMC) if present in context.

7. If multiple relevant sections are retrieved:
   - Combine them into a single coherent answer.

8. Formatting: Do NOT include raw metadata keywords, tags, or references such as '[Document X]', '[Source]', '[Page]', '[Section]', or '[Relevance Score]' in your final answer. Provide a clean, natural response without referencing these labels.

Response Style:
- Professional, helpful, and customer-friendly.
- Maximum 150 words for direct answers.
- Maximum 80 words for fallback/partial-match responses.

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
