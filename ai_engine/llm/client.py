"""
SENTRY — LLM Client
=====================
Wraps the OpenAI GPT-4 API with two distinct generation modes:

    1. baseline_generate(query)
       Raw LLM response with no retrieval grounding.
       Used as the CONTROL condition in the evaluation study.
       Represents what a standard chatbot would produce.

    2. grounded_generate(query, context_chunks)
       RAG-augmented response where retrieved document chunks
       are injected into the prompt as verified context.
       Used as the EXPERIMENTAL condition in the evaluation study.
       This is SENTRY's core contribution.

Keeping both modes in one client ensures the only variable
between conditions is the presence/absence of retrieved context —
model, temperature, and token limits are identical.

Used by: ai_engine/rag/pipeline.py
"""

import time
from typing import List, Dict, Any

from openai import OpenAI

from config.settings import settings


# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """You are SENTRY, an AI cybersecurity training \
assistant deployed on a Pepper humanoid robot. You help employees at Small \
and Medium Enterprises understand cybersecurity threats and safe practices.

Answer the user's question clearly and accurately. Be concise and practical. \
If you are uncertain about something, say so rather than guessing."""

GROUNDED_SYSTEM_PROMPT = """You are SENTRY, an AI cybersecurity training \
assistant deployed on a Pepper humanoid robot. You help employees at Small \
and Medium Enterprises understand cybersecurity threats and safe practices.

You will be given VERIFIED CONTEXT retrieved from authoritative cybersecurity \
sources including the OWASP Top 10 and Kenyan cybersecurity legislation. \
You MUST base your response on this context.

Rules:
- Only use information present in the provided context.
- If the context does not contain enough information to answer, say so clearly.
- Do not introduce facts, statistics, or claims not present in the context.
- When referencing legal content, mention the relevant Act by name.
- Be concise, practical, and appropriate for a non-technical employee audience."""


class LLMClient:
    """
    OpenAI GPT-4 wrapper for SENTRY's two generation modes.
    """

    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise EnvironmentError(
                "[LLMClient] OPENAI_API_KEY is not set. Check your .env file."
            )
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.LLM_MODEL
        self._max_tokens = settings.LLM_MAX_TOKENS
        self._temperature = settings.LLM_TEMPERATURE
        print(f"[LLMClient] Initialised. Model: {self._model}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def baseline_generate(self, query: str) -> Dict[str, Any]:
        """
        Generate a response using only the LLM's parametric knowledge.
        No retrieval. No grounding. This is the control condition.

        Args:
            query: The user's question.

        Returns:
            Dict containing:
                - "response":       The generated text.
                - "mode":           "baseline"
                - "model":          Model name used.
                - "latency_ms":     Response time in milliseconds.
                - "prompt_tokens":  Tokens used in the prompt.
                - "completion_tokens": Tokens used in the completion.
        """
        if not query or not query.strip():
            raise ValueError("[LLMClient] Query cannot be empty.")

        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        return self._call_api(messages=messages, mode="baseline")

    def grounded_generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a response grounded in retrieved document chunks.
        This is the experimental condition — SENTRY's RAG output.

        Args:
            query:          The user's question.
            context_chunks: Retrieved chunks from the Retriever,
                            each with "text", "source", "doc_type", "score".

        Returns:
            Dict containing:
                - "response":           The grounded generated text.
                - "mode":               "grounded"
                - "model":              Model name used.
                - "latency_ms":         Response time in milliseconds.
                - "prompt_tokens":      Tokens used in the prompt.
                - "completion_tokens":  Tokens used in the completion.
                - "sources_used":       List of source documents cited.
        """
        if not query or not query.strip():
            raise ValueError("[LLMClient] Query cannot be empty.")

        if not context_chunks:
            raise ValueError(
                "[LLMClient] context_chunks cannot be empty for grounded generation. "
                "Use baseline_generate() if no retrieval context is available."
            )

        # Build the grounded prompt with injected context
        context_block = self._build_context_block(context_chunks)
        user_message = (
            f"VERIFIED CONTEXT:\n{context_block}\n\n"
            f"USER QUESTION:\n{query}"
        )

        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        result = self._call_api(messages=messages, mode="grounded")

        # Append sources metadata for traceability
        result["sources_used"] = list({
            c["source"] for c in context_chunks
        })

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        mode: str,
    ) -> Dict[str, Any]:
        """
        Make the OpenAI API call with timing and token tracking.
        """
        start_time = time.time()

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        response_text = completion.choices[0].message.content.strip()
        usage = completion.usage

        return {
            "response": response_text,
            "mode": mode,
            "model": self._model,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }

    def _build_context_block(
        self, chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved chunks into a numbered context block
        for injection into the grounded prompt.

        Each chunk is labelled with its source document so the
        LLM can reference it — this also enables source attribution
        on Pepper's tablet display.
        """
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            doc_type = chunk.get("doc_type", "unknown")
            text = chunk.get("text", "").strip()
            score = chunk.get("score", 0.0)

            lines.append(
                f"[{i}] Source: {source} ({doc_type}) | "
                f"Relevance: {score}\n{text}"
            )

        return "\n\n".join(lines)