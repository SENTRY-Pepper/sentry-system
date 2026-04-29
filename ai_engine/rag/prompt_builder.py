"""
SENTRY — Prompt Builder
=========================
Assembles the grounded prompt sent to the LLM during RAG-augmented
generation. Separating this logic from the LLM client means:

    - Prompt structure can be tested independently of API calls
    - Prompt engineering decisions are clearly documented
    - The evaluation study can inspect exactly what the LLM received
    - Future prompt variations (e.g. chain-of-thought) can be added
      without touching the LLM client

Prompt structure (grounded mode):
    [System prompt]
        SENTRY persona + grounding rules

    [User message]
        VERIFIED CONTEXT:
            [1] Source: filename (doc_type) | Relevance: score
                <chunk text>
            [2] ...
            [N] ...

        USER QUESTION:
            <query>

Used by: ai_engine/rag/pipeline.py
"""

from typing import List, Dict, Any
import tiktoken
from config.settings import settings


class PromptBuilder:
    """
    Builds structured prompts for both generation modes.

    Responsibilities:
        - Format retrieved chunks into a numbered, labelled context block
        - Enforce a token budget so the total prompt stays within
          GPT-4's context window
        - Expose the assembled prompt for inspection and logging
          (important for evaluation traceability)
    """

    # GPT-4 context window is 8192 tokens.
    # We reserve tokens for: system prompt (~200), response (~1024),
    # and the user question (~100). The remainder is the context budget.
    CONTEXT_TOKEN_BUDGET: int = 6500

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(settings.TIKTOKEN_ENCODING)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_grounded_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assemble the grounded user message with retrieved context.

        Args:
            query:          The user's question.
            context_chunks: Retrieved chunks (text, source, doc_type, score).

        Returns:
            Dict containing:
                - "user_message":       The full formatted user message string
                                        (context block + question).
                - "context_block":      Just the context section (for logging).
                - "chunks_used":        Number of chunks that fit in the budget.
                - "chunks_truncated":   Number of chunks dropped due to budget.
                - "context_tokens":     Token count of the context block.
                - "query_tokens":       Token count of the query.
                - "sources":            List of source document names included.
        """
        if not query or not query.strip():
            raise ValueError("[PromptBuilder] Query cannot be empty.")
        if not context_chunks:
            raise ValueError("[PromptBuilder] context_chunks cannot be empty.")

        # Fit as many chunks as possible within the token budget
        fitted_chunks, truncated_count = self._fit_chunks_to_budget(context_chunks)

        # Build the formatted context block
        context_block = self._format_context_block(fitted_chunks)

        # Assemble the full user message
        user_message = (
            f"VERIFIED CONTEXT:\n"
            f"{context_block}\n\n"
            f"USER QUESTION:\n{query.strip()}"
        )

        context_tokens = self._count_tokens(context_block)
        query_tokens = self._count_tokens(query)
        sources = list({c["source"] for c in fitted_chunks})

        return {
            "user_message": user_message,
            "context_block": context_block,
            "chunks_used": len(fitted_chunks),
            "chunks_truncated": truncated_count,
            "context_tokens": context_tokens,
            "query_tokens": query_tokens,
            "sources": sources,
        }

    def build_baseline_prompt(self, query: str) -> Dict[str, Any]:
        """
        Assemble the baseline user message — no context, raw query only.
        Kept here so both prompt types are built in one place.

        Args:
            query: The user's question.

        Returns:
            Dict containing:
                - "user_message": The plain query string.
                - "query_tokens": Token count of the query.
        """
        if not query or not query.strip():
            raise ValueError("[PromptBuilder] Query cannot be empty.")

        return {
            "user_message": query.strip(),
            "query_tokens": self._count_tokens(query),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_chunks_to_budget(
        self,
        chunks: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Greedily include chunks in relevance order until the token
        budget is exhausted. Chunks are already ranked by the retriever
        (most relevant first), so we preserve that order.

        Returns:
            (fitted_chunks, truncated_count)
        """
        fitted = []
        tokens_used = 0

        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk["text"])
            if tokens_used + chunk_tokens <= self.CONTEXT_TOKEN_BUDGET:
                fitted.append(chunk)
                tokens_used += chunk_tokens
            else:
                # Once budget is hit, remaining chunks are dropped
                break

        truncated = len(chunks) - len(fitted)
        if truncated > 0:
            print(
                f"[PromptBuilder] Budget limit reached — "
                f"{truncated} chunk(s) truncated from context."
            )

        return fitted, truncated

    def _format_context_block(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Format chunks into a numbered, labelled block.

        Format per chunk:
            [N] Source: filename (doc_type) | Relevance: 0.xx
            <chunk text>
        """
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            doc_type = chunk.get("doc_type", "unknown")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            header = f"[{i}] Source: {source} ({doc_type}) " f"| Relevance: {score}"
            lines.append(f"{header}\n{text}")

        return "\n\n".join(lines)

    def _count_tokens(self, text: str) -> int:
        """Return the token count for a string."""
        return len(self._encoding.encode(text))
