from typing import List, Dict, Any
import tiktoken
from config.settings import settings


class PromptBuilder:
    CONTEXT_TOKEN_BUDGET: int = settings.RAG_CONTEXT_TOKEN_BUDGET

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(settings.TIKTOKEN_ENCODING)

    # Public API

    def build_grounded_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("[PromptBuilder] Query cannot be empty.")
        if not context_chunks:
            raise ValueError("[PromptBuilder] context_chunks cannot be empty.")

        fitted_chunks, truncated_count = self._fit_chunks_to_budget(context_chunks)

        context_block = self._format_context_block(fitted_chunks)

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
        if not query or not query.strip():
            raise ValueError("[PromptBuilder] Query cannot be empty.")

        return {
            "user_message": query.strip(),
            "query_tokens": self._count_tokens(query),
        }

    # Internal helpers

    def _fit_chunks_to_budget(
        self,
        chunks: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int]:
        fitted = []
        tokens_used = 0

        candidate_chunks = chunks[: settings.RAG_MAX_CONTEXT_CHUNKS]

        for chunk in candidate_chunks:
            chunk_tokens = self._count_tokens(chunk["text"])
            if tokens_used + chunk_tokens <= self.CONTEXT_TOKEN_BUDGET:
                fitted.append(chunk)
                tokens_used += chunk_tokens
            else:
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
