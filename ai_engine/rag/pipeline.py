"""
SENTRY — RAG Pipeline Orchestrator
=====================================
The central coordinator of SENTRY's AI engine. Wires together:

    Retriever → PromptBuilder → LLMClient

into a single end-to-end pipeline that the middleware calls for
every incoming query from the Pepper robot.

Two public methods mirror the two evaluation conditions:

    query_grounded(query)   — EXPERIMENTAL: full RAG pipeline
        1. Embed the query
        2. Retrieve top-k relevant chunks from ChromaDB
        3. Build a grounded prompt with retrieved context
        4. Generate a response constrained to that context
        5. Return response + full traceability metadata

    query_baseline(query)   — CONTROL: LLM only, no retrieval
        1. Build a plain prompt with no context
        2. Generate a response from GPT-4's parametric memory only
        3. Return response + metadata

The metadata returned by both methods is designed to feed directly
into the evaluation module (hallucination scoring, grounding accuracy,
latency tracking) without any post-processing.

Used by: middleware/routes/query_routes.py
"""

import time
from typing import Dict, Any

from ai_engine.rag.retriever import Retriever
from ai_engine.rag.prompt_builder import PromptBuilder
from ai_engine.llm.client import LLMClient


class RAGPipeline:
    """
    End-to-end RAG pipeline for SENTRY.

    Initialised once at middleware startup and reused across
    all requests — model loading and ChromaDB connection are
    expensive and should not be repeated per query.
    """

    def __init__(self) -> None:
        print("[RAGPipeline] Initialising components...")
        self._retriever = Retriever()
        self._prompt_builder = PromptBuilder()
        self._llm = LLMClient()
        print("[RAGPipeline] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_grounded(
        self,
        query: str,
        doc_type_filter: str = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline — experimental condition.

        Retrieves relevant chunks, builds a grounded prompt,
        generates a response constrained to retrieved context.

        Args:
            query:           The user's question from Pepper.
            doc_type_filter: Optional — restrict retrieval to
                             "owasp" or "legal" sources only.

        Returns:
            Structured result dict for the middleware and evaluator:
                - "query":              Original question.
                - "mode":               "grounded"
                - "response":           LLM-generated grounded answer.
                - "sources":            Documents used in context.
                - "retrieved_chunks":   Full chunk list with scores.
                - "chunks_used":        Chunks that fit in prompt budget.
                - "chunks_truncated":   Chunks dropped due to budget.
                - "context_tokens":     Tokens used by context block.
                - "query_tokens":       Tokens used by the query.
                - "prompt_tokens":      Total tokens sent to LLM.
                - "completion_tokens":  Tokens in LLM response.
                - "retrieval_ms":       Time spent on retrieval.
                - "generation_ms":      Time spent on LLM call.
                - "total_ms":           End-to-end latency.
                - "model":              LLM model used.
        """
        if not query or not query.strip():
            raise ValueError("[RAGPipeline] Query cannot be empty.")

        pipeline_start = time.time()

        # Step 1: Retrieve
        retrieval_start = time.time()
        chunks = self._retriever.retrieve(
            query=query,
            doc_type_filter=doc_type_filter,
        )
        retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

        if not chunks:
            return self._no_results_response(query, retrieval_ms)

        # Step 2: Build grounded prompt
        prompt_data = self._prompt_builder.build_grounded_prompt(
            query=query,
            context_chunks=chunks,
        )

        # Step 3: Generate grounded response
        generation_start = time.time()
        llm_result = self._llm.grounded_generate(
            query=query,
            context_chunks=chunks[:prompt_data["chunks_used"]],
        )
        generation_ms = round((time.time() - generation_start) * 1000, 2)

        total_ms = round((time.time() - pipeline_start) * 1000, 2)

        return {
            "query": query,
            "mode": "grounded",
            "response": llm_result["response"],
            "sources": prompt_data["sources"],
            "retrieved_chunks": chunks,
            "chunks_used": prompt_data["chunks_used"],
            "chunks_truncated": prompt_data["chunks_truncated"],
            "context_tokens": prompt_data["context_tokens"],
            "query_tokens": prompt_data["query_tokens"],
            "prompt_tokens": llm_result["prompt_tokens"],
            "completion_tokens": llm_result["completion_tokens"],
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "model": llm_result["model"],
        }

    def query_baseline(self, query: str) -> Dict[str, Any]:
        """
        Baseline pipeline — control condition.

        No retrieval. Sends the raw query directly to the LLM.
        Used to measure hallucination rate without grounding.

        Args:
            query: The user's question.

        Returns:
            Structured result dict:
                - "query":              Original question.
                - "mode":               "baseline"
                - "response":           LLM-generated ungrounded answer.
                - "sources":            Empty list (no retrieval).
                - "retrieved_chunks":   Empty list (no retrieval).
                - "query_tokens":       Tokens used by the query.
                - "prompt_tokens":      Total tokens sent to LLM.
                - "completion_tokens":  Tokens in LLM response.
                - "retrieval_ms":       0 (no retrieval performed).
                - "generation_ms":      Time spent on LLM call.
                - "total_ms":           End-to-end latency.
                - "model":              LLM model used.
        """
        if not query or not query.strip():
            raise ValueError("[RAGPipeline] Query cannot be empty.")

        pipeline_start = time.time()

        prompt_data = self._prompt_builder.build_baseline_prompt(query)

        generation_start = time.time()
        llm_result = self._llm.baseline_generate(query)
        generation_ms = round((time.time() - generation_start) * 1000, 2)

        total_ms = round((time.time() - pipeline_start) * 1000, 2)

        return {
            "query": query,
            "mode": "baseline",
            "response": llm_result["response"],
            "sources": [],
            "retrieved_chunks": [],
            "query_tokens": prompt_data["query_tokens"],
            "prompt_tokens": llm_result["prompt_tokens"],
            "completion_tokens": llm_result["completion_tokens"],
            "retrieval_ms": 0,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "model": llm_result["model"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _no_results_response(
        self, query: str, retrieval_ms: float
    ) -> Dict[str, Any]:
        """
        Fallback when retrieval returns no chunks.
        Returns a safe, honest response rather than hallucinating.
        """
        return {
            "query": query,
            "mode": "grounded",
            "response": (
                "I was unable to find relevant information in my "
                "knowledge base to answer that question. Please consult "
                "your IT security team or a qualified cybersecurity professional."
            ),
            "sources": [],
            "retrieved_chunks": [],
            "chunks_used": 0,
            "chunks_truncated": 0,
            "context_tokens": 0,
            "query_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retrieval_ms": retrieval_ms,
            "generation_ms": 0,
            "total_ms": retrieval_ms,
            "model": "",
        }