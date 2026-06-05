import time
from typing import Dict, Any

from ai_engine.generation.client import LLMClient
from ai_engine.generation.prompt_builder import PromptBuilder
from ai_engine.retrieval.retriever import Retriever


class RAGPipeline:
    def __init__(self) -> None:
        print("[RAGPipeline] Initialising components...")
        self._retriever = Retriever()
        self._prompt_builder = PromptBuilder()
        self._llm = LLMClient()
        print("[RAGPipeline] Ready.")

    # Public API

    def query_grounded(
        self,
        query: str,
        doc_type_filter: str = None,
    ) -> Dict[str, Any]:
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
        llm_result = self._llm.grounded_generate_from_prompt(
            user_message=prompt_data["user_message"],
            sources_used=prompt_data["sources"],
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

    # Internal helpers

    def _no_results_response(self, query: str, retrieval_ms: float) -> Dict[str, Any]:
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
