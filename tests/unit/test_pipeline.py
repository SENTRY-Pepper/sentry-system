"""
SENTRY — RAG Pipeline Test
Full end-to-end test of both pipeline modes against the live
ChromaDB knowledge base and GPT-4 API.
Run: python tests/unit/test_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_engine.rag.pipeline import RAGPipeline


def print_result(result: dict) -> None:
    """Pretty-print a pipeline result."""
    print(f"  Query:              {result['query']}")
    print(f"  Mode:               {result['mode']}")
    print(f"  Model:              {result['model']}")
    print(f"  Sources:            {result['sources']}")
    print(f"  Chunks used:        {result.get('chunks_used', 'N/A')}")
    print(f"  Context tokens:     {result.get('context_tokens', 'N/A')}")
    print(f"  Prompt tokens:      {result['prompt_tokens']}")
    print(f"  Completion tokens:  {result['completion_tokens']}")
    print(f"  Retrieval latency:  {result['retrieval_ms']} ms")
    print(f"  Generation latency: {result['generation_ms']} ms")
    print(f"  Total latency:      {result['total_ms']} ms")
    print(f"  Response:\n{result['response']}")


def test_grounded_pipeline():
    print("=== Test 1: Grounded Pipeline (Full RAG) ===")
    pipeline = RAGPipeline()

    result = pipeline.query_grounded(
        "What are the legal penalties for hacking in Kenya?"
    )
    print_result(result)

    assert result["mode"] == "grounded"
    assert result["response"] != ""
    assert len(result["sources"]) > 0
    assert len(result["retrieved_chunks"]) > 0
    assert result["retrieval_ms"] > 0
    assert result["total_ms"] < 15000
    print("\n>>> Grounded pipeline test PASSED")


def test_baseline_pipeline():
    print("\n=== Test 2: Baseline Pipeline (No RAG) ===")
    pipeline = RAGPipeline()

    result = pipeline.query_baseline(
        "What are the legal penalties for hacking in Kenya?"
    )
    print_result(result)

    assert result["mode"] == "baseline"
    assert result["response"] != ""
    assert result["sources"] == []
    assert result["retrieved_chunks"] == []
    assert result["retrieval_ms"] == 0
    print("\n>>> Baseline pipeline test PASSED")


def test_same_query_both_modes():
    """
    Run the same query through both modes and compare.
    This mirrors exactly what the evaluation study does.
    """
    print("\n=== Test 3: Same Query — Both Modes Side by Side ===")
    pipeline = RAGPipeline()
    query = "How should an employee handle a suspicious USB drive?"

    grounded = pipeline.query_grounded(query)
    baseline = pipeline.query_baseline(query)

    print(f"\nQuery: {query}")
    print(f"\n--- BASELINE response ({baseline['total_ms']} ms) ---")
    print(baseline["response"])
    print(f"\n--- GROUNDED response ({grounded['total_ms']} ms) ---")
    print(grounded["response"])
    print(f"\nGrounded sources: {grounded['sources']}")

    assert grounded["mode"] == "grounded"
    assert baseline["mode"] == "baseline"
    assert grounded["response"] != baseline["response"]
    print("\n>>> Side-by-side comparison test PASSED")


if __name__ == "__main__":
    test_grounded_pipeline()
    print()
    test_baseline_pipeline()
    print()
    test_same_query_both_modes()
    print("\n" + "=" * 60)
    print("All pipeline tests PASSED")