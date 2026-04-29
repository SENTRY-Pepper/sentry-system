"""
SENTRY — Prompt Builder Test
Run: python tests/unit/test_prompt_builder.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_engine.rag.prompt_builder import PromptBuilder


def test_grounded_prompt():
    print("=== Test 1: Grounded Prompt Assembly ===")
    builder = PromptBuilder()

    chunks = [
        {
            "text": (
                "Section 23 of the Computer Misuse and Cybercrimes Act (2018) "
                "states that any person who fraudulently causes a computer to "
                "perform any function with intent to gain unauthorized access "
                "commits an offence liable to a fine not exceeding five million "
                "shillings or imprisonment not exceeding three years."
            ),
            "source": "Computer-Misuse-and-Cybercrimes-Act.pdf",
            "doc_type": "legal",
            "score": 0.87,
        },
        {
            "text": (
                "Phishing attacks deceive users into revealing sensitive "
                "information. Employees should verify sender addresses, avoid "
                "clicking suspicious links, and report suspected phishing "
                "to their IT team immediately."
            ),
            "source": "A07_2025-Authentication_Failures.md",
            "doc_type": "owasp",
            "score": 0.81,
        },
    ]

    result = builder.build_grounded_prompt(
        query="What is phishing and what are the legal consequences in Kenya?",
        context_chunks=chunks,
    )

    print(f"Chunks used:        {result['chunks_used']}")
    print(f"Chunks truncated:   {result['chunks_truncated']}")
    print(f"Context tokens:     {result['context_tokens']}")
    print(f"Query tokens:       {result['query_tokens']}")
    print(f"Sources:            {result['sources']}")
    print(f"\nContext block:\n{result['context_block']}")
    print(f"\nFull user message:\n{result['user_message']}")

    assert result["chunks_used"] == 2
    assert result["chunks_truncated"] == 0
    assert result["context_tokens"] > 0
    assert result["query_tokens"] > 0
    assert len(result["sources"]) == 2
    assert "VERIFIED CONTEXT:" in result["user_message"]
    assert "USER QUESTION:" in result["user_message"]
    print("\n>>> Grounded prompt test PASSED")


def test_baseline_prompt():
    print("\n=== Test 2: Baseline Prompt Assembly ===")
    builder = PromptBuilder()

    result = builder.build_baseline_prompt("What is SQL injection?")

    print(f"Query tokens:   {result['query_tokens']}")
    print(f"User message:   {result['user_message']}")

    assert result["user_message"] == "What is SQL injection?"
    assert result["query_tokens"] > 0
    print(">>> Baseline prompt test PASSED")


def test_token_budget_enforcement():
    print("\n=== Test 3: Token Budget Enforcement ===")
    builder = PromptBuilder()

    # Create a chunk that is deliberately huge
    huge_chunk = {
        "text": "This is a cybersecurity concept. " * 300,
        "source": "big-document.pdf",
        "doc_type": "owasp",
        "score": 0.9,
    }
    small_chunk = {
        "text": "Short relevant fact about SQL injection.",
        "source": "A05_2025-Injection.md",
        "doc_type": "owasp",
        "score": 0.7,
    }

    result = builder.build_grounded_prompt(
        query="Tell me about cybersecurity.",
        context_chunks=[huge_chunk, small_chunk],
    )

    print(f"Chunks used:      {result['chunks_used']}")
    print(f"Chunks truncated: {result['chunks_truncated']}")
    print(f"Context tokens:   {result['context_tokens']}")

    assert result["context_tokens"] <= PromptBuilder.CONTEXT_TOKEN_BUDGET
    print(">>> Token budget test PASSED")


if __name__ == "__main__":
    test_grounded_prompt()
    test_baseline_prompt()
    test_token_budget_enforcement()
    print("\n" + "=" * 60)
    print("All prompt builder tests PASSED")
