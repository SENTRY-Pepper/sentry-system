"""
SENTRY — LLM Client Test
Manual test for baseline and grounded generation modes.
Run: python tests/unit/test_llm_client.py
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_engine.llm.client import LLMClient


@pytest.mark.live
def test_baseline():
    print("=== Test 1: Baseline Generation ===")
    client = LLMClient()

    result = client.baseline_generate(
        "What is phishing and how should an employee respond to a suspicious email?"
    )

    print(f"Mode:               {result['mode']}")
    print(f"Model:              {result['model']}")
    print(f"Latency:            {result['latency_ms']} ms")
    print(f"Prompt tokens:      {result['prompt_tokens']}")
    print(f"Completion tokens:  {result['completion_tokens']}")
    print(f"Response:\n{result['response']}")
    print()

    assert result["mode"] == "baseline"
    assert result["response"] != ""
    assert result["latency_ms"] > 0
    print(">>> Baseline test PASSED")


@pytest.mark.live
def test_grounded():
    print("=== Test 2: Grounded Generation ===")
    client = LLMClient()

    mock_chunks = [
        {
            "text": (
                "Section 23 of the Computer Misuse and Cybercrimes Act (2018) "
                "states that any person who fraudulently and without authority "
                "causes a computer to perform any function with intent to secure "
                "access to any data held in any computer commits an offence and "
                "is liable on conviction to a fine not exceeding five million "
                "shillings or imprisonment for a term not exceeding three years."
            ),
            "source": "Computer-Misuse-and-Cybercrimes-Act.pdf",
            "doc_type": "legal",
            "score": 0.87,
        },
        {
            "text": (
                "Phishing attacks are a form of social engineering where attackers "
                "deceive users into revealing sensitive information such as usernames, "
                "passwords, and credit card numbers. Employees should verify sender "
                "addresses, avoid clicking suspicious links, and report suspected "
                "phishing attempts to their IT team immediately."
            ),
            "source": "A07_2025-Authentication_Failures.md",
            "doc_type": "owasp",
            "score": 0.81,
        },
    ]

    result = client.grounded_generate(
        query="What is phishing and what are the legal consequences in Kenya?",
        context_chunks=mock_chunks,
    )

    print(f"Mode:               {result['mode']}")
    print(f"Model:              {result['model']}")
    print(f"Latency:            {result['latency_ms']} ms")
    print(f"Prompt tokens:      {result['prompt_tokens']}")
    print(f"Completion tokens:  {result['completion_tokens']}")
    print(f"Sources used:       {result['sources_used']}")
    print(f"Response:\n{result['response']}")
    print()

    assert result["mode"] == "grounded"
    assert result["response"] != ""
    assert len(result["sources_used"]) == 2
    assert result["latency_ms"] > 0
    print(">>> Grounded test PASSED")


if __name__ == "__main__":
    test_baseline()
    print("=" * 60)
    print()
    test_grounded()
