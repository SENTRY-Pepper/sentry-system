"""
SENTRY — Evaluation Module Test
Run: python tests/unit/test_evaluation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics.hallucination_scorer import HallucinationScorer
from evaluation.metrics.grounding_scorer import GroundingScorer


# Simulate realistic pipeline outputs
GROUNDED_RESULT = {
    "query": "What are the legal penalties for phishing in Kenya?",
    "mode": "grounded",
    "response": (
        "Phishing is an offence under the Computer Misuse and Cybercrimes Act. "
        "A person who fraudulently causes a computer to perform any function "
        "with intent to gain unauthorized access commits an offence. "
        "The penalty is a fine not exceeding five million shillings or "
        "imprisonment not exceeding three years, or both. "
        "Employees should report suspicious emails immediately to their IT team."
    ),
    "retrieved_chunks": [
        {
            "text": (
                "A person who fraudulently causes a computer to perform any "
                "function with intent to gain unauthorized access commits an "
                "offence liable to a fine not exceeding five million shillings "
                "or imprisonment not exceeding three years."
            ),
            "source": "Computer-Misuse-and-Cybercrimes-Act.pdf",
            "doc_type": "legal",
            "score": 0.87,
            "chunk_index": 21,
        }
    ],
    "sources": ["Computer-Misuse-and-Cybercrimes-Act.pdf"],
    "total_ms": 12000,
    "prompt_tokens": 400,
    "completion_tokens": 120,
    "retrieval_ms": 300,
    "generation_ms": 11700,
    "model": "gpt-4",
}

BASELINE_RESULT = {
    "query": "What are the legal penalties for phishing in Kenya?",
    "mode": "baseline",
    "response": (
        "Phishing is a serious cybercrime globally. "
        "In many countries, penalties can include fines and imprisonment. "
        "Organizations should train employees to recognise suspicious emails. "
        "Always verify sender addresses before clicking any links. "
        "Using multi-factor authentication significantly reduces risk."
    ),
    "retrieved_chunks": [],
    "sources": [],
    "total_ms": 8000,
    "prompt_tokens": 80,
    "completion_tokens": 100,
    "retrieval_ms": 0,
    "generation_ms": 8000,
    "model": "gpt-4",
}


def test_hallucination_scorer():
    print("=== Test 1: HallucinationScorer ===")
    scorer = HallucinationScorer()

    # Score grounded response
    grounded_score = scorer.score(
        response=GROUNDED_RESULT["response"],
        context_chunks=GROUNDED_RESULT["retrieved_chunks"],
    )
    print("\nGrounded response scores:")
    print(f"  Grounding accuracy:   {grounded_score['grounding_accuracy']}")
    print(f"  Hallucination rate:   {grounded_score['hallucination_rate']}")
    print(f"  Total sentences:      {grounded_score['total_sentences']}")
    print(f"  Grounded sentences:   {grounded_score['grounded_sentences']}")
    print(f"  Ungrounded sentences: {grounded_score['ungrounded_sentences']}")

    # Score baseline response (no context)
    baseline_score = scorer.score(
        response=BASELINE_RESULT["response"],
        context_chunks=[],
    )
    print("\nBaseline response scores:")
    print(f"  Grounding accuracy:   {baseline_score['grounding_accuracy']}")
    print(f"  Hallucination rate:   {baseline_score['hallucination_rate']}")

    assert grounded_score["grounding_accuracy"] > baseline_score["grounding_accuracy"]
    assert grounded_score["hallucination_rate"] < baseline_score["hallucination_rate"]
    print("\n>>> HallucinationScorer test PASSED")


def test_score_pair():
    print("\n=== Test 2: Score Pair Comparison ===")
    scorer = HallucinationScorer()

    comparison = scorer.score_pair(
        grounded_result=GROUNDED_RESULT,
        baseline_result=BASELINE_RESULT,
    )

    print(f"  Grounding improvement:    {comparison['grounding_improvement']}")
    print(f"  Hallucination reduction:  {comparison['hallucination_reduction']}")
    print(f"  Latency cost (ms):        {comparison['latency_cost_ms']}")

    assert comparison["grounding_improvement"] >= 0
    assert comparison["hallucination_reduction"] >= 0
    print(">>> Score pair test PASSED")


def test_grounding_scorer_session():
    print("\n=== Test 3: GroundingScorer Session ===")

    scorer = GroundingScorer(
        participant_id="P001_TEST",
        condition="grounded",
    )

    # Record two query pairs
    scorer.record(GROUNDED_RESULT, BASELINE_RESULT, scenario_id="phishing-01")
    scorer.record(GROUNDED_RESULT, BASELINE_RESULT, scenario_id="phishing-02")

    report = scorer.generate_report()
    print(f"  Session ID:                {report['session_id']}")
    print(f"  Participant:               {report['participant_id']}")
    print(f"  Total queries:             {report['total_queries']}")
    print(f"  Mean grounding accuracy:   {report['aggregate']['mean_grounding_accuracy']}")
    print(f"  Mean hallucination (RAG):  {report['aggregate']['mean_hallucination_rate_grounded']}")
    print(f"  Mean hallucination (base): {report['aggregate']['mean_hallucination_rate_baseline']}")
    print(f"  Mean improvement:          {report['aggregate']['mean_grounding_improvement']}")

    assert report["total_queries"] == 2
    assert report["aggregate"]["mean_grounding_accuracy"] > 0

    # Save log
    log_path = scorer.save_session_log()
    assert log_path.exists()
    print(f"  Log saved to:              {log_path}")

    # Export DataFrame
    df = scorer.to_dataframe()
    print(f"\n  DataFrame shape: {df.shape}")
    print(df.to_string(index=False))

    assert len(df) == 2
    assert "grounding_accuracy" in df.columns
    print("\n>>> GroundingScorer session test PASSED")


if __name__ == "__main__":
    test_hallucination_scorer()
    test_score_pair()
    test_grounding_scorer_session()
    print("\n" + "=" * 60)
    print("All evaluation tests PASSED")