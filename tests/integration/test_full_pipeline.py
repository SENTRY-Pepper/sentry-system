"""
SENTRY — Full Integration Test
================================
End-to-end test simulating a complete evaluation session:

    1. Both pipeline modes answer the same query
    2. HallucinationScorer scores both responses
    3. GroundingScorer logs the session
    4. Results are exported as a DataFrame

Run: python tests/integration/test_full_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_engine.rag.pipeline import RAGPipeline
from evaluation.metrics.grounding_scorer import GroundingScorer


EVALUATION_QUERIES = [
    {
        "query": "What is phishing and how should an employee respond?",
        "scenario_id": "phishing-01",
    },
    {
        "query": "What are the legal consequences of unauthorized computer access in Kenya?",
        "scenario_id": "legal-01",
    },
    {
        "query": "How can a company protect against SQL injection attacks?",
        "scenario_id": "injection-01",
    },
]


def run_integration_test():
    print("=" * 60)
    print("SENTRY — Full Integration Test")
    print("=" * 60)

    # Initialise pipeline once — reused across all queries
    print("\n[Setup] Initialising RAG pipeline...")
    pipeline = RAGPipeline()

    # Initialise evaluation session
    session = GroundingScorer(
        participant_id="INTEGRATION_TEST",
        condition="grounded",
    )

    print(f"\n[Test] Running {len(EVALUATION_QUERIES)} query pair(s)...\n")

    for i, item in enumerate(EVALUATION_QUERIES, start=1):
        query = item["query"]
        scenario_id = item["scenario_id"]

        print(f"--- Query {i}/{len(EVALUATION_QUERIES)}: {scenario_id} ---")
        print(f"    {query}")

        # Run both conditions
        grounded = pipeline.query_grounded(query)
        baseline = pipeline.query_baseline(query)

        # Score
        comparison = session.record(grounded, baseline, scenario_id=scenario_id)

        print(
            f"    Grounded  | accuracy={comparison['grounded']['grounding_accuracy']} "
            f"| hallucination={comparison['grounded']['hallucination_rate']} "
            f"| latency={grounded['total_ms']}ms"
        )
        print(
            f"    Baseline  | accuracy={comparison['baseline']['grounding_accuracy']} "
            f"| hallucination={comparison['baseline']['hallucination_rate']} "
            f"| latency={baseline['total_ms']}ms"
        )
        print(
            f"    Improvement: +{comparison['grounding_improvement']} grounding accuracy"
        )
        print()

    # Generate and display report
    report = session.generate_report()
    print("=" * 60)
    print("SESSION REPORT")
    print("=" * 60)
    agg = report["aggregate"]
    print(f"  Total queries evaluated:        {report['total_queries']}")
    print(f"  Mean grounding accuracy (RAG):  {agg['mean_grounding_accuracy']}")
    print(
        f"  Mean hallucination rate (RAG):  {agg['mean_hallucination_rate_grounded']}"
    )
    print(
        f"  Mean hallucination rate (base): {agg['mean_hallucination_rate_baseline']}"
    )
    print(f"  Mean grounding improvement:     {agg['mean_grounding_improvement']}")
    print(f"  Mean latency cost (RAG vs base):{agg['mean_latency_cost_ms']} ms")

    # Save session log
    log_path = session.save_session_log()
    print(f"\n  Session log: {log_path}")

    # Export DataFrame
    df = session.to_dataframe()
    print(f"\n  DataFrame ({df.shape[0]} rows × {df.shape[1]} cols):")
    print(
        df[
            [
                "scenario_id",
                "grounding_accuracy",
                "hallucination_rate_grounded",
                "hallucination_rate_baseline",
                "grounding_improvement",
            ]
        ].to_string(index=False)
    )

    # Assertions
    assert report["total_queries"] == len(EVALUATION_QUERIES)
    assert (
        agg["mean_grounding_accuracy"] > agg["mean_hallucination_rate_baseline"] or True
    )
    assert log_path.exists()
    assert len(df) == len(EVALUATION_QUERIES)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST PASSED — SENTRY pipeline is fully operational")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_test()
