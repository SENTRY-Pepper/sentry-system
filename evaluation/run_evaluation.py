"""
SENTRY — Evaluation Study Runner
===================================
Orchestrates a complete participant evaluation session for Phase 5.

What this script does:
    1. Registers a participant session via the middleware API
    2. For each evaluation query:
        a. Runs both pipeline modes (grounded + baseline)
        b. Scores both responses with HallucinationScorer
        c. Logs the evaluation record to PostgreSQL
        d. Saves results to the local session log
    3. Closes the session with assessment scores
    4. Generates a complete session report
    5. Exports a DataFrame for statistical analysis

Usage:
    python evaluation/run_evaluation.py \
        --participant P001 \
        --condition grounded \
        --organisation SME_NAIROBI \
        --pre-score 45.0 \
        --post-score 72.0

This script is run once per participant during the user study.
All output is saved to evaluation/logs/ and evaluation/reports/.

The aggregate analysis across all participants (t-test, Cohen's d,
effect size) is performed separately in evaluation/analyse_results.py
after all sessions are complete.
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_engine.rag.pipeline import RAGPipeline
from evaluation.metrics.hallucination_scorer import HallucinationScorer
from evaluation.metrics.grounding_scorer import GroundingScorer
from config.settings import settings

# ------------------------------------------------------------------
# Evaluation query set
# These are the standardised queries used across all participants.
# All participants receive the same queries in the same order
# to ensure comparability between conditions.
# ------------------------------------------------------------------

EVALUATION_QUERIES = [
    {
        "query": "What is phishing and how should an employee respond to a suspicious email?",
        "scenario_id": "phishing-01",
        "scenario_type": "phishing",
    },
    {
        "query": "What are the legal consequences of unauthorized computer access in Kenya?",
        "scenario_id": "legal-01",
        "scenario_type": "legal",
    },
    {
        "query": "How can a company protect against SQL injection attacks?",
        "scenario_id": "injection-01",
        "scenario_type": "technical",
    },
    {
        "query": "What should an employee do if they find an unknown USB drive at work?",
        "scenario_id": "usb-01",
        "scenario_type": "physical",
    },
    {
        "query": "What is social engineering and how can employees recognise it?",
        "scenario_id": "social-01",
        "scenario_type": "social_engineering",
    },
]

MIDDLEWARE_BASE_URL = f"http://localhost:{settings.MIDDLEWARE_PORT}"


# ------------------------------------------------------------------
# Middleware API helpers
# ------------------------------------------------------------------

async def start_session(
    client: httpx.AsyncClient,
    participant_id: str,
    condition: str,
    organisation_id: str,
) -> str:
    """Register session with middleware and return session_id."""
    response = await client.post(
        "/api/v1/sessions/start",
        json={
            "participant_id": participant_id,
            "condition": condition,
            "organisation_id": organisation_id,
        },
    )
    response.raise_for_status()
    data = response.json()
    print(f"[Evaluation] Session started: {data['session_id']}")
    return data["session_id"]


async def log_eval_record(
    client: httpx.AsyncClient,
    session_id: str,
    comparison: dict,
    pipeline_result: dict,
) -> None:
    """Post one evaluation log record to the middleware."""
    await client.post(
        "/api/v1/sessions/eval-log",
        json={
            "session_id": session_id,
            "scenario_id": comparison.get("scenario_id"),
            "query": pipeline_result["query"],
            "mode": pipeline_result["mode"],
            "response": pipeline_result["response"],
            "grounding_accuracy": comparison["grounded"]["grounding_accuracy"],
            "hallucination_rate": comparison["grounded"]["hallucination_rate"],
            "grounding_improvement": comparison["grounding_improvement"],
            "retrieval_ms": pipeline_result.get("retrieval_ms"),
            "generation_ms": pipeline_result.get("generation_ms"),
            "total_ms": pipeline_result.get("total_ms"),
            "prompt_tokens": pipeline_result.get("prompt_tokens"),
            "completion_tokens": pipeline_result.get("completion_tokens"),
            "sources": ",".join(pipeline_result.get("sources", [])),
        },
    )


async def end_session(
    client: httpx.AsyncClient,
    session_id: str,
    pre_score: float,
    post_score: float,
    duration_seconds: int,
) -> dict:
    """Close the session and return the final summary."""
    response = await client.post(
        "/api/v1/sessions/end",
        json={
            "session_id": session_id,
            "pre_assessment_score": pre_score,
            "post_assessment_score": post_score,
            "duration_seconds": duration_seconds,
        },
    )
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------------
# Main evaluation runner
# ------------------------------------------------------------------

async def run_evaluation(
    participant_id: str,
    condition: str,
    organisation_id: str,
    pre_score: float,
    post_score: float,
) -> None:
    print("=" * 60)
    print("SENTRY — Evaluation Session Runner")
    print("=" * 60)
    print(f"  Participant:   {participant_id}")
    print(f"  Condition:     {condition}")
    print(f"  Organisation:  {organisation_id}")
    print(f"  Pre-score:     {pre_score}")
    print(f"  Post-score:    {post_score}")
    print(f"  Queries:       {len(EVALUATION_QUERIES)}")
    print("=" * 60)

    session_start_time = datetime.utcnow()

    # Initialise components
    print("\n[Setup] Initialising RAG pipeline...")
    pipeline = RAGPipeline()

    hallucination_scorer = HallucinationScorer()
    grounding_session = GroundingScorer(
        participant_id=participant_id,
        condition=condition,
    )

    async with httpx.AsyncClient(
        base_url=MIDDLEWARE_BASE_URL, timeout=60
    ) as client:

        # Verify server is ready
        health = await client.get("/health")
        if not health.json().get("pipeline_ready"):
            raise RuntimeError(
                "Middleware pipeline is not ready. "
                "Start the server first: "
                "uvicorn middleware.main:app --host 0.0.0.0 --port 8000"
            )

        # Register session
        session_id = await start_session(
            client, participant_id, condition, organisation_id
        )

        print(f"\n[Evaluation] Running {len(EVALUATION_QUERIES)} queries...\n")

        # Process each query
        for i, item in enumerate(EVALUATION_QUERIES, start=1):
            query = item["query"]
            scenario_id = item["scenario_id"]
            scenario_type = item["scenario_type"]

            print(f"--- Query {i}/{len(EVALUATION_QUERIES)}: {scenario_id} ---")
            print(f"    {query}")

            # Run both pipeline modes
            grounded_result = pipeline.query_grounded(query)
            baseline_result = pipeline.query_baseline(query)

            # Score the pair
            comparison = grounding_session.record(
                grounded_result,
                baseline_result,
                scenario_id=scenario_id,
            )

            print(f"    Sources retrieved: {grounded_result.get('sources', [])}")

            # Log to PostgreSQL via middleware
            await log_eval_record(
                client, session_id, comparison, grounded_result
            )

            # Print per-query results
            g = comparison["grounded"]
            b = comparison["baseline"]
            print(
                f"    Grounded  | accuracy={g['grounding_accuracy']} "
                f"hallucination={g['hallucination_rate']} "
                f"latency={grounded_result['total_ms']}ms"
            )
            print(
                f"    Baseline  | accuracy={b['grounding_accuracy']} "
                f"hallucination={b['hallucination_rate']} "
                f"latency={baseline_result['total_ms']}ms"
            )
            print(
                f"    Improvement: +{comparison['grounding_improvement']} "
                f"grounding accuracy"
            )
            print()

        # Calculate session duration
        duration_seconds = int(
            (datetime.utcnow() - session_start_time).total_seconds()
        )

        # Close the session
        session_summary = await end_session(
            client=client,
            session_id=session_id,
            pre_score=pre_score,
            post_score=post_score,
            duration_seconds=duration_seconds,
        )

    # Generate and display report
    report = grounding_session.generate_report()
    agg = report["aggregate"]

    print("=" * 60)
    print("SESSION COMPLETE — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Session ID:                    {session_id}")
    print(f"  Participant:                   {participant_id}")
    print(f"  Condition:                     {condition}")
    print(f"  Total queries:                 {report['total_queries']}")
    print(f"  Duration:                      {duration_seconds}s")
    print()
    print("  --- Grounding Metrics ---")
    print(f"  Mean grounding accuracy (RAG): {agg['mean_grounding_accuracy']}")
    print(f"  Mean hallucination rate (RAG): {agg['mean_hallucination_rate_grounded']}")
    print(f"  Mean hallucination (baseline): {agg['mean_hallucination_rate_baseline']}")
    print(f"  Mean grounding improvement:    {agg['mean_grounding_improvement']}")
    print(f"  Mean latency cost:             {agg['mean_latency_cost_ms']}ms")
    print()
    print("  --- Assessment ---")
    print(f"  Pre-score:                     {pre_score}%")
    print(f"  Post-score:                    {post_score}%")
    print(f"  Knowledge gain:                {session_summary['knowledge_gain']}%")
    print(f"  Relative improvement:          {session_summary['relative_improvement_pct']}%")
    print("=" * 60)

    # Save local session log
    log_path = grounding_session.save_session_log()
    print(f"\n  Local log saved:  {log_path}")

    # Save DataFrame to CSV for statistical analysis
    df = grounding_session.to_dataframe()
    csv_path = (
        Path(settings.EVAL_REPORT_DIR)
        / f"session_{participant_id}_{grounding_session.session_id}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"  CSV exported:     {csv_path}")

    # Save full report JSON
    report["session_id_middleware"] = session_id
    report["assessment"] = {
        "pre_score": pre_score,
        "post_score": post_score,
        "knowledge_gain": session_summary["knowledge_gain"],
        "relative_improvement_pct": session_summary["relative_improvement_pct"],
    }
    report_path = (
        Path(settings.EVAL_REPORT_DIR)
        / f"report_{participant_id}_{grounding_session.session_id}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved:     {report_path}")
    print()
    print("Evaluation session complete.")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SENTRY Evaluation Session Runner"
    )
    parser.add_argument(
        "--participant",
        required=True,
        help="Anonymised participant ID (e.g. P001)",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["grounded", "baseline"],
        help="Study condition for this participant",
    )
    parser.add_argument(
        "--organisation",
        default="SENTRY_STUDY",
        help="Organisation identifier (default: SENTRY_STUDY)",
    )
    parser.add_argument(
        "--pre-score",
        type=float,
        required=True,
        help="Pre-assessment score (0-100)",
    )
    parser.add_argument(
        "--post-score",
        type=float,
        required=True,
        help="Post-assessment score (0-100)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_evaluation(
            participant_id=args.participant,
            condition=args.condition,
            organisation_id=args.organisation,
            pre_score=args.pre_score,
            post_score=args.post_score,
        )
    )