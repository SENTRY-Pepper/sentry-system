"""
SENTRY — Database Connection and Models Test
Run: python tests/unit/test_database.py
"""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text
from backend.database.connection import engine, init_db
from backend.database.models import (
    TrainingSession,
    ScenarioInteraction,
    AssessmentResult,
    EvaluationLog,
)


async def test_connection():
    print("=== Test 1: Database Connection ===")
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT version()"))
        version = result.scalar()
        print(f"  PostgreSQL version: {version}")
    print(">>> Connection test PASSED")


async def test_table_creation():
    print("\n=== Test 2: Table Creation ===")
    await init_db()
    print("  Tables created: training_sessions, scenario_interactions,")
    print("                  assessment_results, evaluation_logs")
    print(">>> Table creation test PASSED")


async def test_insert_and_query():
    print("\n=== Test 3: Insert and Query ===")
    from backend.database.connection import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        # Create a test session
        session = TrainingSession(
            participant_id="TEST_P001",
            condition="grounded",
            organisation_id="TEST_ORG",
        )
        db.add(session)
        await db.flush()  # Get the ID without committing

        print(f"  Created session: {session.id}")
        print(f"  Participant:     {session.participant_id}")
        print(f"  Condition:       {session.condition}")

        # Add a scenario interaction
        interaction = ScenarioInteraction(
            session_id=session.id,
            scenario_id="phishing-01",
            scenario_type="phishing",
            decision="correct",
            employee_response="I would report it to IT",
            response_time_ms=4200,
            correction_loops=0,
            ai_latency_ms=6800.5,
            ai_sources="Computer-Misuse-and-Cybercrimes-Act.pdf,A07_2025-Authentication_Failures.md",
        )
        db.add(interaction)

        # Add assessment result
        assessment = AssessmentResult(
            session_id=session.id,
            pre_score=45.0,
            post_score=72.0,
            knowledge_gain=27.0,
            relative_improvement_pct=60.0,
        )
        db.add(assessment)

        # Add evaluation log
        eval_log = EvaluationLog(
            session_id=session.id,
            scenario_id="phishing-01",
            query="What is phishing?",
            mode="grounded",
            response="Phishing is...",
            grounding_accuracy=0.75,
            hallucination_rate=0.25,
            grounding_improvement=0.75,
            retrieval_ms=350.0,
            generation_ms=6200.0,
            total_ms=6550.0,
            prompt_tokens=400,
            completion_tokens=180,
            sources="Computer-Misuse-and-Cybercrimes-Act.pdf",
        )
        db.add(eval_log)
        await db.commit()

        print(f"  Interaction logged: {interaction.scenario_id} → {interaction.decision}")
        print(f"  Assessment:         pre={assessment.pre_score} post={assessment.post_score}")
        print(f"  Knowledge gain:     {assessment.knowledge_gain}%")
        print(f"  Eval log:           grounding={eval_log.grounding_accuracy}")

        # Clean up test data
        await db.delete(session)
        await db.commit()
        print("  Test data cleaned up.")

    print(">>> Insert and query test PASSED")


async def run_all():
    await test_connection()
    await test_table_creation()
    await test_insert_and_query()
    print("\n" + "=" * 60)
    print("All database tests PASSED")


if __name__ == "__main__":
    asyncio.run(run_all())