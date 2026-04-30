"""
SENTRY — Database Connection and Models Test
Run: python tests/unit/test_database.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.database.models import (
    AssessmentResult,
    Base,
    EvaluationLog,
    ScenarioInteraction,
    TrainingSession,
)
from config.settings import settings


def make_engine():
    """
    Create a fresh async engine with NullPool.
    NullPool disables connection pooling entirely — every operation
    opens and closes its own connection. This eliminates the
    'Future attached to a different loop' error in pytest because
    no connection is shared between event loops.
    """
    return create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        echo=False,
    )


async def test_connection():
    print("\n=== Test 1: Database Connection ===")
    engine = make_engine()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"  PostgreSQL version: {version}")
        assert version is not None
        print(">>> Connection test PASSED")
    finally:
        await engine.dispose()


async def test_table_creation():
    print("\n=== Test 2: Table Creation ===")
    engine = make_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' "
                    "AND table_name IN ("
                    "'training_sessions', 'scenario_interactions', "
                    "'assessment_results', 'evaluation_logs'"
                    ")"
                )
            )
            tables = {row[0] for row in result.fetchall()}

        expected = {
            "training_sessions",
            "scenario_interactions",
            "assessment_results",
            "evaluation_logs",
        }
        assert expected == tables, f"Missing tables: {expected - tables}"
        print(f"  Tables confirmed: {sorted(tables)}")
        print(">>> Table creation test PASSED")
    finally:
        await engine.dispose()


async def test_insert_and_query():
    print("\n=== Test 3: Insert and Query ===")
    engine = make_engine()
    try:
        # Ensure tables exist
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with session_factory() as db:
            session = TrainingSession(
                participant_id="TEST_P001",
                condition="grounded",
                organisation_id="TEST_ORG",
            )
            db.add(session)
            await db.flush()

            print(f"  Created session: {session.id}")
            print(f"  Participant:     {session.participant_id}")
            print(f"  Condition:       {session.condition}")

            interaction = ScenarioInteraction(
                session_id=session.id,
                scenario_id="phishing-01",
                scenario_type="phishing",
                decision="correct",
                employee_response="I would report it to IT",
                response_time_ms=4200,
                correction_loops=0,
                ai_latency_ms=6800.5,
                ai_sources="Computer-Misuse-and-Cybercrimes-Act.pdf",
            )
            db.add(interaction)

            assessment = AssessmentResult(
                session_id=session.id,
                pre_score=45.0,
                post_score=72.0,
                knowledge_gain=27.0,
                relative_improvement_pct=60.0,
            )
            db.add(assessment)

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
            await db.flush()

            print(
                f"  Interaction logged: {interaction.scenario_id} "
                f"→ {interaction.decision}"
            )
            print(
                f"  Assessment:         pre={assessment.pre_score} "
                f"post={assessment.post_score}"
            )
            print(f"  Knowledge gain:     {assessment.knowledge_gain}%")
            print(f"  Eval log:           grounding={eval_log.grounding_accuracy}")

            assert session.id is not None
            assert interaction.id is not None
            assert assessment.knowledge_gain == 27.0
            assert eval_log.grounding_accuracy == 0.75

            # Clean up test data
            await db.delete(session)
            await db.commit()
            print("  Test data cleaned up.")

        print(">>> Insert and query test PASSED")
    finally:
        await engine.dispose()
