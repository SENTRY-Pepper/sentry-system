"""
SENTRY — Session Route Handlers
==================================
API endpoints for session lifecycle management and interaction logging.
Called by Timothy's Android app during training sessions.

Endpoints:
    POST /api/v1/sessions/start         — Create a new training session
    POST /api/v1/sessions/end           — Close and score a session
    POST /api/v1/sessions/interaction   — Log one scenario interaction
    POST /api/v1/sessions/eval-log      — Log one evaluation record
    GET  /api/v1/sessions/{session_id}  — Retrieve a session summary

Used by:
    middleware/main.py           (router registration)
    mobile_app ApiService.kt     (Android app calls)
    pepper_interface/            (NAOqi layer calls)
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database.connection import get_db
from backend.database.models import (
    TrainingSession,
    ScenarioInteraction,
    EvaluationLog,
    AssessmentResult,
)
from middleware.validators.session_validator import (
    SessionStartRequest,
    SessionStartResponse,
    SessionEndRequest,
    SessionEndResponse,
    InteractionLogRequest,
    InteractionLogResponse,
    EvaluationLogRequest,
    SessionSummary,
)

logger = logging.getLogger("sentry.sessions")
router = APIRouter()


# ------------------------------------------------------------------
# Session lifecycle
# ------------------------------------------------------------------

@router.post(
    "/sessions/start",
    response_model=SessionStartResponse,
    summary="Start a training session",
    tags=["Sessions"],
)
async def start_session(
    body: SessionStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by the Android app when an employee begins training.
    Creates a new TrainingSession record and returns the session_id
    that must be included in all subsequent requests.
    """
    session = TrainingSession(
        participant_id=body.participant_id,
        condition=body.condition,
        organisation_id=body.organisation_id,
    )
    db.add(session)
    await db.flush()

    logger.info(
        f"[sessions/start] New session: {session.id} "
        f"participant={body.participant_id} condition={body.condition}"
    )

    return SessionStartResponse(
        session_id=session.id,
        participant_id=session.participant_id,
        condition=session.condition,
        started_at=session.started_at,
    )


@router.post(
    "/sessions/end",
    response_model=SessionEndResponse,
    summary="End a training session",
    tags=["Sessions"],
)
async def end_session(
    body: SessionEndRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by the Android app when a session completes.
    Updates the session with final scores, computes knowledge gain,
    and marks it as complete.
    """
    # Fetch the session
    result = await db.execute(
        select(TrainingSession).where(TrainingSession.id == body.session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id}' not found.",
        )

    # Compute knowledge gain if both scores provided
    knowledge_gain = None
    relative_improvement = None

    if body.pre_assessment_score is not None and body.post_assessment_score is not None:
        knowledge_gain = round(
            body.post_assessment_score - body.pre_assessment_score, 2
        )
        if body.pre_assessment_score > 0:
            relative_improvement = round(
                (knowledge_gain / body.pre_assessment_score) * 100, 2
            )

    # Update session record
    from datetime import datetime, timezone
    session.completed_at = datetime.now(timezone.utc)
    session.duration_seconds = body.duration_seconds
    session.pre_assessment_score = body.pre_assessment_score
    session.post_assessment_score = body.post_assessment_score
    session.knowledge_gain = knowledge_gain
    session.is_complete = True

    # Create assessment result record
    assessment = AssessmentResult(
        session_id=session.id,
        pre_score=body.pre_assessment_score or 0.0,
        post_score=body.post_assessment_score,
        knowledge_gain=knowledge_gain,
        relative_improvement_pct=relative_improvement,
    )
    db.add(assessment)

    logger.info(
        f"[sessions/end] Closed session: {session.id} "
        f"gain={knowledge_gain}"
    )

    return SessionEndResponse(
        session_id=session.id,
        participant_id=session.participant_id,
        duration_seconds=session.duration_seconds,
        pre_assessment_score=session.pre_assessment_score,
        post_assessment_score=session.post_assessment_score,
        knowledge_gain=knowledge_gain,
        relative_improvement_pct=relative_improvement,
        is_complete=True,
    )


# ------------------------------------------------------------------
# Interaction logging
# ------------------------------------------------------------------

@router.post(
    "/sessions/interaction",
    response_model=InteractionLogResponse,
    summary="Log a scenario interaction",
    tags=["Sessions"],
)
async def log_interaction(
    body: InteractionLogRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Called after each scenario decision by the employee.
    Logs the decision, response time, correction loops, and
    AI latency for behavioural analytics.
    """
    interaction = ScenarioInteraction(
        session_id=body.session_id,
        scenario_id=body.scenario_id,
        scenario_type=body.scenario_type,
        decision=body.decision,
        employee_response=body.employee_response,
        response_time_ms=body.response_time_ms,
        correction_loops=body.correction_loops,
        ai_latency_ms=body.ai_latency_ms,
        ai_sources=body.ai_sources,
    )
    db.add(interaction)
    await db.flush()

    logger.info(
        f"[sessions/interaction] Logged: {body.scenario_id} "
        f"decision={body.decision} session={body.session_id}"
    )

    return InteractionLogResponse(
        interaction_id=interaction.id,
        session_id=interaction.session_id,
        scenario_id=interaction.scenario_id,
        decision=interaction.decision,
        logged=True,
    )


@router.post(
    "/sessions/eval-log",
    summary="Log an evaluation record",
    tags=["Sessions"],
)
async def log_evaluation(
    body: EvaluationLogRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by Derick's evaluation module after scoring a query pair.
    Links grounding accuracy and hallucination scores to the session.
    """
    eval_log = EvaluationLog(
        session_id=body.session_id,
        scenario_id=body.scenario_id,
        query=body.query,
        mode=body.mode,
        response=body.response,
        grounding_accuracy=body.grounding_accuracy,
        hallucination_rate=body.hallucination_rate,
        grounding_improvement=body.grounding_improvement,
        retrieval_ms=body.retrieval_ms,
        generation_ms=body.generation_ms,
        total_ms=body.total_ms,
        prompt_tokens=body.prompt_tokens,
        completion_tokens=body.completion_tokens,
        sources=body.sources,
    )
    db.add(eval_log)

    logger.info(
        f"[sessions/eval-log] mode={body.mode} "
        f"grounding={body.grounding_accuracy} session={body.session_id}"
    )

    return {
        "logged": True,
        "session_id": body.session_id,
        "mode": body.mode,
    }


# ------------------------------------------------------------------
# Session retrieval
# ------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}",
    response_model=SessionSummary,
    summary="Get session summary",
    tags=["Sessions"],
)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve a session record by ID.
    Used by the Android app's ResultsActivity to display
    the final session summary to the employee.
    """
    result = await db.execute(
        select(TrainingSession).where(TrainingSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    return SessionSummary(
        session_id=session.id,
        participant_id=session.participant_id,
        condition=session.condition,
        is_complete=session.is_complete,
        duration_seconds=session.duration_seconds,
        pre_assessment_score=session.pre_assessment_score,
        post_assessment_score=session.post_assessment_score,
        knowledge_gain=session.knowledge_gain,
        started_at=session.started_at,
    )