"""
SENTRY — Analytics Route Handlers
====================================
API endpoints for aggregated analytics data.
Powers Timothy's organisational dashboard and
Derick's research study report generation.

Endpoints:
    GET /api/v1/analytics/study     — Full study aggregate (research)
    GET /api/v1/analytics/sessions  — List all sessions
    GET /api/v1/analytics/organisation/{org_id} — Org-level summary

Used by:
    middleware/main.py
    mobile_app dashboard screens
    evaluation/run_evaluation.py
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.database.connection import get_db
from backend.database.models import (
    TrainingSession,
    EvaluationLog,
    ScenarioInteraction,
)
from middleware.validators.session_validator import (
    SessionSummary,
    StudyAnalytics,
    OrganisationAnalytics,
)

logger = logging.getLogger("sentry.analytics")
router = APIRouter()


@router.get(
    "/analytics/study",
    response_model=StudyAnalytics,
    summary="Full study analytics",
    description=(
        "Returns aggregated metrics across all participants "
        "for the research evaluation report."
    ),
    tags=["Analytics"],
)
async def get_study_analytics(db: AsyncSession = Depends(get_db)):
    """
    Aggregates all session and evaluation data for the study report.
    This is the primary analytics endpoint for Derick's evaluation chapter.
    """
    # Total sessions per condition
    grounded_count = await db.scalar(
        select(func.count(TrainingSession.id)).where(
            TrainingSession.condition == "grounded"
        )
    )
    baseline_count = await db.scalar(
        select(func.count(TrainingSession.id)).where(
            TrainingSession.condition == "baseline"
        )
    )

    # Grounding metrics from evaluation logs
    grounded_accuracy = await db.scalar(
        select(func.avg(EvaluationLog.grounding_accuracy)).where(
            EvaluationLog.mode == "grounded"
        )
    )
    baseline_accuracy = await db.scalar(
        select(func.avg(EvaluationLog.grounding_accuracy)).where(
            EvaluationLog.mode == "baseline"
        )
    )
    grounded_hallucination = await db.scalar(
        select(func.avg(EvaluationLog.hallucination_rate)).where(
            EvaluationLog.mode == "grounded"
        )
    )
    baseline_hallucination = await db.scalar(
        select(func.avg(EvaluationLog.hallucination_rate)).where(
            EvaluationLog.mode == "baseline"
        )
    )
    mean_improvement = await db.scalar(
        select(func.avg(EvaluationLog.grounding_improvement))
    )

    # Knowledge gain per condition
    grounded_gain = await db.scalar(
        select(func.avg(TrainingSession.knowledge_gain)).where(
            TrainingSession.condition == "grounded",
            TrainingSession.is_complete == True,
        )
    )
    baseline_gain = await db.scalar(
        select(func.avg(TrainingSession.knowledge_gain)).where(
            TrainingSession.condition == "baseline",
            TrainingSession.is_complete == True,
        )
    )

    def safe_round(val, digits=4):
        return round(float(val), digits) if val is not None else None

    return StudyAnalytics(
        total_sessions=(grounded_count or 0) + (baseline_count or 0),
        grounded_sessions=grounded_count or 0,
        baseline_sessions=baseline_count or 0,
        mean_grounding_accuracy_grounded=safe_round(grounded_accuracy),
        mean_grounding_accuracy_baseline=safe_round(baseline_accuracy),
        mean_hallucination_rate_grounded=safe_round(grounded_hallucination),
        mean_hallucination_rate_baseline=safe_round(baseline_hallucination),
        mean_grounding_improvement=safe_round(mean_improvement),
        mean_knowledge_gain_grounded=safe_round(grounded_gain),
        mean_knowledge_gain_baseline=safe_round(baseline_gain),
    )


@router.get(
    "/analytics/sessions",
    response_model=List[SessionSummary],
    summary="List all sessions",
    tags=["Analytics"],
)
async def list_sessions(
    condition: Optional[str] = Query(default=None),
    organisation_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns a list of all training sessions.
    Filterable by condition and organisation.
    """
    query = select(TrainingSession)

    if condition:
        query = query.where(TrainingSession.condition == condition)
    if organisation_id:
        query = query.where(
            TrainingSession.organisation_id == organisation_id
        )

    query = query.limit(limit)
    result = await db.execute(query)
    sessions = result.scalars().all()

    return [
        SessionSummary(
            session_id=s.id,
            participant_id=s.participant_id,
            condition=s.condition,
            is_complete=s.is_complete,
            duration_seconds=s.duration_seconds,
            pre_assessment_score=s.pre_assessment_score,
            post_assessment_score=s.post_assessment_score,
            knowledge_gain=s.knowledge_gain,
            started_at=s.started_at,
        )
        for s in sessions
    ]


@router.get(
    "/analytics/organisation/{organisation_id}",
    response_model=OrganisationAnalytics,
    summary="Organisation-level analytics",
    tags=["Analytics"],
)
async def get_organisation_analytics(
    organisation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns aggregated training metrics for one organisation.
    Powers the organisational dashboard in the Android app.
    """
    total = await db.scalar(
        select(func.count(TrainingSession.id)).where(
            TrainingSession.organisation_id == organisation_id
        )
    )
    completed = await db.scalar(
        select(func.count(TrainingSession.id)).where(
            TrainingSession.organisation_id == organisation_id,
            TrainingSession.is_complete == True,
        )
    )
    mean_pre = await db.scalar(
        select(func.avg(TrainingSession.pre_assessment_score)).where(
            TrainingSession.organisation_id == organisation_id
        )
    )
    mean_post = await db.scalar(
        select(func.avg(TrainingSession.post_assessment_score)).where(
            TrainingSession.organisation_id == organisation_id
        )
    )
    mean_gain = await db.scalar(
        select(func.avg(TrainingSession.knowledge_gain)).where(
            TrainingSession.organisation_id == organisation_id
        )
    )

    # Join with eval logs for grounding metrics
    grounding = await db.scalar(
        select(func.avg(EvaluationLog.grounding_accuracy))
        .join(TrainingSession, EvaluationLog.session_id == TrainingSession.id)
        .where(TrainingSession.organisation_id == organisation_id)
    )
    hallucination = await db.scalar(
        select(func.avg(EvaluationLog.hallucination_rate))
        .join(TrainingSession, EvaluationLog.session_id == TrainingSession.id)
        .where(TrainingSession.organisation_id == organisation_id)
    )

    def safe_round(val, digits=2):
        return round(float(val), digits) if val is not None else None

    return OrganisationAnalytics(
        organisation_id=organisation_id,
        total_sessions=total or 0,
        completed_sessions=completed or 0,
        mean_pre_score=safe_round(mean_pre),
        mean_post_score=safe_round(mean_post),
        mean_knowledge_gain=safe_round(mean_gain),
        mean_grounding_accuracy=safe_round(grounding),
        mean_hallucination_rate=safe_round(hallucination),
    )