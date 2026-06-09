"""
SENTRY organisation, account, and manager analytics routes.

These endpoints introduce the Phase 6 role split:

- Admin accounts own research analytics.
- Manager accounts own organisation trainee management and performance.
- Trainee accounts own learning sessions.

Authentication is prototype-grade bearer-token validation against persisted
users. The data model is real and persisted; token signing and expiry remain
future hardening work.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db
from backend.database.models import (
    Organisation,
    ScenarioInteraction,
    TrainingSession,
    User,
)
from middleware.auth import require_roles
from middleware.validators.session_validator import (
    DepartmentAnalytics,
    ManagerOverviewAnalytics,
    OrganisationCreateRequest,
    OrganisationResponse,
    TraineeAnalytics,
    UserCreateRequest,
    UserLoginRequest,
    UserLoginResponse,
    UserResponse,
    WeaknessAnalytics,
)

router = APIRouter()


def canonicalise(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return cleaned.upper()


def hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        participant_id=user.participant_id,
        display_name=user.display_name,
        role=user.role,
        organisation_id=user.organisation_id,
        department=user.department,
        position=user.position,
        is_active=user.is_active,
        created_at=user.created_at,
    )


async def ensure_organisation(
    db: AsyncSession,
    organisation_id: Optional[str],
    fallback_name: Optional[str] = None,
) -> Optional[str]:
    if not organisation_id:
        return None

    canonical_id = canonicalise(organisation_id)
    existing = await db.scalar(
        select(Organisation).where(Organisation.canonical_id == canonical_id)
    )
    if existing:
        return existing.canonical_id

    db.add(
        Organisation(
            name=fallback_name or organisation_id.strip() or canonical_id,
            canonical_id=canonical_id,
        )
    )
    await db.flush()
    return canonical_id


async def find_user(
    db: AsyncSession,
    participant_id: str,
    role: Optional[str] = None,
    organisation_id: Optional[str] = None,
) -> Optional[User]:
    query = select(User).where(User.participant_id == participant_id)
    if role:
        query = query.where(User.role == role)
    if organisation_id:
        query = query.where(User.organisation_id == organisation_id)
    result = await db.execute(query.limit(1))
    return result.scalar_one_or_none()


@router.post(
    "/organisations",
    response_model=OrganisationResponse,
    tags=["Organisations"],
)
async def create_organisation(
    body: OrganisationCreateRequest,
    _: User = Depends(require_roles("admin", "manager")),
    db: AsyncSession = Depends(get_db),
):
    canonical_id = canonicalise(body.canonical_id or body.name)
    existing = await db.scalar(
        select(Organisation).where(Organisation.canonical_id == canonical_id)
    )
    if existing:
        return OrganisationResponse(
            id=existing.id,
            name=existing.name,
            canonical_id=existing.canonical_id,
            is_active=existing.is_active,
            created_at=existing.created_at,
        )

    organisation = Organisation(name=body.name, canonical_id=canonical_id)
    db.add(organisation)
    await db.flush()
    return OrganisationResponse(
        id=organisation.id,
        name=organisation.name,
        canonical_id=organisation.canonical_id,
        is_active=organisation.is_active,
        created_at=organisation.created_at,
    )


@router.post("/users/login", response_model=UserLoginResponse, tags=["Users"])
async def login_user(body: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    role = body.role.lower()
    if role not in {"admin", "manager", "trainee"}:
        raise HTTPException(status_code=400, detail="Invalid role.")

    organisation_id = await ensure_organisation(db, body.organisation_id)
    user = await find_user(db, body.participant_id, role, organisation_id)

    if user and user.pin_hash != hash_pin(body.pin):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    if user and not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive.")

    if not user:
        user = User(
            participant_id=body.participant_id,
            display_name=body.participant_id,
            role=role,
            pin_hash=hash_pin(body.pin),
            organisation_id=organisation_id,
        )
        db.add(user)
        await db.flush()

    token = f"{user.id}.{int(datetime.now(timezone.utc).timestamp())}"
    return UserLoginResponse(token=token, user=user_response(user))


@router.get(
    "/manager/trainees", response_model=list[TraineeAnalytics], tags=["Manager"]
)
async def list_trainees(
    organisation_id: str = Query(...),
    current_user: User = Depends(require_roles("admin", "manager")),
    db: AsyncSession = Depends(get_db),
):
    org_id = canonicalise(current_user.organisation_id or organisation_id)
    return await trainee_analytics(db, org_id)


@router.post("/manager/trainees", response_model=UserResponse, tags=["Admin"])
async def create_trainee(
    body: UserCreateRequest,
    current_user: User = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    role = body.role.lower()
    if role not in {"trainee", "manager", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid role.")

    org_id = await ensure_organisation(
        db,
        current_user.organisation_id or body.organisation_id,
    )
    existing = await find_user(db, body.participant_id, role, org_id)
    if existing:
        raise HTTPException(status_code=409, detail="User already exists.")

    user = User(
        participant_id=body.participant_id,
        display_name=body.display_name,
        role=role,
        pin_hash=hash_pin(body.pin),
        organisation_id=org_id,
        department=body.department,
        position=body.position,
    )
    db.add(user)
    await db.flush()
    return user_response(user)


@router.patch(
    "/manager/trainees/{user_id}/deactivate",
    response_model=UserResponse,
    tags=["Admin"],
)
async def deactivate_trainee(
    user_id: str,
    current_user: User = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.organisation_id != current_user.organisation_id:
        raise HTTPException(status_code=403, detail="Cannot manage this trainee.")
    user.is_active = False
    await db.flush()
    return user_response(user)


@router.get(
    "/manager/analytics/overview",
    response_model=ManagerOverviewAnalytics,
    tags=["Manager"],
)
async def manager_overview(
    organisation_id: str = Query(...),
    current_user: User = Depends(require_roles("manager")),
    db: AsyncSession = Depends(get_db),
):
    org_id = canonicalise(current_user.organisation_id or organisation_id)
    total_sessions = await db.scalar(
        select(func.count(TrainingSession.id)).where(
            TrainingSession.organisation_id == org_id
        )
    )
    completed_sessions = await db.scalar(
        select(func.count(TrainingSession.id)).where(
            TrainingSession.organisation_id == org_id,
            TrainingSession.is_complete.is_(True),
        )
    )
    average_score = await db.scalar(
        select(func.avg(TrainingSession.post_assessment_score)).where(
            TrainingSession.organisation_id == org_id,
            TrainingSession.is_complete.is_(True),
        )
    )
    trainee_count = await db.scalar(
        select(func.count(User.id)).where(
            User.organisation_id == org_id,
            User.role == "trainee",
        )
    )
    active_trainees = await db.scalar(
        select(func.count(User.id)).where(
            User.organisation_id == org_id,
            User.role == "trainee",
            User.is_active.is_(True),
        )
    )
    risky_answers = await db.scalar(
        select(func.count(ScenarioInteraction.id))
        .join(TrainingSession, ScenarioInteraction.session_id == TrainingSession.id)
        .where(
            TrainingSession.organisation_id == org_id,
            ScenarioInteraction.decision == "risky",
        )
    )

    completion_rate = (
        round(((completed_sessions or 0) / (total_sessions or 1)) * 100, 2)
        if total_sessions
        else 0.0
    )

    return ManagerOverviewAnalytics(
        organisation_id=org_id,
        trainee_count=trainee_count or 0,
        active_trainees=active_trainees or 0,
        total_sessions=total_sessions or 0,
        completed_sessions=completed_sessions or 0,
        average_score=round(float(average_score), 2)
        if average_score is not None
        else None,
        completion_rate=completion_rate,
        risky_answers=risky_answers or 0,
        top_weaknesses=await weakness_analytics(db, org_id, limit=5),
        departments=await department_analytics(db, org_id),
    )


@router.get(
    "/manager/analytics/weaknesses",
    response_model=list[WeaknessAnalytics],
    tags=["Manager"],
)
async def manager_weaknesses(
    organisation_id: str = Query(...),
    limit: int = Query(default=10, le=50),
    current_user: User = Depends(require_roles("manager")),
    db: AsyncSession = Depends(get_db),
):
    org_id = canonicalise(current_user.organisation_id or organisation_id)
    return await weakness_analytics(db, org_id, limit=limit)


async def weakness_analytics(
    db: AsyncSession,
    organisation_id: str,
    limit: int = 10,
) -> list[WeaknessAnalytics]:
    risky = func.sum(case((ScenarioInteraction.decision == "risky", 1), else_=0))
    correct = func.sum(case((ScenarioInteraction.decision == "correct", 1), else_=0))
    total = func.count(ScenarioInteraction.id)

    result = await db.execute(
        select(
            ScenarioInteraction.scenario_id,
            ScenarioInteraction.scenario_type,
            risky.label("risky_answers"),
            correct.label("correct_answers"),
            total.label("total_answers"),
        )
        .join(TrainingSession, ScenarioInteraction.session_id == TrainingSession.id)
        .where(TrainingSession.organisation_id == organisation_id)
        .group_by(ScenarioInteraction.scenario_id, ScenarioInteraction.scenario_type)
        .order_by(desc("risky_answers"), desc("total_answers"))
        .limit(limit)
    )

    rows = result.all()
    return [
        WeaknessAnalytics(
            scenario_id=row.scenario_id,
            scenario_type=row.scenario_type,
            risky_answers=int(row.risky_answers or 0),
            correct_answers=int(row.correct_answers or 0),
            total_answers=int(row.total_answers or 0),
            risk_rate=round(
                (float(row.risky_answers or 0) / float(row.total_answers or 1)) * 100,
                2,
            ),
        )
        for row in rows
    ]


async def department_analytics(
    db: AsyncSession,
    organisation_id: str,
) -> list[DepartmentAnalytics]:
    department_expr = func.coalesce(User.department, "Unassigned").label("department")
    risky = func.sum(case((ScenarioInteraction.decision == "risky", 1), else_=0))
    result = await db.execute(
        select(
            department_expr,
            func.count(func.distinct(User.id)).label("trainee_count"),
            func.count(func.distinct(TrainingSession.id))
            .filter(TrainingSession.is_complete.is_(True))
            .label("completed_sessions"),
            func.avg(TrainingSession.post_assessment_score).label("average_score"),
            risky.label("risky_answers"),
        )
        .select_from(User)
        .outerjoin(TrainingSession, TrainingSession.user_id == User.id)
        .outerjoin(
            ScenarioInteraction, ScenarioInteraction.session_id == TrainingSession.id
        )
        .where(User.organisation_id == organisation_id, User.role == "trainee")
        .group_by(department_expr)
        .order_by(desc("risky_answers"))
    )
    return [
        DepartmentAnalytics(
            department=row.department,
            trainee_count=int(row.trainee_count or 0),
            completed_sessions=int(row.completed_sessions or 0),
            average_score=round(float(row.average_score), 2)
            if row.average_score is not None
            else None,
            risky_answers=int(row.risky_answers or 0),
        )
        for row in result.all()
    ]


async def trainee_analytics(
    db: AsyncSession,
    organisation_id: str,
) -> list[TraineeAnalytics]:
    users_result = await db.execute(
        select(User)
        .where(User.organisation_id == organisation_id, User.role == "trainee")
        .order_by(User.display_name)
    )
    users = users_result.scalars().all()

    analytics: list[TraineeAnalytics] = []
    for user in users:
        sessions_result = await db.execute(
            select(TrainingSession)
            .where(
                TrainingSession.organisation_id == organisation_id,
                (
                    (TrainingSession.user_id == user.id)
                    | (TrainingSession.participant_id == user.participant_id)
                ),
            )
            .order_by(desc(TrainingSession.started_at))
        )
        sessions = sessions_result.scalars().all()
        scores = [
            s.post_assessment_score
            for s in sessions
            if s.post_assessment_score is not None
        ]

        weakness_rows = await weakness_for_participant(
            db,
            organisation_id,
            user.id,
            user.participant_id,
        )
        risky_total = sum(row.risky_answers for row in weakness_rows)

        analytics.append(
            TraineeAnalytics(
                user_id=user.id,
                participant_id=user.participant_id,
                display_name=user.display_name,
                department=user.department,
                position=user.position,
                is_active=user.is_active,
                session_count=len(sessions),
                completed_sessions=sum(1 for s in sessions if s.is_complete),
                average_score=round(float(sum(scores) / len(scores)), 2)
                if scores
                else None,
                best_score=round(float(max(scores)), 2) if scores else None,
                last_score=round(float(scores[0]), 2) if scores else None,
                last_session_at=sessions[0].started_at if sessions else None,
                risky_answers=risky_total,
                weakest_categories=[row.scenario_type for row in weakness_rows[:3]],
            )
        )
    return analytics


async def weakness_for_participant(
    db: AsyncSession,
    organisation_id: str,
    user_id: str,
    participant_id: str,
) -> list[WeaknessAnalytics]:
    risky = func.sum(case((ScenarioInteraction.decision == "risky", 1), else_=0))
    correct = func.sum(case((ScenarioInteraction.decision == "correct", 1), else_=0))
    total = func.count(ScenarioInteraction.id)
    result = await db.execute(
        select(
            ScenarioInteraction.scenario_id,
            ScenarioInteraction.scenario_type,
            risky.label("risky_answers"),
            correct.label("correct_answers"),
            total.label("total_answers"),
        )
        .join(TrainingSession, ScenarioInteraction.session_id == TrainingSession.id)
        .where(
            TrainingSession.organisation_id == organisation_id,
            (
                (TrainingSession.user_id == user_id)
                | (TrainingSession.participant_id == participant_id)
            ),
        )
        .group_by(ScenarioInteraction.scenario_id, ScenarioInteraction.scenario_type)
        .order_by(desc("risky_answers"), desc("total_answers"))
    )
    return [
        WeaknessAnalytics(
            scenario_id=row.scenario_id,
            scenario_type=row.scenario_type,
            risky_answers=int(row.risky_answers or 0),
            correct_answers=int(row.correct_answers or 0),
            total_answers=int(row.total_answers or 0),
            risk_rate=round(
                (float(row.risky_answers or 0) / float(row.total_answers or 1)) * 100,
                2,
            ),
        )
        for row in result.all()
        if int(row.risky_answers or 0) > 0
    ]
