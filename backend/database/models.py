import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.connection import Base


def utcnow() -> datetime:
    """Return current UTC time — used as column defaults."""
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    """Generate a new UUID string — used as primary key default."""
    return str(uuid.uuid4())


# TrainingSession


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    participant_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # "grounded" or "baseline" — which study condition
    condition: Mapped[str] = mapped_column(String(20), nullable=False)
    organisation_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Duration in seconds — computed on session end
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pre_assessment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_assessment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    knowledge_gain: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        # Computed as: post_score - pre_score
    )
    is_complete: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    interactions: Mapped[list["ScenarioInteraction"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    assessment: Mapped[Optional["AssessmentResult"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    eval_logs: Mapped[list["EvaluationLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    user: Mapped[Optional["User"]] = relationship(back_populates="sessions")

    def __repr__(self) -> str:
        return (
            f"TrainingSession(id={self.id}, "
            f"participant={self.participant_id}, "
            f"condition={self.condition})"
        )


# Organisation


class Organisation(Base):
    __tablename__ = "organisations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    canonical_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    users: Mapped[list["User"]] = relationship(back_populates="organisation")

    def __repr__(self) -> str:
        return f"Organisation(canonical_id={self.canonical_id})"


# User


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    participant_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    pin_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    organisation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("organisations.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    department: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    organisation: Mapped[Optional["Organisation"]] = relationship(
        back_populates="users"
    )
    sessions: Mapped[list["TrainingSession"]] = relationship(back_populates="user")

    def __repr__(self) -> str:
        return (
            f"User(participant_id={self.participant_id}, "
            f"role={self.role}, org={self.organisation_id})"
        )


# ScenarioInteraction


class ScenarioInteraction(Base):
    __tablename__ = "scenario_interactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scenario_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    scenario_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    # "correct" or "risky"
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    # Raw text of what the employee said/selected
    employee_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Milliseconds from scenario prompt to employee response
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # How many times the employee was sent back to retry
    correction_loops: Mapped[int] = mapped_column(Integer, default=0)
    # AI response latency from middleware
    ai_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Which documents grounded the AI response (comma-separated)
    ai_sources: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationship
    session: Mapped["TrainingSession"] = relationship(back_populates="interactions")

    def __repr__(self) -> str:
        return (
            f"ScenarioInteraction(scenario={self.scenario_id}, "
            f"decision={self.decision})"
        )


# AssessmentResult


class AssessmentResult(Base):
    __tablename__ = "assessment_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # Scores as percentages (0.0 – 100.0)
    pre_score: Mapped[float] = mapped_column(Float, nullable=False)
    post_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    knowledge_gain: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Relative improvement percentage
    relative_improvement_pct: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        # ((post - pre) / pre) * 100
    )
    pre_taken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    post_taken_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationship
    session: Mapped["TrainingSession"] = relationship(back_populates="assessment")

    def __repr__(self) -> str:
        return (
            f"AssessmentResult(pre={self.pre_score}, "
            f"post={self.post_score}, "
            f"gain={self.knowledge_gain})"
        )


# EvaluationLog


class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scenario_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    # AI response text
    response: Mapped[str] = mapped_column(Text, nullable=False)
    # Grounding metrics from HallucinationScorer
    grounding_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hallucination_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grounding_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Latency metrics
    retrieval_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    generation_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Token usage
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Source documents used
    sources: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        # Stored as comma-separated string
    )
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationship
    session: Mapped["TrainingSession"] = relationship(back_populates="eval_logs")

    def __repr__(self) -> str:
        return (
            f"EvaluationLog(mode={self.mode}, " f"grounding={self.grounding_accuracy})"
        )
