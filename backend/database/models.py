"""
SENTRY — Database Models
==========================
SQLAlchemy ORM table definitions for the SENTRY system.

Tables:
    TrainingSession     — One record per employee training session
    ScenarioInteraction — One record per scenario within a session
    AssessmentResult    — Pre/post assessment scores per session
    EvaluationLog       — RAG vs baseline comparison records (research)

Design principles:
    - All personal identifiers are anonymised (session_id only, no names)
    - Timestamps are UTC throughout
    - Foreign keys enforce relational integrity
    - The EvaluationLog table feeds directly into Derick's evaluation
      module for the research study

Used by:
    backend/database/connection.py  (Base registration)
    middleware/routes/session_routes.py
    middleware/routes/analytics_routes.py
"""

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
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.connection import Base


def utcnow() -> datetime:
    """Return current UTC time — used as column defaults."""
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    """Generate a new UUID string — used as primary key default."""
    return str(uuid.uuid4())


# ------------------------------------------------------------------
# TrainingSession
# ------------------------------------------------------------------

class TrainingSession(Base):
    """
    Represents one complete training session for one employee.

    Created when an employee starts a session on the SENTRY app.
    Updated when the session ends with final scores and duration.

    Anonymisation: participant_id is an internal code assigned by the
    organisation — SENTRY never stores names or employee IDs.
    """
    __tablename__ = "training_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    participant_id: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )
    # "grounded" or "baseline" — which study condition
    condition: Mapped[str] = mapped_column(String(20), nullable=False)
    organisation_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Duration in seconds — computed on session end
    duration_seconds: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    pre_assessment_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    post_assessment_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    knowledge_gain: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
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

    def __repr__(self) -> str:
        return (
            f"TrainingSession(id={self.id}, "
            f"participant={self.participant_id}, "
            f"condition={self.condition})"
        )


# ------------------------------------------------------------------
# ScenarioInteraction
# ------------------------------------------------------------------

class ScenarioInteraction(Base):
    """
    Records one scenario interaction within a training session.

    One session contains multiple interactions — one per scenario
    presented to the employee. If the employee makes a wrong decision
    and is corrected, correction_loops increments.

    This table provides Timothy's behavioural metrics:
        - Decision accuracy per scenario type
        - Response latency
        - Correction loop count
        - Risky action frequency
    """
    __tablename__ = "scenario_interactions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scenario_id: Mapped[str] = mapped_column(
        String(50), nullable=False
        # e.g. "phishing-01", "usb-drop-02"
    )
    scenario_type: Mapped[str] = mapped_column(
        String(50), nullable=False
        # e.g. "phishing", "usb_drop", "password", "network", "social_engineering"
    )
    # "correct" or "risky"
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    # Raw text of what the employee said/selected
    employee_response: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    # Milliseconds from scenario prompt to employee response
    response_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    # How many times the employee was sent back to retry
    correction_loops: Mapped[int] = mapped_column(Integer, default=0)
    # AI response latency from middleware
    ai_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    # Which documents grounded the AI response (comma-separated)
    ai_sources: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationship
    session: Mapped["TrainingSession"] = relationship(
        back_populates="interactions"
    )

    def __repr__(self) -> str:
        return (
            f"ScenarioInteraction(scenario={self.scenario_id}, "
            f"decision={self.decision})"
        )


# ------------------------------------------------------------------
# AssessmentResult
# ------------------------------------------------------------------

class AssessmentResult(Base):
    """
    Stores pre and post assessment scores for a session.

    The pre-assessment is administered before any training content.
    The post-assessment uses structurally identical questions with
    different surface details to prevent memorisation inflation.

    knowledge_gain = post_score - pre_score
    The main objective targets >= 30% improvement (relative gain).
    """
    __tablename__ = "assessment_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # One assessment record per session
        index=True,
    )
    # Scores as percentages (0.0 – 100.0)
    pre_score: Mapped[float] = mapped_column(Float, nullable=False)
    post_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    knowledge_gain: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    # Relative improvement percentage
    relative_improvement_pct: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
        # ((post - pre) / pre) * 100
    )
    pre_taken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    post_taken_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationship
    session: Mapped["TrainingSession"] = relationship(
        back_populates="assessment"
    )

    def __repr__(self) -> str:
        return (
            f"AssessmentResult(pre={self.pre_score}, "
            f"post={self.post_score}, "
            f"gain={self.knowledge_gain})"
        )


# ------------------------------------------------------------------
# EvaluationLog
# ------------------------------------------------------------------

class EvaluationLog(Base):
    """
    Research evaluation record — one per query pair during the study.

    Stores the automated hallucination and grounding scores computed
    by Derick's HallucinationScorer for each query processed during
    a participant session.

    This table is the bridge between Timothy's session data and
    Derick's evaluation module. It enables:
        - Per-session grounding accuracy tracking
        - Aggregate statistical analysis across all participants
        - t-test and effect size computation in the final report
    """
    __tablename__ = "evaluation_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scenario_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(
        String(20), nullable=False
        # "grounded" or "baseline"
    )
    # AI response text
    response: Mapped[str] = mapped_column(Text, nullable=False)
    # Grounding metrics from HallucinationScorer
    grounding_accuracy: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    hallucination_rate: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    grounding_improvement: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    # Latency metrics
    retrieval_ms: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    generation_ms: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    total_ms: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    # Token usage
    prompt_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    completion_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    # Source documents used
    sources: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
        # Stored as comma-separated string
    )
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Relationship
    session: Mapped["TrainingSession"] = relationship(
        back_populates="eval_logs"
    )

    def __repr__(self) -> str:
        return (
            f"EvaluationLog(mode={self.mode}, "
            f"grounding={self.grounding_accuracy})"
        )