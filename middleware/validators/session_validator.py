"""
SENTRY — Session Request & Response Validators
================================================
Pydantic models for all session and analytics endpoints.
These define the exact JSON contract between Timothy's Android
app and the FastAPI middleware.

Used by:
    middleware/routes/session_routes.py
    middleware/routes/analytics_routes.py
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ------------------------------------------------------------------
# Session models
# ------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    """
    Sent by the Android app when an employee begins a session.
    """
    participant_id: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Anonymised participant code assigned by organisation.",
        examples=["P001", "EMP_A3"],
    )
    condition: str = Field(
        ...,
        description="Study condition: 'grounded' or 'baseline'.",
        examples=["grounded"],
    )
    organisation_id: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Organisation identifier for group-level analytics.",
    )

    @classmethod
    def validate_condition(cls, v):
        if v not in ("grounded", "baseline"):
            raise ValueError("condition must be 'grounded' or 'baseline'.")
        return v


class SessionStartResponse(BaseModel):
    """Returned to the app after a session is successfully created."""
    session_id: str
    participant_id: str
    condition: str
    started_at: datetime


class SessionEndRequest(BaseModel):
    """
    Sent by the Android app when an employee completes a session.
    """
    session_id: str = Field(..., description="The session ID from SessionStartResponse.")
    pre_assessment_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Pre-training assessment score (0-100).",
    )
    post_assessment_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Post-training assessment score (0-100).",
    )
    duration_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total session duration in seconds.",
    )


class SessionEndResponse(BaseModel):
    """Returned after a session is successfully closed."""
    session_id: str
    participant_id: str
    duration_seconds: Optional[int]
    pre_assessment_score: Optional[float]
    post_assessment_score: Optional[float]
    knowledge_gain: Optional[float]
    relative_improvement_pct: Optional[float]
    is_complete: bool


# ------------------------------------------------------------------
# Scenario interaction models
# ------------------------------------------------------------------

class InteractionLogRequest(BaseModel):
    """
    Sent by the app after each scenario interaction completes.
    Logs the employee's decision and associated metrics.
    """
    session_id: str
    scenario_id: str = Field(
        ...,
        examples=["phishing-01", "usb-drop-02"],
    )
    scenario_type: str = Field(
        ...,
        examples=["phishing", "usb_drop", "password", "network", "social_engineering"],
    )
    decision: str = Field(
        ...,
        description="'correct' or 'risky'",
    )
    employee_response: Optional[str] = Field(
        default=None,
        max_length=500,
    )
    response_time_ms: Optional[int] = Field(default=None, ge=0)
    correction_loops: int = Field(default=0, ge=0)
    ai_latency_ms: Optional[float] = Field(default=None, ge=0)
    ai_sources: Optional[str] = Field(
        default=None,
        description="Comma-separated source document names.",
    )


class InteractionLogResponse(BaseModel):
    """Confirmation of a logged interaction."""
    interaction_id: str
    session_id: str
    scenario_id: str
    decision: str
    logged: bool


# ------------------------------------------------------------------
# Evaluation log models
# ------------------------------------------------------------------

class EvaluationLogRequest(BaseModel):
    """
    Sent after each RAG query pair is scored by the evaluation module.
    Links grounding metrics to a session for the research study.
    """
    session_id: str
    scenario_id: Optional[str] = None
    query: str
    mode: str  # "grounded" or "baseline"
    response: str
    grounding_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hallucination_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    grounding_improvement: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    total_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    sources: Optional[str] = None


# ------------------------------------------------------------------
# Analytics models
# ------------------------------------------------------------------

class SessionSummary(BaseModel):
    """Summary of a single session — returned in analytics responses."""
    session_id: str
    participant_id: str
    condition: str
    is_complete: bool
    duration_seconds: Optional[int]
    pre_assessment_score: Optional[float]
    post_assessment_score: Optional[float]
    knowledge_gain: Optional[float]
    started_at: datetime


class OrganisationAnalytics(BaseModel):
    """Aggregated analytics for an organisation's training sessions."""
    organisation_id: str
    total_sessions: int
    completed_sessions: int
    mean_pre_score: Optional[float]
    mean_post_score: Optional[float]
    mean_knowledge_gain: Optional[float]
    mean_grounding_accuracy: Optional[float]
    mean_hallucination_rate: Optional[float]


class StudyAnalytics(BaseModel):
    """
    Aggregated research study analytics across all participants.
    Used for the evaluation report and statistical analysis.
    """
    total_sessions: int
    grounded_sessions: int
    baseline_sessions: int
    mean_grounding_accuracy_grounded: Optional[float]
    mean_grounding_accuracy_baseline: Optional[float]
    mean_hallucination_rate_grounded: Optional[float]
    mean_hallucination_rate_baseline: Optional[float]
    mean_grounding_improvement: Optional[float]
    mean_knowledge_gain_grounded: Optional[float]
    mean_knowledge_gain_baseline: Optional[float]