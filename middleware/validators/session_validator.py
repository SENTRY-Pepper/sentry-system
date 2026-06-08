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

from typing import List, Optional
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
    user_id: Optional[str] = Field(
        default=None,
        max_length=36,
        description="Optional persisted trainee user identifier.",
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

    session_id: str = Field(
        ..., description="The session ID from SessionStartResponse."
    )
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


# ------------------------------------------------------------------
# Organisation and user models
# ------------------------------------------------------------------


class OrganisationCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    canonical_id: Optional[str] = Field(default=None, max_length=50)


class OrganisationResponse(BaseModel):
    id: str
    name: str
    canonical_id: str
    is_active: bool
    created_at: datetime


class UserCreateRequest(BaseModel):
    participant_id: str = Field(..., min_length=2, max_length=50)
    display_name: str = Field(..., min_length=2, max_length=120)
    role: str = Field(..., description="admin, manager, or trainee")
    pin: str = Field(..., min_length=4, max_length=32)
    organisation_id: Optional[str] = Field(default=None, max_length=50)
    department: Optional[str] = Field(default=None, max_length=80)
    position: Optional[str] = Field(default=None, max_length=80)


class UserResponse(BaseModel):
    id: str
    participant_id: str
    display_name: str
    role: str
    organisation_id: Optional[str]
    department: Optional[str]
    position: Optional[str]
    is_active: bool
    created_at: datetime


class UserLoginRequest(BaseModel):
    participant_id: str = Field(..., min_length=2, max_length=50)
    pin: str = Field(..., min_length=4, max_length=32)
    role: str = Field(..., description="admin, manager, or trainee")
    organisation_id: Optional[str] = Field(default=None, max_length=50)


class UserLoginResponse(BaseModel):
    token: str
    user: UserResponse


class TraineeAnalytics(BaseModel):
    user_id: Optional[str]
    participant_id: str
    display_name: str
    department: Optional[str]
    position: Optional[str]
    is_active: bool
    session_count: int
    completed_sessions: int
    average_score: Optional[float]
    best_score: Optional[float]
    last_score: Optional[float]
    last_session_at: Optional[datetime]
    risky_answers: int
    weakest_categories: List[str]


class WeaknessAnalytics(BaseModel):
    scenario_id: str
    scenario_type: str
    risky_answers: int
    correct_answers: int
    total_answers: int
    risk_rate: float


class DepartmentAnalytics(BaseModel):
    department: str
    trainee_count: int
    completed_sessions: int
    average_score: Optional[float]
    risky_answers: int


class ManagerOverviewAnalytics(BaseModel):
    organisation_id: str
    trainee_count: int
    active_trainees: int
    total_sessions: int
    completed_sessions: int
    average_score: Optional[float]
    completion_rate: float
    risky_answers: int
    top_weaknesses: List[WeaknessAnalytics]
    departments: List[DepartmentAnalytics]
