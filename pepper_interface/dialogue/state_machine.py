"""
SENTRY — Dialogue State Machine
==================================
Manages the state of a single training session.
Transitions between states based on employee decisions
and AI feedback from Derick's middleware.

States:
    IDLE → GREETING → SCENARIO_PROMPT → AWAITING_INPUT
    → EVALUATING → FEEDBACK → (loop or) SESSION_COMPLETE

Used by: pepper_interface/dialogue/dialogue_manager.py
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


class SessionState(Enum):
    IDLE = auto()
    GREETING = auto()
    PRE_ASSESSMENT = auto()
    SCENARIO_PROMPT = auto()
    AWAITING_INPUT = auto()
    EVALUATING = auto()
    FETCHING_AI_RESPONSE = auto()
    FEEDBACK = auto()
    CORRECTION_LOOP = auto()
    NEXT_SCENARIO = auto()
    POST_ASSESSMENT = auto()
    SESSION_COMPLETE = auto()
    ERROR = auto()


@dataclass
class SessionContext:
    """
    All mutable state for one training session.
    Passed between state handlers so nothing is stored globally.
    """

    session_id: str
    participant_id: str
    condition: str  # "grounded" or "baseline"
    organisation_id: str

    current_state: SessionState = SessionState.IDLE
    current_scenario_index: int = 0
    total_scenarios: int = 5

    # Per-interaction tracking
    current_scenario_id: Optional[str] = None
    current_decision: Optional[str] = None
    correction_loops: int = 0
    max_correction_loops: int = 2

    # Session-level metrics
    correct_count: int = 0
    risky_count: int = 0
    response_times_ms: list = field(default_factory=list)
    interaction_log: list = field(default_factory=list)

    # Assessment scores
    pre_assessment_score: Optional[float] = None
    post_assessment_score: Optional[float] = None

    # Timing
    session_start_time: Optional[float] = None
    scenario_start_time: Optional[float] = None

    @property
    def is_complete(self) -> bool:
        return self.current_state == SessionState.SESSION_COMPLETE

    @property
    def scenarios_remaining(self) -> int:
        return self.total_scenarios - self.current_scenario_index

    @property
    def accuracy_pct(self) -> float:
        total = self.correct_count + self.risky_count
        if total == 0:
            return 0.0
        return round((self.correct_count / total) * 100, 1)

    def log_interaction(
        self,
        scenario_id: str,
        decision: str,
        response_time_ms: int,
        ai_latency_ms: float,
        ai_sources: str,
    ) -> None:
        self.interaction_log.append(
            {
                "scenario_id": scenario_id,
                "decision": decision,
                "response_time_ms": response_time_ms,
                "correction_loops": self.correction_loops,
                "ai_latency_ms": ai_latency_ms,
                "ai_sources": ai_sources,
            }
        )
        if decision == "correct":
            self.correct_count += 1
        else:
            self.risky_count += 1
        self.correction_loops = 0


class StateMachine:
    """
    Drives session state transitions.
    Each method returns the new state after processing.
    """

    def transition(
        self,
        context: SessionContext,
        event: str,
        payload: dict = None,
    ) -> SessionState:
        """
        Process an event and transition to the next state.

        Events:
            "start"           → IDLE → GREETING
            "greeting_done"   → GREETING → PRE_ASSESSMENT
            "assessment_done" → PRE_ASSESSMENT → SCENARIO_PROMPT
            "scenario_shown"  → SCENARIO_PROMPT → AWAITING_INPUT
            "response_received" → AWAITING_INPUT → EVALUATING
            "evaluation_done" → EVALUATING → FETCHING_AI_RESPONSE
            "ai_response_done" → FETCHING_AI_RESPONSE → FEEDBACK
            "feedback_done"   → FEEDBACK → NEXT_SCENARIO or SESSION_COMPLETE
            "retry"           → CORRECTION_LOOP → SCENARIO_PROMPT
            "error"           → any → ERROR
        """
        payload = payload or {}
        state = context.current_state

        transitions = {
            (SessionState.IDLE, "start"): SessionState.GREETING,
            (SessionState.GREETING, "greeting_done"): SessionState.PRE_ASSESSMENT,
            (
                SessionState.PRE_ASSESSMENT,
                "assessment_done",
            ): SessionState.SCENARIO_PROMPT,
            (
                SessionState.SCENARIO_PROMPT,
                "scenario_shown",
            ): SessionState.AWAITING_INPUT,
            (SessionState.AWAITING_INPUT, "response_received"): SessionState.EVALUATING,
            (
                SessionState.EVALUATING,
                "evaluation_done",
            ): SessionState.FETCHING_AI_RESPONSE,
            (
                SessionState.FETCHING_AI_RESPONSE,
                "ai_response_done",
            ): SessionState.FEEDBACK,
            (SessionState.CORRECTION_LOOP, "retry"): SessionState.SCENARIO_PROMPT,
        }

        # Feedback completion — advance or complete
        if state == SessionState.FEEDBACK and event == "feedback_done":
            if context.scenarios_remaining <= 0:
                new_state = SessionState.POST_ASSESSMENT
            else:
                new_state = SessionState.NEXT_SCENARIO

        # After correction check
        elif state == SessionState.EVALUATING and event == "needs_correction":
            if context.correction_loops < context.max_correction_loops:
                context.correction_loops += 1
                new_state = SessionState.CORRECTION_LOOP
            else:
                # Max retries reached — move on with explanation
                new_state = SessionState.FETCHING_AI_RESPONSE

        elif state == SessionState.NEXT_SCENARIO and event == "scenario_loaded":
            new_state = SessionState.SCENARIO_PROMPT

        elif state == SessionState.POST_ASSESSMENT and event == "assessment_done":
            new_state = SessionState.SESSION_COMPLETE

        elif event == "error":
            new_state = SessionState.ERROR

        else:
            new_state = transitions.get((state, event), state)

        context.current_state = new_state
        return new_state
