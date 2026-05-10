"""
SENTRY — Base Scenario
========================
Abstract base class that all five cybersecurity training scenarios
inherit from. Defines the contract every scenario must fulfil:
    - A scenario prompt Pepper reads aloud / displays on tablet
    - A set of correct and risky response keywords
    - An evaluate_response() method that classifies employee input
    - A follow-up query string for Derick's RAG pipeline

Used by: pepper_interface/dialogue/dialogue_manager.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScenarioResult:
    """
    The outcome of one employee interaction with a scenario.
    Passed back to the dialogue manager for state transitions
    and to the middleware for session logging.
    """

    scenario_id: str
    scenario_type: str
    decision: str  # "correct" or "risky"
    employee_response: str
    confidence: float  # 0.0 – 1.0 how confident the classifier is
    rag_query: str  # Query sent to Derick's /query endpoint
    correction_needed: bool


class BaseScenario(ABC):
    """
    Abstract base for all SENTRY training scenarios.
    Each scenario encapsulates one attack vector simulation.
    """

    @property
    @abstractmethod
    def scenario_id(self) -> str:
        """Unique identifier e.g. 'phishing-01'"""

    @property
    @abstractmethod
    def scenario_type(self) -> str:
        """Category e.g. 'phishing', 'usb_drop'"""

    @property
    @abstractmethod
    def title(self) -> str:
        """Short display title for the tablet UI"""

    @property
    @abstractmethod
    def prompt(self) -> str:
        """The scenario description Pepper reads to the employee"""

    @property
    @abstractmethod
    def choices(self) -> list[dict]:
        """
        List of choice dicts for the tablet UI.
        Each dict: {"id": str, "text": str, "is_correct": bool}
        """

    @property
    @abstractmethod
    def correct_keywords(self) -> list[str]:
        """Keywords indicating a correct response in free-text input"""

    @property
    @abstractmethod
    def risky_keywords(self) -> list[str]:
        """Keywords indicating a risky response"""

    @property
    @abstractmethod
    def rag_query(self) -> str:
        """Query sent to Derick's middleware for grounded feedback"""

    @property
    @abstractmethod
    def pepper_intro(self) -> str:
        """What Pepper says when introducing this scenario"""

    @property
    @abstractmethod
    def pepper_correct_response(self) -> str:
        """What Pepper says immediately when employee is correct"""

    @property
    @abstractmethod
    def pepper_risky_response(self) -> str:
        """What Pepper says immediately when employee makes risky choice"""

    def evaluate_response(self, employee_response: str) -> ScenarioResult:
        """
        Classify the employee's free-text or button response.
        Checks for correct keywords first, then risky keywords.
        Defaults to risky if neither matches (conservative approach).
        """
        response_lower = employee_response.lower().strip()

        correct_score = sum(1 for kw in self.correct_keywords if kw in response_lower)
        risky_score = sum(1 for kw in self.risky_keywords if kw in response_lower)

        total = correct_score + risky_score
        if total == 0:
            # Cannot classify — treat as risky, conservative default
            decision = "risky"
            confidence = 0.5
        elif correct_score > risky_score:
            decision = "correct"
            confidence = round(correct_score / total, 2)
        else:
            decision = "risky"
            confidence = round(risky_score / total, 2)

        return ScenarioResult(
            scenario_id=self.scenario_id,
            scenario_type=self.scenario_type,
            decision=decision,
            employee_response=employee_response,
            confidence=confidence,
            rag_query=self.rag_query,
            correction_needed=(decision == "risky"),
        )

    def evaluate_choice(self, choice_id: str) -> ScenarioResult:
        """
        Evaluate a tablet button selection by choice_id.
        Simpler and more reliable than keyword matching for button UI.
        """
        selected = next((c for c in self.choices if c["id"] == choice_id), None)
        if not selected:
            decision = "risky"
            employee_response = "unknown choice"
        else:
            decision = "correct" if selected["is_correct"] else "risky"
            employee_response = selected["text"]

        return ScenarioResult(
            scenario_id=self.scenario_id,
            scenario_type=self.scenario_type,
            decision=decision,
            employee_response=employee_response,
            confidence=1.0,  # Button selection is unambiguous
            rag_query=self.rag_query,
            correction_needed=(decision == "risky"),
        )
