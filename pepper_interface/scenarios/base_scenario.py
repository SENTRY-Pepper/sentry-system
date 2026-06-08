# -*- coding: utf-8 -*-
"""
SENTRY - Base Scenario
======================
Python 2.7-compatible scenario contract for Pepper/NAOqi runtime.
"""

from abc import ABCMeta, abstractmethod


class ScenarioResult(object):
    def __init__(
        self,
        scenario_id,
        scenario_type,
        decision,
        employee_response,
        confidence,
        rag_query,
        correction_needed,
    ):
        self.scenario_id = scenario_id
        self.scenario_type = scenario_type
        self.decision = decision
        self.employee_response = employee_response
        self.confidence = confidence
        self.rag_query = rag_query
        self.correction_needed = correction_needed


class BaseScenario(object):
    """
    Abstract base for all SENTRY training scenarios.
    Each scenario encapsulates one attack vector simulation.
    """

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def scenario_id(self):
        """Unique identifier e.g. 'phishing-01'"""

    @property
    @abstractmethod
    def scenario_type(self):
        """Category e.g. 'phishing', 'usb_drop'"""

    @property
    @abstractmethod
    def title(self):
        """Short display title for the tablet UI"""

    @property
    @abstractmethod
    def prompt(self):
        """The scenario description Pepper reads to the employee"""

    @property
    @abstractmethod
    def choices(self):
        """
        List of choice dicts for the tablet UI.
        Each dict: {"id": str, "text": str, "is_correct": bool}
        """

    @property
    @abstractmethod
    def correct_keywords(self):
        """Keywords indicating a correct response in free-text input"""

    @property
    @abstractmethod
    def risky_keywords(self):
        """Keywords indicating a risky response"""

    @property
    @abstractmethod
    def rag_query(self):
        """Query sent to the middleware for grounded feedback"""

    @property
    @abstractmethod
    def pepper_intro(self):
        """What Pepper says when introducing this scenario"""

    @property
    @abstractmethod
    def pepper_correct_response(self):
        """What Pepper says immediately when employee is correct"""

    @property
    @abstractmethod
    def pepper_risky_response(self):
        """What Pepper says immediately when employee makes risky choice"""

    def evaluate_response(self, employee_response):
        """
        Classify the employee's free-text or spoken response.
        Checks for correct keywords first, then risky keywords.
        Defaults to risky if neither matches.
        """
        response_lower = employee_response.lower().strip()

        correct_score = sum(1 for kw in self.correct_keywords if kw in response_lower)
        risky_score = sum(1 for kw in self.risky_keywords if kw in response_lower)

        total = correct_score + risky_score
        if total == 0:
            decision = "risky"
            confidence = 0.5
        elif correct_score > risky_score:
            decision = "correct"
            confidence = round(float(correct_score) / float(total), 2)
        else:
            decision = "risky"
            confidence = round(float(risky_score) / float(total), 2)

        return ScenarioResult(
            scenario_id=self.scenario_id,
            scenario_type=self.scenario_type,
            decision=decision,
            employee_response=employee_response,
            confidence=confidence,
            rag_query=self.rag_query,
            correction_needed=(decision == "risky"),
        )

    def evaluate_choice(self, choice_id):
        """
        Evaluate a tablet button or spoken option selection by choice_id.
        """
        selected = None
        for choice in self.choices:
            if choice["id"] == choice_id:
                selected = choice
                break

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
            confidence=1.0,
            rag_query=self.rag_query,
            correction_needed=(decision == "risky"),
        )
