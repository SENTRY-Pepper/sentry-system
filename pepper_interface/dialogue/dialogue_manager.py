"""
SENTRY — Dialogue Manager
===========================
Orchestrates a complete training session. Coordinates between:
    - The state machine (what state are we in?)
    - The scenario library (what content to show?)
    - The middleware client (what does the AI say?)
    - Session logging (what happened?)

This is the central coordinator on Timothy's side.
It is designed to be called from pepper_client.py (NAOqi layer)
but can also be driven from the Android app directly.

Used by: pepper_interface/pepper_client.py
"""

import time
from typing import Optional

from pepper_interface.scenarios import ALL_SCENARIOS
from pepper_interface.scenarios.base_scenario import ScenarioResult
from pepper_interface.dialogue.state_machine import (
    StateMachine,
    SessionContext,
)
from pepper_interface.middleware_client import MiddlewareClient


class DialogueManager:
    """
    Drives one complete SENTRY training session from greeting to results.
    """

    def __init__(self, middleware_base_url: str = "http://localhost:8000"):
        self._state_machine = StateMachine()
        self._client = MiddlewareClient(base_url=middleware_base_url)
        self._scenarios = ALL_SCENARIOS
        self._context: Optional[SessionContext] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        participant_id: str,
        condition: str,
        organisation_id: str,
    ) -> SessionContext:
        """
        Initialise a new session context and register it
        with the middleware backend.
        """
        session_data = self._client.start_session(
            participant_id=participant_id,
            condition=condition,
            organisation_id=organisation_id,
        )

        self._context = SessionContext(
            session_id=session_data["session_id"],
            participant_id=participant_id,
            condition=condition,
            organisation_id=organisation_id,
            total_scenarios=len(self._scenarios),
            session_start_time=time.time(),
        )

        self._state_machine.transition(self._context, "start")
        print(f"[DialogueManager] Session started: {self._context.session_id}")
        return self._context

    def get_greeting(self) -> str:
        """Returns Pepper's opening greeting for the session."""
        greeting = (
            "Hello! I am Pepper, your SENTRY cybersecurity training assistant. "
            "Today we will work through five real-world security scenarios together. "
            "For each one, I will describe a situation and ask what you would do. "
            "There are no trick questions — just realistic situations that employees "
            "face every day. Are you ready to begin?"
        )
        self._state_machine.transition(self._context, "greeting_done")
        return greeting

    def get_current_scenario(self):
        """Return the current scenario object."""
        idx = self._context.current_scenario_index
        if idx >= len(self._scenarios):
            return None
        return self._scenarios[idx]

    def present_scenario(self) -> dict:
        """
        Return scenario data for the current scenario.
        Called when transitioning into SCENARIO_PROMPT state.
        """
        scenario = self.get_current_scenario()
        if not scenario:
            return {}

        self._context.current_scenario_id = scenario.scenario_id
        self._context.scenario_start_time = time.time()

        self._state_machine.transition(self._context, "scenario_shown")

        return {
            "scenario_id": scenario.scenario_id,
            "scenario_type": scenario.scenario_type,
            "title": scenario.title,
            "pepper_intro": scenario.pepper_intro,
            "prompt": scenario.prompt,
            "choices": scenario.choices,
            "scenario_number": self._context.current_scenario_index + 1,
            "total_scenarios": self._context.total_scenarios,
        }

    def process_response(self, choice_id: str = None, free_text: str = None) -> dict:
        """
        Process the employee's response — either a button selection
        or free-text voice input.

        Returns a dict with:
            - decision: "correct" or "risky"
            - pepper_immediate: what Pepper says before the AI responds
            - ai_response: grounded explanation from the middleware
            - sources: documents used for grounding
            - needs_retry: bool
        """
        scenario = self.get_current_scenario()
        if not scenario:
            return {"error": "No active scenario"}

        # Calculate response time
        response_time_ms = int(
            (time.time() - (self._context.scenario_start_time or time.time())) * 1000
        )

        # Evaluate the response
        if choice_id:
            result: ScenarioResult = scenario.evaluate_choice(choice_id)
        elif free_text:
            result: ScenarioResult = scenario.evaluate_response(free_text)
        else:
            return {"error": "No response provided"}

        self._state_machine.transition(self._context, "response_received")
        self._context.current_decision = result.decision

        # Determine if correction loop is needed
        if result.correction_needed:
            self._state_machine.transition(self._context, "evaluation_done")
        else:
            self._state_machine.transition(self._context, "evaluation_done")

        # Get grounded AI response from middleware
        self._state_machine.transition(self._context, "evaluation_done")
        ai_start = time.time()

        try:
            if self._context.condition == "grounded":
                ai_data = self._client.grounded_query(
                    query=result.rag_query,
                    scenario_id=result.scenario_id,
                )
            else:
                ai_data = self._client.baseline_query(
                    query=result.rag_query,
                    scenario_id=result.scenario_id,
                )
        except Exception as e:
            print(f"[DialogueManager] AI query failed: {e}")
            ai_data = {
                "response": "I encountered an issue retrieving additional information. "
                "Please consult your IT security team.",
                "sources": [],
                "total_ms": 0,
            }

        ai_latency_ms = round((time.time() - ai_start) * 1000, 2)
        self._state_machine.transition(self._context, "ai_response_done")

        # Log the interaction
        self._context.log_interaction(
            scenario_id=result.scenario_id,
            decision=result.decision,
            response_time_ms=response_time_ms,
            ai_latency_ms=ai_data.get("total_ms", ai_latency_ms),
            ai_sources=",".join(ai_data.get("sources", [])),
        )

        # Post interaction to middleware
        try:
            self._client.log_interaction(
                session_id=self._context.session_id,
                scenario_id=result.scenario_id,
                scenario_type=result.scenario_type,
                decision=result.decision,
                employee_response=result.employee_response,
                response_time_ms=response_time_ms,
                correction_loops=self._context.correction_loops,
                ai_latency_ms=ai_data.get("total_ms", 0),
                ai_sources=",".join(ai_data.get("sources", [])),
            )
        except Exception as e:
            print(f"[DialogueManager] Interaction log failed: {e}")

        # Immediate Pepper speech (before AI response loads)
        pepper_immediate = (
            scenario.pepper_correct_response
            if result.decision == "correct"
            else scenario.pepper_risky_response
        )

        return {
            "decision": result.decision,
            "pepper_immediate": pepper_immediate,
            "ai_response": ai_data.get("response", ""),
            "sources": ai_data.get("sources", []),
            "needs_retry": result.correction_needed
            and self._context.correction_loops < self._context.max_correction_loops,
            "ai_latency_ms": ai_data.get("total_ms", 0),
        }

    def advance_scenario(self) -> bool:
        """
        Move to the next scenario.
        Returns True if there is a next scenario, False if session is complete.
        """
        self._context.current_scenario_index += 1
        self._state_machine.transition(self._context, "feedback_done")

        if self._context.current_scenario_index >= self._context.total_scenarios:
            return False

        self._state_machine.transition(self._context, "scenario_loaded")
        return True

    def end_session(
        self,
        pre_score: float,
        post_score: float,
    ) -> dict:
        """
        Close the session, compute knowledge gain, and post to middleware.
        Returns the final session summary.
        """
        duration_seconds = int(
            time.time() - (self._context.session_start_time or time.time())
        )

        try:
            summary = self._client.end_session(
                session_id=self._context.session_id,
                pre_score=pre_score,
                post_score=post_score,
                duration_seconds=duration_seconds,
            )
        except Exception as e:
            print(f"[DialogueManager] End session failed: {e}")
            summary = {}

        self._state_machine.transition(self._context, "assessment_done")

        return {
            "session_id": self._context.session_id,
            "participant_id": self._context.participant_id,
            "correct_count": self._context.correct_count,
            "risky_count": self._context.risky_count,
            "accuracy_pct": self._context.accuracy_pct,
            "duration_seconds": duration_seconds,
            "pre_score": pre_score,
            "post_score": post_score,
            "knowledge_gain": summary.get("knowledge_gain"),
            "relative_improvement_pct": summary.get("relative_improvement_pct"),
            "interaction_log": self._context.interaction_log,
        }
