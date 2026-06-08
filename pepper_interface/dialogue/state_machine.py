# -*- coding: utf-8 -*-
"""
SENTRY - Dialogue State Machine
===============================
Python 2.7-compatible session state machine for Pepper/NAOqi runtime.
"""


class _State(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "SessionState.%s" % self.name


class SessionState(object):
    IDLE = _State("IDLE")
    GREETING = _State("GREETING")
    PRE_ASSESSMENT = _State("PRE_ASSESSMENT")
    SCENARIO_PROMPT = _State("SCENARIO_PROMPT")
    AWAITING_INPUT = _State("AWAITING_INPUT")
    EVALUATING = _State("EVALUATING")
    FETCHING_AI_RESPONSE = _State("FETCHING_AI_RESPONSE")
    FEEDBACK = _State("FEEDBACK")
    CORRECTION_LOOP = _State("CORRECTION_LOOP")
    NEXT_SCENARIO = _State("NEXT_SCENARIO")
    POST_ASSESSMENT = _State("POST_ASSESSMENT")
    SESSION_COMPLETE = _State("SESSION_COMPLETE")
    ERROR = _State("ERROR")


class SessionContext(object):
    """
    All mutable state for one training session.
    Passed between state handlers so nothing is stored globally.
    """

    def __init__(
        self,
        session_id,
        participant_id,
        condition,
        organisation_id,
        current_state=SessionState.IDLE,
        current_scenario_index=0,
        total_scenarios=5,
        session_start_time=None,
    ):
        self.session_id = session_id
        self.participant_id = participant_id
        self.condition = condition
        self.organisation_id = organisation_id
        self.current_state = current_state
        self.current_scenario_index = current_scenario_index
        self.total_scenarios = total_scenarios

        self.current_scenario_id = None
        self.current_decision = None
        self.correction_loops = 0
        self.max_correction_loops = 2

        self.correct_count = 0
        self.risky_count = 0
        self.response_times_ms = []
        self.interaction_log = []

        self.pre_assessment_score = None
        self.post_assessment_score = None

        self.session_start_time = session_start_time
        self.scenario_start_time = None

    @property
    def is_complete(self):
        return self.current_state == SessionState.SESSION_COMPLETE

    @property
    def scenarios_remaining(self):
        return self.total_scenarios - self.current_scenario_index

    @property
    def accuracy_pct(self):
        total = self.correct_count + self.risky_count
        if total == 0:
            return 0.0
        return round((float(self.correct_count) / float(total)) * 100, 1)

    def log_interaction(
        self,
        scenario_id,
        decision,
        response_time_ms,
        ai_latency_ms,
        ai_sources,
    ):
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


class StateMachine(object):
    """
    Drives session state transitions.
    """

    def transition(self, context, event, payload=None):
        payload = payload or {}
        state = context.current_state

        transitions = {
            (SessionState.IDLE, "start"): SessionState.GREETING,
            (SessionState.GREETING, "greeting_done"): SessionState.PRE_ASSESSMENT,
            (SessionState.PRE_ASSESSMENT, "assessment_done"): SessionState.SCENARIO_PROMPT,
            (SessionState.SCENARIO_PROMPT, "scenario_shown"): SessionState.AWAITING_INPUT,
            (SessionState.AWAITING_INPUT, "response_received"): SessionState.EVALUATING,
            (SessionState.EVALUATING, "evaluation_done"): SessionState.FETCHING_AI_RESPONSE,
            (SessionState.FETCHING_AI_RESPONSE, "ai_response_done"): SessionState.FEEDBACK,
            (SessionState.CORRECTION_LOOP, "retry"): SessionState.SCENARIO_PROMPT,
        }

        if state == SessionState.FEEDBACK and event == "feedback_done":
            if context.scenarios_remaining <= 0:
                new_state = SessionState.POST_ASSESSMENT
            else:
                new_state = SessionState.NEXT_SCENARIO
        elif state == SessionState.EVALUATING and event == "needs_correction":
            if context.correction_loops < context.max_correction_loops:
                context.correction_loops += 1
                new_state = SessionState.CORRECTION_LOOP
            else:
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
