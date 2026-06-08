"""
SENTRY — Pepper Interface Unit Tests
No network calls — tests scenario logic and state machine only.
Run: python tests/unit/test_pepper_interface.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pepper_interface.scenarios import ALL_SCENARIOS, SCENARIO_MAP
from pepper_interface.scenarios.phishing_scenario import PhishingScenario
from pepper_interface.dialogue.state_machine import (
    StateMachine,
    SessionContext,
    SessionState,
)
from pepper_interface.dialogue.dialogue_manager import DialogueManager
from pepper_interface.pepper_client import PepperClient


def test_all_scenarios_load():
    print("=== Test 1: All scenarios load correctly ===")
    assert len(ALL_SCENARIOS) == 5
    for s in ALL_SCENARIOS:
        assert s.scenario_id
        assert s.scenario_type
        assert s.prompt
        assert len(s.choices) >= 3
        assert s.rag_query
        print(f"  Loaded: {s.scenario_id} ({s.scenario_type})")
    print(">>> PASSED")


def test_scenario_map():
    print("\n=== Test 2: Scenario map lookup ===")
    assert "phishing-01" in SCENARIO_MAP
    assert "usb-drop-01" in SCENARIO_MAP
    assert "password-01" in SCENARIO_MAP
    assert "social-engineering-01" in SCENARIO_MAP
    assert "network-01" in SCENARIO_MAP
    print(">>> PASSED")


def test_choice_evaluation_correct():
    print("\n=== Test 3: Correct choice evaluation ===")
    scenario = PhishingScenario()
    result = scenario.evaluate_choice("report")
    assert result.decision == "correct"
    assert result.confidence == 1.0
    assert not result.correction_needed
    print(f"  Decision: {result.decision} | Confidence: {result.confidence}")
    print(">>> PASSED")


def test_choice_evaluation_risky():
    print("\n=== Test 4: Risky choice evaluation ===")
    scenario = PhishingScenario()
    result = scenario.evaluate_choice("click")
    assert result.decision == "risky"
    assert result.correction_needed
    print(
        f"  Decision: {result.decision} | Correction needed: {result.correction_needed}"
    )
    print(">>> PASSED")


def test_free_text_evaluation():
    print("\n=== Test 5: Free text keyword evaluation ===")
    scenario = PhishingScenario()

    correct_result = scenario.evaluate_response(
        "I would report it to the IT team immediately"
    )
    assert correct_result.decision == "correct"
    print(f"  Correct response → {correct_result.decision}")

    risky_result = scenario.evaluate_response(
        "I would click the link to verify my account"
    )
    assert risky_result.decision == "risky"
    print(f"  Risky response → {risky_result.decision}")
    print(">>> PASSED")


def test_state_machine_transitions():
    print("\n=== Test 6: State machine transitions ===")
    sm = StateMachine()
    ctx = SessionContext(
        session_id="test-session",
        participant_id="TEST_P001",
        condition="grounded",
        organisation_id="TEST_ORG",
    )

    assert ctx.current_state == SessionState.IDLE

    sm.transition(ctx, "start")
    assert ctx.current_state == SessionState.GREETING
    print(f"  IDLE → start → {ctx.current_state.name}")

    sm.transition(ctx, "greeting_done")
    assert ctx.current_state == SessionState.PRE_ASSESSMENT
    print(f"  GREETING → greeting_done → {ctx.current_state.name}")

    sm.transition(ctx, "assessment_done")
    assert ctx.current_state == SessionState.SCENARIO_PROMPT
    print(f"  PRE_ASSESSMENT → assessment_done → {ctx.current_state.name}")

    sm.transition(ctx, "scenario_shown")
    assert ctx.current_state == SessionState.AWAITING_INPUT
    print(f"  SCENARIO_PROMPT → scenario_shown → {ctx.current_state.name}")

    print(">>> PASSED")


def test_session_context_metrics():
    print("\n=== Test 7: Session context metric tracking ===")
    ctx = SessionContext(
        session_id="test-session",
        participant_id="TEST_P001",
        condition="grounded",
        organisation_id="TEST_ORG",
    )

    ctx.log_interaction(
        "phishing-01",
        "correct",
        4200,
        6800.0,
        "Computer-Misuse-and-Cybercrimes-Act.pdf",
    )
    ctx.log_interaction("usb-drop-01", "risky", 2100, 5200.0, "A08_2025.md")
    ctx.log_interaction("password-01", "correct", 3800, 7100.0, "A07_2025.md")

    assert ctx.correct_count == 2
    assert ctx.risky_count == 1
    assert ctx.accuracy_pct == round((2 / 3) * 100, 1)
    assert len(ctx.interaction_log) == 3

    print(f"  Correct: {ctx.correct_count} | Risky: {ctx.risky_count}")
    print(f"  Accuracy: {ctx.accuracy_pct}%")
    print(f"  Interactions logged: {len(ctx.interaction_log)}")
    print(">>> PASSED")


def test_pepper_voice_choice_mapping():
    print("\n=== Test 8: Pepper voice choice mapping ===")
    pepper = PepperClient("127.0.0.1", simulation=True)
    scenario = PhishingScenario()
    scenario_data = {"choices": scenario.choices}

    vocabulary, phrase_map = pepper._choice_vocabulary_and_map(scenario_data)

    assert "option a" in vocabulary
    assert phrase_map["option a"] == ("choice", "click")
    assert phrase_map["choice c"] == ("choice", "report")
    assert phrase_map["question"] == ("question", None)
    assert phrase_map["repeat"] == ("repeat", None)
    print(">>> PASSED")


def test_dialogue_manager_spoken_question_routes_to_grounded_query():
    print("\n=== Test 9: Spoken question routes through middleware client ===")

    class FakeClient:
        def __init__(self):
            self.grounded_calls = []
            self.baseline_calls = []

        def grounded_query(self, query, scenario_id=None):
            self.grounded_calls.append((query, scenario_id))
            return {
                "response": "Grounded answer",
                "sources": ["OWASP"],
                "total_ms": 42,
            }

        def baseline_query(self, query, scenario_id=None):
            self.baseline_calls.append((query, scenario_id))
            return {
                "response": "Baseline answer",
                "sources": [],
                "total_ms": 24,
            }

    manager = DialogueManager()
    fake_client = FakeClient()
    manager._client = fake_client
    manager._context = SessionContext(
        session_id="test-session",
        participant_id="TEST_P001",
        condition="grounded",
        organisation_id="TEST_ORG",
    )
    manager._context.current_scenario_id = "phishing-01"

    answer = manager.answer_question("what is phishing")

    assert answer["response"] == "Grounded answer"
    assert answer["sources"] == ["OWASP"]
    assert answer["latency_ms"] == 42
    assert fake_client.grounded_calls == [("what is phishing", "phishing-01")]
    assert fake_client.baseline_calls == []
    print(">>> PASSED")


if __name__ == "__main__":
    test_all_scenarios_load()
    test_scenario_map()
    test_choice_evaluation_correct()
    test_choice_evaluation_risky()
    test_free_text_evaluation()
    test_state_machine_transitions()
    test_session_context_metrics()
    test_pepper_voice_choice_mapping()
    test_dialogue_manager_spoken_question_routes_to_grounded_query()
    print("\n" + "=" * 60)
    print("All pepper interface tests PASSED")
