# -*- coding: utf-8 -*-
"""
SENTRY - Pepper Robot Client
==============================
NAOqi Python 2.7 entry point for the Pepper robot.
Controls speech synthesis, gesture execution, and tablet display.

IMPORTANT: This file must remain Python 2.7 compatible.
    - No f-strings (use .format() or % formatting)
    - No type hints
    - No walrus operator
    - No pathlib

Run on Pepper directly or on a laptop with NAOqi SDK installed.

Usage:
    python pepper_interface/pepper_client.py --ip 192.168.x.x --port 9559
    python pepper_interface/pepper_client.py --simulation
"""

import sys
import os
import time
import argparse

# Add project root to path so pepper_interface and config are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# NAOqi SDK imports — only available on Pepper or with SDK installed
try:
    import qi  # noqa: F401

    NAOQI_AVAILABLE = True
except ImportError:
    NAOQI_AVAILABLE = False
    print("[PepperClient] NAOqi SDK not found. Running in simulation mode.")

# ------------------------------------------------------------------
# Gesture mappings — NAOqi animation names
# These are built-in Pepper animations from the NAOqi animation library
# ------------------------------------------------------------------
GESTURES = {
    "greeting": "animations/Stand/Gestures/Hey_1",
    "thinking": "animations/Stand/Gestures/Think_1",
    "correct": "animations/Stand/Gestures/Enthusiastic_4",
    "wrong": "animations/Stand/Gestures/No_3",
    "explaining": "animations/Stand/Gestures/Explain_1",
    "encouraging": "animations/Stand/Gestures/Encouragement_1",
    "attention": "animations/Stand/Gestures/ShowTablet_1",
    "farewell": "animations/Stand/Gestures/BowShort_1",
}

# ------------------------------------------------------------------
# Tablet HTML templates
# Served from Pepper's local tablet browser
# ------------------------------------------------------------------
TABLET_WELCOME_HTML = """
<html>
<head>
<style>
  body {{
    background: #085041;
    font-family: Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
    color: #E1F5EE;
  }}
  h1 {{ font-size: 42px; margin-bottom: 10px; }}
  p  {{ font-size: 20px; color: #9FE1CB; }}
</style>
</head>
<body>
  <h1>SENTRY</h1>
  <p>Cybersecurity Training</p>
  <p style="margin-top:30px;font-size:16px;">Powered by grounded AI</p>
</body>
</html>
"""

TABLET_SCENARIO_HTML = """
<html>
<head>
<style>
  body {{
    background: #F8FAF9;
    font-family: Arial, sans-serif;
    padding: 24px;
    margin: 0;
  }}
  .header {{
    background: #085041;
    color: #E1F5EE;
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 18px;
    font-size: 16px;
  }}
  .scenario-number {{
    font-size: 13px;
    color: #9FE1CB;
    margin-bottom: 4px;
  }}
  .prompt {{
    font-size: 17px;
    color: #1A1A1A;
    line-height: 1.6;
    margin-bottom: 20px;
    background: white;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #E0E0E0;
  }}
  .choice {{
    background: white;
    border: 1.5px solid #E0E0E0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 15px;
    color: #1A1A1A;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .choice:active {{ background: #E1F5EE; border-color: #0F6E56; }}
  .choice-letter {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #085041;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: bold;
    flex-shrink: 0;
  }}
</style>
</head>
<body>
  <div class="header">
    <div class="scenario-number">Scenario {scenario_number} of {total_scenarios} &mdash; {scenario_type}</div>
    <div style="font-size:18px;font-weight:bold;">{title}</div>
  </div>
  <div class="prompt">{prompt}</div>
  {choices_html}
</body>
</html>
"""

TABLET_FEEDBACK_HTML = """
<html>
<head>
<style>
  body {{
    background: #F8FAF9;
    font-family: Arial, sans-serif;
    padding: 24px;
    margin: 0;
  }}
  .result-banner {{
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 18px;
    font-size: 18px;
    font-weight: bold;
  }}
  .correct {{ background: #E1F5EE; color: #085041; border: 1px solid #9FE1CB; }}
  .risky   {{ background: #FCEBEB; color: #791F1F; border: 1px solid #F7C1C1; }}
  .ai-response {{
    background: white;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 16px;
    font-size: 15px;
    line-height: 1.7;
    color: #1A1A1A;
    margin-bottom: 16px;
  }}
  .sources {{
    font-size: 12px;
    color: #6B6B6B;
    margin-top: 10px;
  }}
  .source-tag {{
    display: inline-block;
    background: #E6F1FB;
    color: #0C447C;
    padding: 3px 8px;
    border-radius: 20px;
    margin: 2px;
    font-size: 11px;
  }}
</style>
</head>
<body>
  <div class="result-banner {css_class}">
    {result_icon} {result_label}
  </div>
  <div class="ai-response">
    {ai_response}
    <div class="sources">
      <strong>Sources:</strong>
      {sources_html}
    </div>
  </div>
</body>
</html>
"""

TABLET_RESULTS_HTML = """
<html>
<head>
<style>
  body {{
    background: #085041;
    font-family: Arial, sans-serif;
    padding: 28px;
    margin: 0;
    color: #E1F5EE;
  }}
  h2 {{ font-size: 24px; margin-bottom: 6px; }}
  .score-row {{
    display: flex;
    gap: 14px;
    margin: 20px 0;
  }}
  .score-card {{
    flex: 1;
    background: rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }}
  .score-number {{ font-size: 30px; font-weight: bold; color: #5DCAA5; }}
  .score-label  {{ font-size: 13px; color: #9FE1CB; margin-top: 4px; }}
  .breakdown {{ background: rgba(255,255,255,0.08); border-radius: 10px; padding: 14px; }}
  .row {{ display: flex; justify-content: space-between; padding: 8px 0;
          border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 14px; }}
  .correct-tag {{ color: #5DCAA5; font-weight: bold; }}
  .risky-tag   {{ color: #F7A8A8; font-weight: bold; }}
</style>
</head>
<body>
  <h2>Session Complete</h2>
  <p style="color:#9FE1CB;font-size:15px;">{participant_id}</p>
  <div class="score-row">
    <div class="score-card">
      <div class="score-number">{accuracy_pct}%</div>
      <div class="score-label">Accuracy</div>
    </div>
    <div class="score-card">
      <div class="score-number">{correct_count}/{total}</div>
      <div class="score-label">Correct</div>
    </div>
    <div class="score-card">
      <div class="score-number">{duration_min}m</div>
      <div class="score-label">Duration</div>
    </div>
  </div>
  <div class="breakdown">
    {breakdown_rows}
  </div>
</body>
</html>
"""


class PepperClient(object):
    """
    Controls the Pepper robot's speech, gestures, and tablet display
    during a SENTRY training session.

    In simulation mode (no NAOqi), all speech and gesture calls
    print to console — useful for development without the robot.
    """

    def __init__(self, ip, port=9559, simulation=False):
        self.ip = ip
        self.port = port
        self.simulation = simulation or not NAOQI_AVAILABLE

        self._tts = None
        self._motion = None
        self._tablet = None
        self._anim = None
        self._speech_reco = None

        if not self.simulation:
            self._connect()

    def _connect(self):
        """Establish NAOqi session and initialise all service proxies."""
        try:
            session = qi.Session()
            session.connect("tcp://{}:{}".format(self.ip, self.port))

            self._tts = session.service("ALTextToSpeech")
            self._motion = session.service("ALMotion")
            self._tablet = session.service("ALTabletService")
            self._anim = session.service("ALAnimationPlayer")
            self._speech_reco = session.service("ALSpeechRecognition")

            # Set language and speech parameters
            self._tts.setLanguage("English")
            self._tts.setParameter("speed", 85)  # Slightly slower for clarity
            self._tts.setParameter("pitchShift", 1.0)

            print(
                "[PepperClient] Connected to Pepper at {}:{}".format(self.ip, self.port)
            )
        except Exception as e:
            print("[PepperClient] Connection failed: {}".format(e))
            self.simulation = True
            print("[PepperClient] Falling back to simulation mode.")

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------

    def say(self, text, animated=False):
        """
        Speak text aloud via Pepper's TTS.
        animated=True plays a gesture alongside the speech.
        """
        if self.simulation:
            print("[Pepper SAYS] {}".format(text))
            return

        try:
            if animated:
                self._tts.say("\\style=didactic\\ " + text)
            else:
                self._tts.say(text)
        except Exception as e:
            print("[PepperClient] TTS error: {}".format(e))

    def say_and_gesture(self, text, gesture_key):
        """Speak while simultaneously playing a gesture animation."""
        if self.simulation:
            print("[Pepper SAYS + GESTURE:{}] {}".format(gesture_key, text))
            return

        gesture = GESTURES.get(gesture_key, GESTURES["explaining"])
        try:
            # Run gesture asynchronously while TTS plays
            future = self._anim.run(gesture, _async=True)
            self._tts.say(text)
            future.wait()
        except Exception as e:
            print("[PepperClient] Say+gesture error: {}".format(e))
            self.say(text)

    # ------------------------------------------------------------------
    # Gestures
    # ------------------------------------------------------------------

    def gesture(self, gesture_key):
        """Play a named gesture animation."""
        if self.simulation:
            print("[Pepper GESTURE] {}".format(gesture_key))
            return
        anim_name = GESTURES.get(gesture_key, GESTURES["explaining"])
        try:
            self._anim.run(anim_name)
        except Exception as e:
            print("[PepperClient] Gesture error: {}".format(e))

    # ------------------------------------------------------------------
    # Tablet display
    # ------------------------------------------------------------------

    def show_welcome(self):
        """Display the SENTRY welcome screen on the tablet."""
        if self.simulation:
            print("[Pepper TABLET] Showing welcome screen")
            return
        try:
            self._tablet.showWebview()
            self._tablet.loadUrl("data:text/html," + TABLET_WELCOME_HTML)
        except Exception as e:
            print("[PepperClient] Tablet welcome error: {}".format(e))

    def show_scenario(self, scenario_data):
        """
        Display a scenario on the tablet.
        scenario_data: dict from DialogueManager.present_scenario()
        """
        letters = ["A", "B", "C", "D"]
        choices_html = ""
        for i, choice in enumerate(scenario_data.get("choices", [])):
            letter = letters[i] if i < len(letters) else str(i + 1)
            choices_html += (
                "<div class='choice'>"
                "<div class='choice-letter'>{letter}</div>"
                "{text}"
                "</div>"
            ).format(letter=letter, text=choice["text"])

        html = TABLET_SCENARIO_HTML.format(
            scenario_number=scenario_data.get("scenario_number", 1),
            total_scenarios=scenario_data.get("total_scenarios", 5),
            scenario_type=scenario_data.get("scenario_type", "")
            .replace("_", " ")
            .title(),
            title=scenario_data.get("title", ""),
            prompt=scenario_data.get("prompt", ""),
            choices_html=choices_html,
        )

        if self.simulation:
            print(
                "[Pepper TABLET] Showing scenario: {}".format(
                    scenario_data.get("title", "")
                )
            )
            return

        try:
            self._tablet.showWebview()
            self._tablet.loadUrl("data:text/html," + html)
        except Exception as e:
            print("[PepperClient] Tablet scenario error: {}".format(e))

    def show_feedback(self, decision, ai_response, sources):
        """
        Display the AI grounded feedback on the tablet.
        """
        is_correct = decision == "correct"
        css_class = "correct" if is_correct else "risky"
        result_icon = "&#10003;" if is_correct else "&#10007;"
        result_label = "Correct decision" if is_correct else "Risky decision"

        sources_html = ""
        for src in sources:
            sources_html += "<span class='source-tag'>{}</span>".format(src)

        html = TABLET_FEEDBACK_HTML.format(
            css_class=css_class,
            result_icon=result_icon,
            result_label=result_label,
            ai_response=ai_response,
            sources_html=sources_html if sources_html else "General knowledge",
        )

        if self.simulation:
            print(
                "[Pepper TABLET] Showing feedback: {} | Sources: {}".format(
                    decision, sources
                )
            )
            return

        try:
            self._tablet.showWebview()
            self._tablet.loadUrl("data:text/html," + html)
        except Exception as e:
            print("[PepperClient] Tablet feedback error: {}".format(e))

    def show_results(self, session_summary):
        """
        Display the final results screen on the tablet.
        session_summary: dict from DialogueManager.end_session()
        """
        log = session_summary.get("interaction_log", [])
        breakdown_rows = ""
        for item in log:
            tag_class = "correct-tag" if item["decision"] == "correct" else "risky-tag"
            tag_text = "Correct" if item["decision"] == "correct" else "Risky"
            breakdown_rows += (
                "<div class='row'>"
                "<span>{scenario_id}</span>"
                "<span class='{tag_class}'>{tag_text}</span>"
                "</div>"
            ).format(
                scenario_id=item["scenario_id"],
                tag_class=tag_class,
                tag_text=tag_text,
            )

        duration_min = max(1, session_summary.get("duration_seconds", 0) // 60)
        total = session_summary.get("correct_count", 0) + session_summary.get(
            "risky_count", 0
        )

        html = TABLET_RESULTS_HTML.format(
            participant_id=session_summary.get("participant_id", ""),
            accuracy_pct=session_summary.get("accuracy_pct", 0),
            correct_count=session_summary.get("correct_count", 0),
            total=total,
            duration_min=duration_min,
            breakdown_rows=breakdown_rows,
        )

        if self.simulation:
            print(
                "[Pepper TABLET] Showing results for {}".format(
                    session_summary.get("participant_id", "")
                )
            )
            return

        try:
            self._tablet.showWebview()
            self._tablet.loadUrl("data:text/html," + html)
        except Exception as e:
            print("[PepperClient] Tablet results error: {}".format(e))

    def hide_tablet(self):
        """Clear the tablet display."""
        if self.simulation:
            print("[Pepper TABLET] Cleared")
            return
        try:
            self._tablet.hideWebview()
        except Exception as e:
            print("[PepperClient] Tablet hide error: {}".format(e))


# ------------------------------------------------------------------
# Full session runner
# Wires PepperClient with DialogueManager for a complete session
# ------------------------------------------------------------------


def run_sentry_session(
    pepper_ip,
    pepper_port,
    middleware_url,
    participant_id,
    condition,
    organisation_id,
    pre_score,
    simulation=False,
):
    """
    Run a complete SENTRY training session on Pepper.
    """
    from pepper_interface.dialogue.dialogue_manager import DialogueManager

    pepper = PepperClient(pepper_ip, pepper_port, simulation=simulation)
    manager = DialogueManager(middleware_base_url=middleware_url)

    print("[Session] Starting session for {}".format(participant_id))

    # 1. Welcome screen
    pepper.show_welcome()
    pepper.say_and_gesture(
        "Hello! I am Pepper, your SENTRY cybersecurity training assistant. "
        "Welcome to today's session.",
        "greeting",
    )
    time.sleep(1)

    # 2. Start session with middleware
    manager.start_session(
        participant_id=participant_id,
        condition=condition,
        organisation_id=organisation_id,
    )

    # 3. Greeting
    greeting = manager.get_greeting()
    pepper.say(greeting)
    time.sleep(2)

    # 4. Run all scenarios
    for i in range(len(manager._scenarios)):
        scenario_data = manager.present_scenario()
        if not scenario_data:
            break

        # Intro speech and show on tablet
        pepper.say_and_gesture(scenario_data["pepper_intro"], "attention")
        time.sleep(1)
        pepper.show_scenario(scenario_data)
        pepper.say(scenario_data["prompt"])

        # In real deployment: wait for tablet button tap from the Android app
        # In simulation: pick the first correct choice automatically
        scenario_obj = manager.get_current_scenario()
        correct_choices = [c for c in scenario_obj.choices if c["is_correct"]]
        simulated_choice = (
            correct_choices[0]["id"]
            if correct_choices
            else scenario_obj.choices[0]["id"]
        )

        print("[Session] Simulated choice: {}".format(simulated_choice))
        pepper.say_and_gesture("I am thinking about your response...", "thinking")

        # 5. Process response and get AI feedback
        feedback = manager.process_response(choice_id=simulated_choice)

        # Immediate Pepper speech
        pepper.say_and_gesture(
            feedback["pepper_immediate"],
            "correct" if feedback["decision"] == "correct" else "wrong",
        )
        time.sleep(1)

        # Show grounded response on tablet
        pepper.show_feedback(
            decision=feedback["decision"],
            ai_response=feedback["ai_response"],
            sources=feedback["sources"],
        )

        # Read a shortened version of the AI response aloud
        ai_text = feedback["ai_response"]
        if len(ai_text) > 300:
            ai_text = ai_text[:300] + "..."
        pepper.say_and_gesture(ai_text, "explaining")
        time.sleep(2)

        # Advance to next scenario
        has_next = manager.advance_scenario()
        if has_next:
            pepper.say("Let us move to the next scenario.")
            time.sleep(1)

    # 6. End session
    post_score = pre_score + 27.0  # Replaced by actual assessment score in production
    summary = manager.end_session(
        pre_score=pre_score,
        post_score=post_score,
    )

    pepper.show_results(summary)
    pepper.say_and_gesture(
        "Well done! You have completed your SENTRY training session. "
        "Your results are now displayed on screen. "
        "Thank you for training with me today.",
        "farewell",
    )

    print("[Session] Complete. Accuracy: {}%".format(summary.get("accuracy_pct")))
    return summary


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTRY Pepper Client")
    parser.add_argument("--ip", default="127.0.0.1", help="Pepper IP address")
    parser.add_argument("--port", type=int, default=9559, help="NAOqi port")
    parser.add_argument(
        "--middleware", default="http://localhost:8000", help="Middleware server URL"
    )
    parser.add_argument("--participant", default="EMP_001", help="Participant ID")
    parser.add_argument(
        "--condition", default="grounded", choices=["grounded", "baseline"]
    )
    parser.add_argument("--organisation", default="SENTRY_STUDY")
    parser.add_argument("--pre-score", type=float, default=45.0)
    parser.add_argument(
        "--simulation", action="store_true", help="Run without a real Pepper robot"
    )

    args = parser.parse_args()

    run_sentry_session(
        pepper_ip=args.ip,
        pepper_port=args.port,
        middleware_url=args.middleware,
        participant_id=args.participant,
        condition=args.condition,
        organisation_id=args.organisation,
        pre_score=args.pre_score,
        simulation=args.simulation,
    )
