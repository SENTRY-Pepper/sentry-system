"""
SENTRY â€” USB Drop Attack Scenario
====================================
Scenario 2: Employee finds an unknown USB drive in the office
and must decide what to do with it.

Based on: OWASP A08 2025 (Software and Data Integrity Failures)
and general physical security best practices.
"""

from pepper_interface.scenarios.base_scenario import BaseScenario


class USBDropScenario(BaseScenario):
    @property
    def scenario_id(self):
        return "usb-drop-01"

    @property
    def scenario_type(self):
        return "usb_drop"

    @property
    def title(self):
        return "USB drop attack"

    @property
    def prompt(self):
        return (
            "You arrive at work on Monday morning and find a USB drive "
            "on your desk. There is no label on it and you do not know "
            "who left it there. It could contain important files â€” "
            "or it could be something else entirely. What do you do?"
        )

    @property
    def choices(self):
        return [
            {
                "id": "plug_in",
                "text": "Plug it into my computer to see what is on it",
                "is_correct": False,
            },
            {
                "id": "plug_other",
                "text": "Plug it into a colleague's computer to be safe",
                "is_correct": False,
            },
            {
                "id": "hand_it",
                "text": "Hand it to the IT department without plugging it in",
                "is_correct": True,
            },
            {
                "id": "discard",
                "text": "Discard it in the bin without plugging it in",
                "is_correct": False,
                # Discarding is not ideal â€” IT should examine it
            },
        ]

    @property
    def correct_keywords(self):
        return [
            "it department",
            "it team",
            "security",
            "hand over",
            "give to",
            "report",
            "do not plug",
            "don't plug",
            "leave it",
            "not connect",
        ]

    @property
    def risky_keywords(self):
        return [
            "plug in",
            "connect",
            "insert",
            "check",
            "look at",
            "open",
            "read",
            "scan",
            "use",
        ]

    @property
    def rag_query(self):
        return (
            "What should an employee do if they find an unknown USB drive "
            "at work? What are the cybersecurity risks of plugging in "
            "an unknown USB device?"
        )

    @property
    def pepper_intro(self):
        return (
            "This next scenario is about a physical security threat that "
            "attackers use to bypass digital defences entirely. "
            "Listen carefully."
        )

    @property
    def pepper_correct_response(self):
        return (
            "Well done. Handing the USB drive to IT without plugging it in "
            "is the safest action. USB drives can carry malware that "
            "activates the moment they are connected. Let me tell you more "
            "about how this attack works."
        )

    @property
    def pepper_risky_response(self):
        return (
            "That is a risky decision. Plugging in an unknown USB drive â€” "
            "even on a different computer â€” can trigger malware that spreads "
            "across your organisation's network. Let me explain what you "
            "should do instead."
        )
