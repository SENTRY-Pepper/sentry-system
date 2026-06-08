"""
SENTRY â€” Phishing Detection Scenario
======================================
Scenario 1: Employee receives a suspicious email and must decide
whether to click, delete, report, or verify via another channel.

Based on: OWASP A07 2025 (Authentication Failures) and
Computer Misuse and Cybercrimes Act 2018, Section 27.
"""

from pepper_interface.scenarios.base_scenario import BaseScenario


class PhishingScenario(BaseScenario):
    @property
    def scenario_id(self):
        return "phishing-01"

    @property
    def scenario_type(self):
        return "phishing"

    @property
    def title(self):
        return "Phishing email detection"

    @property
    def prompt(self):
        return (
            "You receive this email at 8:43 AM from IT Support at "
            "support@heriitage-bank.com â€” notice the spelling. "
            "The subject reads: Urgent â€” verify your account immediately. "
            "The email says your account will be suspended in 24 hours "
            "unless you click a link to verify your credentials. "
            "What do you do?"
        )

    @property
    def choices(self):
        return [
            {
                "id": "click",
                "text": "Click the link and verify my account",
                "is_correct": False,
            },
            {
                "id": "delete",
                "text": "Delete the email without clicking anything",
                "is_correct": False,
                # Deleting is better than clicking but not the best action
                # â€” reporting is the correct response
            },
            {
                "id": "report",
                "text": "Report it to IT immediately and do not click",
                "is_correct": True,
            },
            {
                "id": "verify_phone",
                "text": "Call IT using the official company directory to verify",
                "is_correct": True,
            },
        ]

    @property
    def correct_keywords(self):
        return [
            "report",
            "it department",
            "it team",
            "security team",
            "do not click",
            "don't click",
            "verify",
            "call",
            "phone",
            "suspicious",
            "fake",
            "phishing",
            "forward",
        ]

    @property
    def risky_keywords(self):
        return [
            "click",
            "open",
            "download",
            "reply",
            "respond",
            "enter password",
            "login",
            "credentials",
        ]

    @property
    def rag_query(self):
        return (
            "What is phishing and what are the legal consequences "
            "of phishing attacks in Kenya? How should an employee "
            "respond to a suspicious email?"
        )

    @property
    def pepper_intro(self):
        return (
            "Let us look at a common attack that targets employees every day. "
            "I will describe a situation â€” listen carefully and tell me "
            "what you would do."
        )

    @property
    def pepper_correct_response(self):
        return (
            "Excellent decision. Reporting suspicious emails to your IT team "
            "is exactly the right action. You have helped protect your "
            "organisation. Let me explain what made this email dangerous."
        )

    @property
    def pepper_risky_response(self):
        return (
            "That choice puts your organisation at risk. Clicking links in "
            "suspicious emails is one of the most common ways attackers "
            "gain access to company systems. Let me explain what you should "
            "look for and what to do instead."
        )
