"""
SENTRY â€” Social Engineering / Voice Phishing Scenario
=======================================================
Scenario 4: Employee receives a phone call from someone claiming
to be from IT support requesting their credentials.

Based on: OWASP A07 2025 and Computer Misuse and Cybercrimes
Act 2018 â€” fraud and impersonation provisions.
"""

from pepper_interface.scenarios.base_scenario import BaseScenario


class SocialEngineeringScenario(BaseScenario):
    @property
    def scenario_id(self):
        return "social-engineering-01"

    @property
    def scenario_type(self):
        return "social_engineering"

    @property
    def title(self):
        return "Voice social engineering"

    @property
    def prompt(self):
        return (
            "You receive a phone call. The caller says they are from your "
            "company's IT helpdesk and that there is a critical security "
            "issue with your account. They say they need your username and "
            "password immediately to fix it before your data is lost. "
            "They sound professional and know your name. What do you do?"
        )

    @property
    def choices(self):
        return [
            {
                "id": "give_credentials",
                "text": "Give them my username and password to fix the issue quickly",
                "is_correct": False,
            },
            {
                "id": "ask_name_comply",
                "text": "Ask for their name and then give them the credentials",
                "is_correct": False,
            },
            {
                "id": "hang_up_report",
                "text": "Hang up and call IT using the official company number to verify",
                "is_correct": True,
            },
            {
                "id": "refuse_report",
                "text": "Refuse to give credentials and report the call to my manager",
                "is_correct": True,
            },
        ]

    @property
    def correct_keywords(self):
        return [
            "hang up",
            "hang-up",
            "refuse",
            "do not give",
            "don't give",
            "call back",
            "verify",
            "official number",
            "report",
            "manager",
            "suspicious",
            "vishing",
            "social engineering",
        ]

    @property
    def risky_keywords(self):
        return [
            "give",
            "provide",
            "share",
            "tell them",
            "say my password",
            "comply",
            "help them",
            "trust",
        ]

    @property
    def rag_query(self):
        return (
            "What is social engineering and vishing? How should employees "
            "respond to phone calls requesting their credentials? "
            "What does Kenyan law say about impersonation and fraud?"
        )

    @property
    def pepper_intro(self):
        return (
            "Attackers do not always use technology to breach organisations. "
            "Sometimes they simply pick up the phone. This is called "
            "social engineering. Listen to this scenario carefully."
        )

    @property
    def pepper_correct_response(self):
        return (
            "Exactly right. Legitimate IT support will never ask for your "
            "password over the phone. Hanging up and calling back on a "
            "verified number is the correct response. You have protected "
            "your organisation from a vishing attack."
        )

    @property
    def pepper_risky_response(self):
        return (
            "That is a dangerous response. No legitimate IT team needs your "
            "password to fix an issue. This is a classic vishing attack â€” "
            "voice phishing. Sharing credentials with an unverified caller "
            "can lead to a full account takeover."
        )
