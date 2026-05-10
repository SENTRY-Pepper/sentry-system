"""
SENTRY — Password Hygiene Scenario
=====================================
Scenario 3: Employee is prompted to create a new account password
and must choose the strongest approach.

Based on: OWASP A07 2025 (Authentication Failures).
"""

from pepper_interface.scenarios.base_scenario import BaseScenario


class PasswordScenario(BaseScenario):
    @property
    def scenario_id(self) -> str:
        return "password-01"

    @property
    def scenario_type(self) -> str:
        return "password"

    @property
    def title(self) -> str:
        return "Password hygiene"

    @property
    def prompt(self) -> str:
        return (
            "Your company is rolling out a new internal system and "
            "you need to create a password for your account. "
            "Your IT policy requires a strong password. "
            "Which of the following approaches do you take?"
        )

    @property
    def choices(self) -> list[dict]:
        return [
            {
                "id": "name_dob",
                "text": "Use my name and date of birth — easy to remember",
                "is_correct": False,
            },
            {
                "id": "reuse",
                "text": "Reuse my email password — I already remember it",
                "is_correct": False,
            },
            {
                "id": "passphrase",
                "text": "Create a long passphrase with symbols and numbers",
                "is_correct": True,
            },
            {
                "id": "password_manager",
                "text": "Use a password manager to generate and store a unique password",
                "is_correct": True,
            },
        ]

    @property
    def correct_keywords(self) -> list[str]:
        return [
            "passphrase",
            "password manager",
            "unique",
            "long",
            "complex",
            "symbols",
            "numbers",
            "random",
            "generate",
            "different",
            "mfa",
            "two factor",
            "multi factor",
        ]

    @property
    def risky_keywords(self) -> list[str]:
        return [
            "name",
            "birthday",
            "date of birth",
            "same password",
            "reuse",
            "simple",
            "easy",
            "short",
            "remember",
        ]

    @property
    def rag_query(self) -> str:
        return (
            "What makes a password strong according to OWASP guidelines? "
            "What are the risks of reusing passwords or using personal "
            "information in passwords?"
        )

    @property
    def pepper_intro(self) -> str:
        return (
            "Passwords are the first line of defence for most systems. "
            "Weak passwords are responsible for a significant number of "
            "data breaches. Let me give you a scenario."
        )

    @property
    def pepper_correct_response(self) -> str:
        return (
            "Excellent. Using a password manager to generate unique, complex "
            "passwords is considered best practice. It means that if one "
            "account is compromised, your other accounts remain safe. "
            "Let me share what the guidelines say."
        )

    @property
    def pepper_risky_response(self) -> str:
        return (
            "That approach leaves your account vulnerable. Weak or reused "
            "passwords are one of the leading causes of account takeovers. "
            "Let me explain what a strong password policy looks like."
        )
