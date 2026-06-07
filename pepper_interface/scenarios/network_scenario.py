"""
SENTRY — Network Hygiene Scenario
====================================
Scenario 5: Employee is working remotely and considers connecting
to public WiFi to access company systems.

Based on: OWASP A02 2025 (Security Misconfiguration) and
general network security best practices.
"""

from pepper_interface.scenarios.base_scenario import BaseScenario


class NetworkScenario(BaseScenario):
    @property
    def scenario_id(self) -> str:
        return "network-01"

    @property
    def scenario_type(self) -> str:
        return "network"

    @property
    def title(self) -> str:
        return "Network hygiene"

    @property
    def prompt(self) -> str:
        return (
            "You are working from a coffee shop and need to access your "
            "company's internal system to submit an urgent report. "
            "The coffee shop has free public WiFi. Your mobile data "
            "is running low. What do you do?"
        )

    @property
    def choices(self) -> list[dict]:
        return [
            {
                "id": "connect_direct",
                "text": "Connect to the public WiFi and access the system directly",
                "is_correct": False,
            },
            {
                "id": "connect_vpn",
                "text": "Connect to public WiFi but use the company VPN first",
                "is_correct": True,
            },
            {
                "id": "mobile_hotspot",
                "text": "Use my phone as a personal mobile hotspot instead",
                "is_correct": True,
            },
            {
                "id": "wait",
                "text": "Wait until I am on a trusted network before accessing company systems",
                "is_correct": True,
            },
        ]

    @property
    def correct_keywords(self) -> list[str]:
        return [
            "vpn",
            "virtual private network",
            "hotspot",
            "mobile data",
            "wait",
            "trusted network",
            "secure connection",
            "encrypt",
            "avoid public",
            "do not connect",
        ]

    @property
    def risky_keywords(self) -> list[str]:
        return [
            "public wifi",
            "connect directly",
            "free wifi",
            "no vpn",
            "without vpn",
        ]

    @property
    def rag_query(self) -> str:
        return (
            "Is it safe to use public WiFi to access company systems? "
            "What are the risks of unsecured networks and how can employees "
            "protect themselves when working remotely?"
        )

    @property
    def pepper_intro(self) -> str:
        return (
            "Remote work has become common, but it introduces new security "
            "risks that many employees are unaware of. Let me describe "
            "a situation you might face."
        )

    @property
    def pepper_correct_response(self) -> str:
        return (
            "Good thinking. Using a VPN or a personal hotspot before "
            "accessing company systems protects your data from interception "
            "on public networks. Let me explain why public WiFi is risky "
            "even when it requires a password."
        )

    @property
    def pepper_risky_response(self) -> str:
        return (
            "Connecting to company systems over unencrypted public WiFi "
            "exposes your data to anyone on the same network. Attackers "
            "can intercept credentials and sensitive data using a technique "
            "called a man-in-the-middle attack. Let me explain what to do instead."
        )
