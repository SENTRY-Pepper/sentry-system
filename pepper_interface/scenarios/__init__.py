"""
SENTRY — Scenario Registry
Exports all five scenarios and a convenience loader.
"""

from pepper_interface.scenarios.phishing_scenario import PhishingScenario
from pepper_interface.scenarios.usb_drop_scenario import USBDropScenario
from pepper_interface.scenarios.password_scenario import PasswordScenario
from pepper_interface.scenarios.social_engineering_scenario import (
    SocialEngineeringScenario,
)
from pepper_interface.scenarios.network_scenario import NetworkScenario

ALL_SCENARIOS = [
    PhishingScenario(),
    USBDropScenario(),
    PasswordScenario(),
    SocialEngineeringScenario(),
    NetworkScenario(),
]

SCENARIO_MAP = {s.scenario_id: s for s in ALL_SCENARIOS}
