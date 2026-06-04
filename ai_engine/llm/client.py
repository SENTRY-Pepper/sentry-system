"""Backward-compatible import for the moved LLM client."""

from ai_engine.generation.client import (
    BASELINE_SYSTEM_PROMPT,
    GROUNDED_SYSTEM_PROMPT,
    LLMClient,
)

__all__ = ["BASELINE_SYSTEM_PROMPT", "GROUNDED_SYSTEM_PROMPT", "LLMClient"]
