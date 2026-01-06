#!/usr/bin/env python3
"""Prompt engineering utilities for blitz chess matches."""

from game_arena.blitz.prompts.time_aware import (
    create_time_aware_prompt_substitutions,
    PreviousResponseData,
    create_response_feedback_text,
    get_move_history_in_format,
)
from game_arena.blitz.prompts.dramatic import (
    create_dramatic_time_pressure_text,
    create_dramatic_instruction_text,
)

__all__ = [
    "create_time_aware_prompt_substitutions",
    "PreviousResponseData",
    "create_response_feedback_text",
    "get_move_history_in_format",
    "create_dramatic_time_pressure_text",
    "create_dramatic_instruction_text",
]

