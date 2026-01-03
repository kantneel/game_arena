#!/usr/bin/env python3
"""Core data types for blitz chess matches."""

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class MoveStats:
    """Statistics for a single move."""
    player: str
    move_number: int
    move_notation: str
    thinking_time: float
    time_remaining_after: float
    reasoning_tokens: Optional[int]
    total_tokens: Optional[int]
    network_latency: float
    retry_count: int = 0
    total_retry_time: float = 0.0
    had_parsing_failure: bool = False


@dataclasses.dataclass
class GameStats:
    """Statistics for a complete game."""
    game_number: int
    winner: str
    result_string: str
    model_a_color: str
    total_moves: int
    duration: float
    move_stats: List[MoveStats]
    model_a_final_time: float
    model_b_final_time: float
    model_a_parsing_failures: int = 0
    model_b_parsing_failures: int = 0

