#!/usr/bin/env python3
"""Data types for blitz chess match data collection."""

import dataclasses
import datetime
from typing import Optional


@dataclasses.dataclass
class MatchMetadata:
    """Metadata for an entire match."""
    match_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    last_updated: Optional[datetime.datetime] = None  # Heartbeat - updated on every move
    model_a: str = ""
    model_b: str = ""
    time_control: str = ""  # e.g., "300+3"
    rethinking_enabled: bool = False
    max_parsing_failures: int = 3
    max_rethinks: int = 2
    reasoning_budget: int = 8000
    parser_choice: str = ""
    total_games: int = 0
    model_a_wins: int = 0
    model_b_wins: int = 0
    draws: int = 0
    final_winner: str = ""
    match_duration_seconds: float = 0.0
    current_game: int = 0  # Which game is currently being played
    first_to: int = 0  # 0 means not using first_to mode
    total_games: int = 0  # 0 means not using fixed games mode  # Number of wins needed to win the match
    
    # New fields for time pressure features
    dramatic_prompts_enabled: bool = False
    stateful_agents_enabled: bool = False
    dramatic_threshold_seconds: float = 60.0  # Time threshold for dramatic prompts
    time_pressure_strategy: str = "none"  # "none", "dramatic", "stateful", "combined"
    
    # Notes/tags for experiment tracking
    notes: Optional[str] = None


@dataclasses.dataclass
class GameRecord:
    """Structured record of a single game for data analysis."""
    # Match identification
    match_id: str
    game_number: int
    timestamp: datetime.datetime
    
    # Game setup
    model_a_color: str  # "white" or "black"
    model_b_color: str  # "white" or "black"
    
    # Game outcome
    winner: str  # "model_a", "model_b", "draw", "error"
    result_string: str  # PGN format like "1-0", "0-1", "1/2-1/2"
    termination_reason: str  # "checkmate", "time_forfeit", "parsing_failure", "draw", "move_limit"
    
    # Game duration and moves
    total_moves: int
    game_duration_seconds: float
    
    # Time management
    model_a_initial_time: float
    model_b_initial_time: float
    model_a_final_time: float
    model_b_final_time: float
    model_a_time_used: float
    model_b_time_used: float
    time_increment: float
    
    # Performance metrics
    model_a_parsing_failures: int
    model_b_parsing_failures: int
    model_a_avg_move_time: float
    model_b_avg_move_time: float
    model_a_total_tokens: int
    model_b_total_tokens: int
    model_a_reasoning_tokens: int
    model_b_reasoning_tokens: int
    model_a_network_retries: int
    model_b_network_retries: int
    
    # Network latency
    model_a_avg_latency: float
    model_b_avg_latency: float


@dataclasses.dataclass
class GameMoveRecord:
    """Detailed record of a single move within a specific game for CSV export."""
    # Required columns as specified
    who_played: str  # The actual model name (e.g., "claude-sonnet-4")
    move_played: str  # Chess move notation
    board_state_before_move: str  # Board state prior to the move
    time_taken_seconds: float  # Time taken to make the move
    response_with_thoughts: str  # Full response text including thoughts
    time_available_at_turn_start: float  # Time remaining when turn began
    thinking_tokens: Optional[int]  # Number of thinking tokens if available
    output_tokens: Optional[int]  # Number of output/generation tokens
    total_tokens: Optional[int]  # Total tokens (prompt + output)
    
    # Additional context (optional, can be included for analysis)
    move_number: int
    color: str  # "white" or "black"
    timestamp: str  # ISO format timestamp
    network_latency: float
    retry_count: int
    
    # New fields for time pressure and stateful analysis
    time_pressure_level: str  # "EXTREME", "HIGH", "MEDIUM", "LOW"
    used_dramatic_prompts: bool  # Whether dramatic time pressure prompts were used
    prompt_template_used: str  # The actual prompt template used
    opponent_time_remaining: float  # Opponent's time when this move was made
    time_increment: int  # Time increment per move
    reasoning_efficiency: Optional[float]  # Reasoning tokens per second
    previous_response_analysis_included: bool  # Whether stateful analysis was included
    time_pressure_category: str  # "under_30s", "under_60s", "under_120s", "comfortable"
    
    # Stateful feedback metrics
    previous_move_time: Optional[float]  # Time taken for previous move (for stateful analysis)
    previous_move_efficiency: Optional[float]  # Previous reasoning efficiency
    time_trend: Optional[str]  # "speeding_up", "slowing_down", "stable", "first_move"

