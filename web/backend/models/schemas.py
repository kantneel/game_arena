#!/usr/bin/env python3
"""Pydantic schemas for API responses."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MatchSummary(BaseModel):
    """Summary of a match for list views."""
    match_id: str
    model_a: str
    model_b: str
    model_a_score: int
    model_b_score: int
    draws: int
    winner: str  # "model_a", "model_b", "draw", "in_progress"
    total_games: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    time_control: str
    status: str  # "completed", "live"
    notes: Optional[str] = None  # Optional notes/tags for experiment tracking


class MatchDetail(BaseModel):
    """Full match details including games."""
    match_id: str
    model_a: str
    model_b: str
    model_a_score: int
    model_b_score: int
    draws: int
    winner: str
    total_games: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    time_control: str
    rethinking_enabled: bool
    games: list["GameSummary"]
    current_game: int = 0  # Current game being played (for live matches)
    notes: Optional[str] = None  # Optional notes/tags for experiment tracking


class GameSummary(BaseModel):
    """Summary of a single game."""
    game_number: int
    white_model: str
    black_model: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    winner: str  # "model_a", "model_b", "draw"
    termination: str  # "checkmate", "time_forfeit", "draw", etc.
    total_moves: int
    duration_seconds: float


class GameDetail(BaseModel):
    """Full game details including moves."""
    game_number: int
    match_id: str
    white_model: str
    black_model: str
    result: str
    winner: str
    termination: str
    total_moves: int
    duration_seconds: float
    moves: list["MoveRecord"]


class MoveRecord(BaseModel):
    """Single move data."""
    move_number: int
    player: str
    color: str
    move: str
    fen_before: str
    time_taken: float
    time_remaining: float
    thinking_tokens: Optional[int] = None
    # Stockfish analysis (populated if move analysis was run)
    centipawn_loss: Optional[float] = None
    is_best_move: Optional[bool] = None
    is_blunder: Optional[bool] = None  # True if CP loss >= 100
    best_move: Optional[str] = None  # The engine's preferred move
    win_probability_loss: Optional[float] = None  # WP loss from 0-1
    # Position complexity metrics
    num_legal_moves: Optional[int] = None  # Number of legal moves available
    eval_sharpness: Optional[int] = None  # CP diff between best and 2nd best move
    position_eval_abs: Optional[int] = None  # Absolute evaluation in CP


class ModelStats(BaseModel):
    """Model statistics for leaderboard."""
    model_id: str
    display_name: str
    elo: int
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    elo_change: int  # Recent change


class LeaderboardResponse(BaseModel):
    """Full leaderboard response."""
    models: list[ModelStats]
    last_updated: datetime


class LiveMatchState(BaseModel):
    """Current state of a live match."""
    match_id: str
    model_a: str
    model_b: str
    current_game: int
    model_a_score: int
    model_b_score: int
    current_fen: str
    last_move: Optional[str] = None
    model_a_time: float
    model_b_time: float
    to_move: str  # "model_a" or "model_b"
    move_count: int


# Allow forward references
MatchDetail.model_rebuild()
GameDetail.model_rebuild()

