#!/usr/bin/env python3
"""Data types for Stockfish move analysis."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class MoveAnalysis:
    """Analysis result for a single chess move."""
    # Game identification
    match_id: str
    game_number: int
    move_number: int
    color: str  # "white" or "black"
    player: str  # "Model A" or "Model B"
    
    # Move information
    move_played: str
    board_fen_before: str
    
    # Engine analysis
    best_move_uci: Optional[str]
    best_move_san: Optional[str]
    best_eval_cp_from_player_pov: int
    played_eval_cp_from_player_pov: int
    centipawn_loss: int
    played_move_rank_among_top: Optional[int]
    
    # Win probability analysis
    best_win_probability: float  # Win probability after best move (0.0-1.0)
    played_win_probability: float  # Win probability after played move (0.0-1.0)
    win_probability_loss: float  # Difference in win probability (0.0-1.0)
    
    # Position complexity metrics
    num_legal_moves: int  # Number of legal moves available (more = more complex)
    eval_sharpness: int  # CP difference between best and 2nd best move (higher = more critical)
    position_eval_abs: int  # Absolute evaluation in CP (0 = equal, higher = more decisive)
    
    # Human-readable strings
    best_eval_str: str
    played_eval_str: str
    cp_loss_str: str
    best_win_prob_str: str  # e.g., "65.2%"
    played_win_prob_str: str  # e.g., "58.1%"
    win_prob_loss_str: str  # e.g., "-7.1%"
    
    # Engine parameters used
    engine_depth: int
    multipv: int

