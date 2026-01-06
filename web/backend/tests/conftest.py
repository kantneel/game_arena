#!/usr/bin/env python3
"""Pytest fixtures for backend tests."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd


@pytest.fixture
def temp_results_dir():
    """Create a temporary results directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_match_dir(temp_results_dir):
    """Create a sample match directory with realistic data."""
    match_id = "test_match_20231225_120000"
    match_dir = temp_results_dir / match_id
    match_dir.mkdir()
    
    # Create metadata
    metadata = {
        "match_id": match_id,
        "start_time": "2023-12-25T12:00:00",
        "end_time": "2023-12-25T13:00:00",
        "model_a": "claude-sonnet-4.5",
        "model_b": "gemini-3-flash",
        "time_control": "300+3",
        "rethinking_enabled": True,
        "total_games": 3,
        "model_a_wins": 2,
        "model_b_wins": 1,
        "draws": 0,
        "final_winner": "model_a",
    }
    
    with open(match_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Create games summary
    games_data = [
        {
            "game_number": 1,
            "model_a_color": "white",
            "winner": "model_a",
            "result_string": "1-0",
            "termination_reason": "checkmate",
            "total_moves": 80,
            "game_duration_seconds": 600.0,
        },
        {
            "game_number": 2,
            "model_a_color": "black",
            "winner": "model_b",
            "result_string": "1-0",
            "termination_reason": "time_forfeit",
            "total_moves": 52,
            "game_duration_seconds": 550.0,
        },
        {
            "game_number": 3,
            "model_a_color": "white",
            "winner": "model_a",
            "result_string": "1-0",
            "termination_reason": "checkmate",
            "total_moves": 38,
            "game_duration_seconds": 480.0,
        },
    ]
    
    pd.DataFrame(games_data).to_csv(match_dir / "games_summary.csv", index=False)
    
    # Create moves for game 1 with varying time pressure
    # We need moves across all pressure levels to test adaptation ratios
    game1_moves = []
    time_remaining_a = 300.0
    time_remaining_b = 300.0
    
    for move_num in range(1, 81):  # More moves to ensure time runs down to critical
        # Alternate between models
        if move_num % 2 == 1:
            # Model A (white) - Claude
            player = "claude-sonnet-4.5"
            color = "white"
            # Simulate realistic time usage - more time early, faster under pressure
            # Use aggressive time consumption to hit all pressure levels
            if time_remaining_a > 120:
                time_taken = 15.0  # Comfortable: slow and deliberate
                thinking_tokens = 12000
            elif time_remaining_a > 60:
                time_taken = 10.0  # Medium: moderate speed
                thinking_tokens = 8000
            elif time_remaining_a > 30:
                time_taken = 5.0  # High: faster
                thinking_tokens = 4000
            else:
                time_taken = 2.0  # Critical: very fast
                thinking_tokens = 2000
            
            time_at_start = time_remaining_a
            time_remaining_a -= time_taken
            time_remaining_a += 3  # increment
            time_remaining_a = max(5, time_remaining_a)  # Don't go below 5s
        else:
            # Model B (black) - Gemini doesn't adapt as much
            player = "gemini-3-flash"
            color = "black"
            # Gemini uses constant time regardless of pressure
            time_taken = 10.0
            thinking_tokens = 7000
            time_at_start = time_remaining_b
            time_remaining_b -= time_taken
            time_remaining_b += 3
            time_remaining_b = max(5, time_remaining_b)
        
        game1_moves.append({
            "who_played": player,
            "move_played": f"e{move_num % 8 + 1}",
            "board_state_before_move": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "time_taken_seconds": max(0.5, time_taken),
            "response_with_thoughts": "Thinking about the position...",
            "time_available_at_turn_start": time_at_start,
            "thinking_tokens": thinking_tokens,
            "output_tokens": 100,
            "total_tokens": thinking_tokens + 100,
            "move_number": move_num,
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "network_latency": 0.1,
            "retry_count": 0,
            "time_pressure_level": "LOW" if time_at_start > 120 else "HIGH" if time_at_start > 30 else "CRITICAL",
            "used_dramatic_prompts": False,
            "prompt_template_used": "standard",
            "opponent_time_remaining": time_remaining_b if player == "claude-sonnet-4.5" else time_remaining_a,
            "time_increment": 3,
            "reasoning_efficiency": thinking_tokens / max(0.5, time_taken),
            "previous_response_analysis_included": False,
            "time_pressure_category": "comfortable" if time_at_start > 120 else "medium" if time_at_start > 60 else "high" if time_at_start > 30 else "critical",
            "previous_move_time": None,
            "previous_move_efficiency": None,
            "time_trend": "stable",
        })
    
    pd.DataFrame(game1_moves).to_csv(match_dir / "game_1_moves.csv", index=False)
    
    return match_dir


@pytest.fixture
def sample_move_analysis(sample_match_dir):
    """Add move analysis data to the sample match."""
    analysis_data = []
    
    for move_num in range(1, 81):
        # Simulate centipawn loss - worse under pressure (later in game)
        if move_num > 60:
            cpl = 120 + (move_num - 60) * 15  # Much worse late game (critical pressure)
        elif move_num > 40:
            cpl = 60 + (move_num - 40) * 3  # Worse late game (high pressure)
        else:
            cpl = 20 + move_num  # Gradual increase (comfortable/medium)
        
        analysis_data.append({
            "game_number": 1,
            "move_number": move_num,
            "player": "claude-sonnet-4.5" if move_num % 2 == 1 else "gemini-3-flash",
            "centipawn_loss": cpl,
        })
    
    pd.DataFrame(analysis_data).to_csv(
        sample_match_dir / "complete_move_analysis.csv",
        index=False
    )
    
    return sample_match_dir


@pytest.fixture
def multiple_matches(temp_results_dir):
    """Create multiple matches for aggregate testing."""
    matches = []
    
    for i in range(3):
        match_id = f"test_match_{i}_20231225"
        match_dir = temp_results_dir / match_id
        match_dir.mkdir()
        
        metadata = {
            "match_id": match_id,
            "start_time": f"2023-12-2{5+i}T12:00:00",
            "end_time": f"2023-12-2{5+i}T13:00:00",
            "model_a": "claude-sonnet-4.5",
            "model_b": "gemini-3-flash" if i % 2 == 0 else "gpt-5.2",
            "time_control": "300+3",
            "total_games": 2,
            "model_a_wins": 1 + (i % 2),
            "model_b_wins": 1 - (i % 2),
            "draws": 0,
            "final_winner": "model_a" if i % 2 == 0 else "model_b",
        }
        
        with open(match_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Create minimal moves data
        moves = [
            {
                "who_played": "claude-sonnet-4.5",
                "move_played": "e4",
                "time_taken_seconds": 5.0,
                "time_available_at_turn_start": 295.0,
                "thinking_tokens": 10000,
                "move_number": 1,
                "color": "white",
                "board_state_before_move": "start",
            },
            {
                "who_played": metadata["model_b"],
                "move_played": "e5",
                "time_taken_seconds": 4.0,
                "time_available_at_turn_start": 296.0,
                "thinking_tokens": 8000,
                "move_number": 2,
                "color": "black",
                "board_state_before_move": "after e4",
            },
        ]
        
        pd.DataFrame(moves).to_csv(match_dir / "game_1_moves.csv", index=False)
        matches.append(match_id)
    
    return temp_results_dir, matches

