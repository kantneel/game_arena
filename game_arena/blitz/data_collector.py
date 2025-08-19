#!/usr/bin/env python3
"""Data collection and analysis utilities for blitz chess matches."""

import datetime
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import dataclasses
from dataclasses import asdict
import csv

from game_arena.blitz import utils


def abbreviate_model_name(model_name: str) -> str:
    """Create a short abbreviation for model names to keep folder names reasonable."""
    # Model abbreviations based on actual models in use (from verification.py name_mappings)
    abbreviations = {
        "claude-sonnet-4": "cs4",
        "claude-opus-4": "co4", 
        "gemini-2.5-flash": "g25f",
        "gemini-2.5-pro": "g25p",
        "gpt-4.1": "gpt41",
        "o3": "o3",
        "o4-mini": "o4m",
        "grok-4": "grok4",
        "deepseek-r1": "dsr1",
        "kimi-k2": "kimik2",
        "qwen3": "qw3",
        # Legacy abbreviations for backward compatibility
        "gpt-4": "gpt4",
        "gpt-4-turbo": "gpt4t",
        "claude-3-sonnet": "c3s",
        "claude-3-opus": "c3o",
        "claude-3.5-sonnet": "c35s",
        "gemini-pro": "gemi"
    }
    
    # Try exact match first
    if model_name.lower() in abbreviations:
        return abbreviations[model_name.lower()]
    
    # Try partial matches for more complex model names
    lower_name = model_name.lower()
    for full_name, abbrev in abbreviations.items():
        if full_name in lower_name:
            return abbrev
    
    # Fallback: take first letters of words, max 6 chars
    words = model_name.replace("-", " ").replace("_", " ").split()
    if len(words) > 1:
        abbrev = "".join(word[0] for word in words if word)[:6]
        return abbrev.lower()
    else:
        # Single word, take first 6 characters
        return model_name[:6].lower()


@dataclasses.dataclass
class MatchMetadata:
    """Metadata for an entire match."""
    match_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
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


class BlitzDataCollector:
    """Collects and manages data for blitz chess matches."""
    
    def __init__(self, data_dir: str = "_results"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.match_metadata: Optional[MatchMetadata] = None
        self.game_records: List[GameRecord] = []
        self.per_game_moves: Dict[int, List[GameMoveRecord]] = {}  # game_number -> moves
        
        # Current match state
        self.current_match_id: Optional[str] = None
        self.current_match_dir: Optional[Path] = None
        
    def start_match(self, 
                   model_a: str,
                   model_b: str,
                   time_control_seconds: int,
                   increment_seconds: int,
                   rethinking_enabled: bool,
                   max_parsing_failures: int,
                   max_rethinks: int,
                   reasoning_budget: int,
                   parser_choice: str) -> str:
        """Start a new match and return the match ID."""
        timestamp = datetime.datetime.now()
        
        # Create abbreviated model names for folder naming
        model_a_abbrev = abbreviate_model_name(model_a)
        model_b_abbrev = abbreviate_model_name(model_b)
        
        # Create match ID with abbreviated model names and timestamp
        match_id = f"{model_a_abbrev}_vs_{model_b_abbrev}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create match directory
        self.current_match_id = match_id
        self.current_match_dir = self.data_dir / match_id
        self.current_match_dir.mkdir(exist_ok=True)
        
        self.match_metadata = MatchMetadata(
            match_id=match_id,
            start_time=timestamp,
            model_a=model_a,
            model_b=model_b,
            time_control=f"{time_control_seconds}+{increment_seconds}",
            rethinking_enabled=rethinking_enabled,
            max_parsing_failures=max_parsing_failures,
            max_rethinks=max_rethinks,
            reasoning_budget=reasoning_budget,
            parser_choice=parser_choice
        )
        
        return match_id
    
    def record_move(self, game_number: int, who_played: str, move_played: str, 
                   board_state_before: str, time_taken: float, response_text: str,
                   time_at_turn_start: float, thinking_tokens: Optional[int],
                   output_tokens: Optional[int], total_tokens: Optional[int],
                   move_number: int, color: str, network_latency: float, 
                   retry_count: int) -> None:
        """Record a single move during gameplay."""
        if not self.current_match_id:
            raise ValueError("No active match. Call start_match() first.")
        
        move_record = GameMoveRecord(
            who_played=who_played,
            move_played=move_played,
            board_state_before_move=board_state_before,
            time_taken_seconds=time_taken,
            response_with_thoughts=response_text,
            time_available_at_turn_start=time_at_turn_start,
            thinking_tokens=thinking_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            move_number=move_number,
            color=color,
            timestamp=datetime.datetime.now().isoformat(),
            network_latency=network_latency,
            retry_count=retry_count
        )
        
        # Initialize game moves list if needed
        if game_number not in self.per_game_moves:
            self.per_game_moves[game_number] = []
        
        self.per_game_moves[game_number].append(move_record)
    
    def record_game(self, game_stats: utils.GameStats, initial_time: float, increment: float) -> None:
        """Record a completed game."""
        if not self.current_match_id:
            raise ValueError("No active match. Call start_match() first.")
        
        # Calculate derived metrics
        model_a_time_used = initial_time - game_stats.model_a_final_time
        model_b_time_used = initial_time - game_stats.model_b_final_time
        
        # Calculate average move times
        model_a_moves = [m for m in game_stats.move_stats if m.player == "Model A"]
        model_b_moves = [m for m in game_stats.move_stats if m.player == "Model B"]
        
        model_a_avg_move_time = sum(m.thinking_time for m in model_a_moves) / len(model_a_moves) if model_a_moves else 0
        model_b_avg_move_time = sum(m.thinking_time for m in model_b_moves) / len(model_b_moves) if model_b_moves else 0
        
        # Calculate token totals
        model_a_total_tokens = sum(m.total_tokens for m in model_a_moves)
        model_b_total_tokens = sum(m.total_tokens for m in model_b_moves)
        model_a_reasoning_tokens = sum(m.reasoning_tokens or 0 for m in model_a_moves)
        model_b_reasoning_tokens = sum(m.reasoning_tokens or 0 for m in model_b_moves)
        
        # Calculate network metrics
        model_a_network_retries = sum(m.retry_count for m in model_a_moves)
        model_b_network_retries = sum(m.retry_count for m in model_b_moves)
        model_a_avg_latency = sum(m.network_latency for m in model_a_moves) / len(model_a_moves) if model_a_moves else 0
        model_b_avg_latency = sum(m.network_latency for m in model_b_moves) / len(model_b_moves) if model_b_moves else 0
        
        # Determine termination reason
        termination_reason = self._determine_termination_reason(game_stats)
        
        game_record = GameRecord(
            match_id=self.current_match_id,
            game_number=game_stats.game_number,
            timestamp=datetime.datetime.now(),
            model_a_color=game_stats.model_a_color,
            model_b_color="black" if game_stats.model_a_color == "white" else "white",
            winner=game_stats.winner,
            result_string=game_stats.result_string,
            termination_reason=termination_reason,
            total_moves=game_stats.total_moves,
            game_duration_seconds=game_stats.duration,
            model_a_initial_time=initial_time,
            model_b_initial_time=initial_time,
            model_a_final_time=game_stats.model_a_final_time,
            model_b_final_time=game_stats.model_b_final_time,
            model_a_time_used=model_a_time_used,
            model_b_time_used=model_b_time_used,
            time_increment=increment,
            model_a_parsing_failures=game_stats.model_a_parsing_failures,
            model_b_parsing_failures=game_stats.model_b_parsing_failures,
            model_a_avg_move_time=model_a_avg_move_time,
            model_b_avg_move_time=model_b_avg_move_time,
            model_a_total_tokens=model_a_total_tokens,
            model_b_total_tokens=model_b_total_tokens,
            model_a_reasoning_tokens=model_a_reasoning_tokens,
            model_b_reasoning_tokens=model_b_reasoning_tokens,
            model_a_network_retries=model_a_network_retries,
            model_b_network_retries=model_b_network_retries,
            model_a_avg_latency=model_a_avg_latency,
            model_b_avg_latency=model_b_avg_latency
        )
        
        self.game_records.append(game_record)
        
        # Save per-game CSV with moves
        self._save_game_moves_csv(game_stats.game_number)
        
        # Save summary data after each game
        self.save_game_data()
    
    def _save_game_moves_csv(self, game_number: int) -> None:
        """Save moves for a specific game to its own CSV file."""
        if game_number not in self.per_game_moves or not self.current_match_dir:
            return
        
        moves = self.per_game_moves[game_number]
        if not moves:
            return
        
        # Create CSV file for this game
        csv_path = self.current_match_dir / f"game_{game_number}_moves.csv"
        
        # Convert moves to dictionaries for CSV writing
        moves_data = [asdict(move) for move in moves]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if moves_data:
                fieldnames = moves_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(moves_data)
    
    def _determine_termination_reason(self, game_stats: utils.GameStats) -> str:
        """Determine how the game ended."""
        if game_stats.winner == "error":
            return "error"
        elif "time" in game_stats.result_string.lower():
            return "time_forfeit"
        elif game_stats.model_a_parsing_failures >= 3 or game_stats.model_b_parsing_failures >= 3:
            return "parsing_failure"
        elif game_stats.result_string in ["1/2-1/2", "draw"]:
            return "draw"
        elif game_stats.total_moves >= 200:  # Assuming move limit is 200
            return "move_limit"
        else:
            return "checkmate"
    
    def end_match(self, final_scores: Dict[str, int]) -> None:
        """End the current match and update metadata."""
        if self.match_metadata:
            self.match_metadata.end_time = datetime.datetime.now()
            self.match_metadata.total_games = len(self.game_records)
            self.match_metadata.model_a_wins = final_scores.get("model_a", 0)
            self.match_metadata.model_b_wins = final_scores.get("model_b", 0)
            self.match_metadata.draws = final_scores.get("draws", 0)
            
            if self.match_metadata.model_a_wins > self.match_metadata.model_b_wins:
                self.match_metadata.final_winner = "model_a"
            elif self.match_metadata.model_b_wins > self.match_metadata.model_a_wins:
                self.match_metadata.final_winner = "model_b"
            else:
                self.match_metadata.final_winner = "draw"
            
            duration = self.match_metadata.end_time - self.match_metadata.start_time
            self.match_metadata.match_duration_seconds = duration.total_seconds()
        
        # Save final data
        self.save_all_data()
    
    def save_game_data(self) -> None:
        """Save data after each game for incremental analysis."""
        if not self.current_match_id or not self.current_match_dir:
            return
        
        # Save games summary as CSV (for easy analysis)
        if self.game_records:
            games_df = pd.DataFrame([asdict(record) for record in self.game_records])
            games_df.to_csv(self.current_match_dir / "games_summary.csv", index=False)
    
    def save_all_data(self) -> None:
        """Save all data at the end of the match."""
        if not self.current_match_id or not self.current_match_dir:
            return
        
        # Save metadata as JSON
        if self.match_metadata:
            metadata_dict = asdict(self.match_metadata)
            # Convert datetime objects to strings
            for key, value in metadata_dict.items():
                if isinstance(value, datetime.datetime):
                    metadata_dict[key] = value.isoformat()
            
            with open(self.current_match_dir / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        
        # Save final CSV files
        self.save_game_data()
        
        # Save summary statistics
        self._save_summary_stats(self.current_match_dir)
        
        print(f"📊 Match data saved to: {self.current_match_dir}")
    
    def _save_summary_stats(self, match_dir: Path) -> None:
        """Generate and save summary statistics."""
        if not self.game_records:
            return
        
        games_df = pd.DataFrame([asdict(record) for record in self.game_records])
        
        summary = {
            "match_overview": {
                "total_games": len(self.game_records),
                "model_a_wins": len(games_df[games_df.winner == "model_a"]),
                "model_b_wins": len(games_df[games_df.winner == "model_b"]),
                "draws": len(games_df[games_df.winner == "draw"]),
                "average_game_duration": games_df.game_duration_seconds.mean(),
                "average_moves_per_game": games_df.total_moves.mean(),
            },
            "time_management": {
                "model_a_avg_time_used": games_df.model_a_time_used.mean(),
                "model_b_avg_time_used": games_df.model_b_time_used.mean(),
                "model_a_avg_final_time": games_df.model_a_final_time.mean(),
                "model_b_avg_final_time": games_df.model_b_final_time.mean(),
                "model_a_avg_move_time": games_df.model_a_avg_move_time.mean(),
                "model_b_avg_move_time": games_df.model_b_avg_move_time.mean(),
            },
            "performance_metrics": {
                "model_a_avg_parsing_failures": games_df.model_a_parsing_failures.mean(),
                "model_b_avg_parsing_failures": games_df.model_b_parsing_failures.mean(),
                "model_a_avg_tokens_per_game": games_df.model_a_total_tokens.mean(),
                "model_b_avg_tokens_per_game": games_df.model_b_total_tokens.mean(),
                "model_a_avg_reasoning_tokens": games_df.model_a_reasoning_tokens.mean(),
                "model_b_avg_reasoning_tokens": games_df.model_b_reasoning_tokens.mean(),
            }
        }
        
        with open(match_dir / "summary_stats.json", 'w') as f:
            json.dump(summary, f, indent=2)


def create_analysis_notebook(match_id: str, data_dir: str = "_results") -> None:
    """Create a Jupyter notebook template for analyzing individual game data."""
    notebook_content = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# Blitz Chess Game Analysis: {match_id}\\n",
    "\\n",
    "This notebook provides turn-level analysis of individual games from the blitz chess match."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import json\\n",
    "import numpy as np\\n",
    "from pathlib import Path\\n",
    "import glob\\n",
    "\\n",
    "# Set up plotting style\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')\\n",
    "\\n",
    "# Load data\\n",
    "data_dir = Path('')\\n",
    "\\n",
    "# Load metadata\\n",
    "with open(data_dir / 'metadata.json') as f:\\n",
    "    metadata = json.load(f)\\n",
    "\\n",
    "# Load games summary data\\n",
    "games_df = pd.read_csv(data_dir / 'games_summary.csv')\\n",
    "\\n",
    "# Load all available game move files\\n",
    "move_files = list(data_dir.glob('game_*_moves.csv'))\\n",
    "print(f\\"Found {{len(move_files)}} game move files\\")\\n",
    "\\n",
    "# Check for move quality analysis data\\n",
    "move_analysis_file = data_dir / 'complete_move_analysis.csv'\\n",
    "has_move_analysis = move_analysis_file.exists()\\n",
    "print(f\\"Move quality analysis available: {{has_move_analysis}}\\")\\n",
    "\\n",
    "# Load moves from first game for detailed analysis\\n",
    "if move_files:\\n",
    "    game_moves_df = pd.read_csv(move_files[0])\\n",
    "    game_num = move_files[0].stem.split('_')[1]\\n",
    "    print(f\\"Analyzing Game {{game_num}} in detail\\")\\n",
    "else:\\n",
    "    game_moves_df = None\\n",
    "    print(\\"No individual game move data available\\")\\n",
    "\\n",
    "# Load move quality analysis if available\\n",
    "if has_move_analysis:\\n",
    "    move_analysis_df = pd.read_csv(move_analysis_file)\\n",
    "    print(f\\"Loaded move analysis for {{len(move_analysis_df)}} moves\\")\\n",
    "else:\\n",
    "    move_analysis_df = None\\n",
    "\\n",
    "print(f\\"Match: {{metadata['match_id']}}\\")\\n",
    "print(f\\"Models: {{metadata['model_a']}} vs {{metadata['model_b']}}\\")\\n",
    "print(f\\"Total Games: {{metadata['total_games']}}\\")\\n",
    "if game_moves_df is not None:\\n",
    "    print(f\\"Moves in Game {{game_num}}: {{len(game_moves_df)}}\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "if game_moves_df is not None:\\n",
    "    # Turn-level duration and move analysis\\n",
    "    plt.figure(figsize=(15, 10))\\n",
    "    \\n",
    "    # Separate data by player\\n",
    "    model_a_moves = game_moves_df[game_moves_df['who_played'] == 'Model A']\\n",
    "    model_b_moves = game_moves_df[game_moves_df['who_played'] == 'Model B']\\n",
    "    \\n",
    "    # Move time over the course of the game\\n",
    "    plt.subplot(2, 3, 1)\\n",
    "    plt.plot(model_a_moves['move_number'], model_a_moves['time_taken_seconds'], \\n",
    "             'o-', label=f\\"{{metadata['model_a']}} (Model A)\\", color='blue', alpha=0.7)\\n",
    "    plt.plot(model_b_moves['move_number'], model_b_moves['time_taken_seconds'], \\n",
    "             's-', label=f\\"{{metadata['model_b']}} (Model B)\\", color='red', alpha=0.7)\\n",
    "    plt.xlabel('Move Number')\\n",
    "    plt.ylabel('Time Taken (seconds)')\\n",
    "    plt.title('Move Time Throughout Game')\\n",
    "    plt.legend()\\n",
    "    \\n",
    "    # Time remaining over the course of the game\\n",
    "    plt.subplot(2, 3, 2)\\n",
    "    plt.plot(model_a_moves['move_number'], model_a_moves['time_available_at_turn_start'], \\n",
    "             'o-', label=f\\"{{metadata['model_a']}} (Model A)\\", color='blue', alpha=0.7)\\n",
    "    plt.plot(model_b_moves['move_number'], model_b_moves['time_available_at_turn_start'], \\n",
    "             's-', label=f\\"{{metadata['model_b']}} (Model B)\\", color='red', alpha=0.7)\\n",
    "    plt.xlabel('Move Number')\\n",
    "    plt.ylabel('Time Remaining (seconds)')\\n",
    "    plt.title('Time Bank Throughout Game')\\n",
    "    plt.legend()\\n",
    "    \\n",
    "    # 2D scatter: Time remaining vs time taken for both models\\n",
    "    plt.subplot(2, 3, 3)\\n",
    "    plt.scatter(model_a_moves['time_available_at_turn_start'], \\n",
    "                model_a_moves['time_taken_seconds'], \\n",
    "                alpha=0.7, color='blue', s=50, label=f\\"{{metadata['model_a']}} (Model A)\\")\\n",
    "    plt.scatter(model_b_moves['time_available_at_turn_start'], \\n",
    "                model_b_moves['time_taken_seconds'], \\n",
    "                alpha=0.7, color='red', s=50, label=f\\"{{metadata['model_b']}} (Model B)\\")\\n",
    "    plt.xlabel('Time Remaining at Turn Start (seconds)')\\n",
    "    plt.ylabel('Time Taken for Move (seconds)')\\n",
    "    plt.title('Time Remaining vs Time Taken')\\n",
    "    plt.legend()\\n",
    "    \\n",
    "    # Combined comparison of thinking patterns\\n",
    "    plt.subplot(2, 3, 4)\\n",
    "    plt.boxplot([model_a_moves['time_taken_seconds'], model_b_moves['time_taken_seconds']], \\n",
    "               labels=[f\\"{{metadata['model_a']}}\\\\n(Model A)\\", f\\"{{metadata['model_b']}}\\\\n(Model B)\\"])\\n",
    "    plt.ylabel('Time Taken per Move (seconds)')\\n",
    "    plt.title('Move Time Distribution Comparison')\\n",
    "    \\n",
    "    plt.tight_layout()\\n",
    "    plt.show()\\n",
    "else:\\n",
    "    print(\\"No move-level data available for detailed analysis\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "if game_moves_df is not None:\\n",
    "    # Token analysis (output tokens for all models, thinking tokens when available)\\n",
    "    plt.figure(figsize=(12, 8))\\n",
    "    \\n",
    "    # Check for available token data\\n",
    "    has_output_tokens = 'output_tokens' in game_moves_df.columns and game_moves_df['output_tokens'].notna().any()\\n",
    "    has_thinking_tokens = 'thinking_tokens' in game_moves_df.columns and game_moves_df['thinking_tokens'].notna().any()\\n",
    "    \\n",
    "    # Use output tokens if available, fallback to thinking tokens\\n",
    "    if has_output_tokens:\\n",
    "        token_col = 'output_tokens'\\n",
    "        token_label = 'Output Tokens'\\n",
    "        model_a_tokens = model_a_moves[model_a_moves['output_tokens'].notna()]\\n",
    "        model_b_tokens = model_b_moves[model_b_moves['output_tokens'].notna()]\\n",
    "    elif has_thinking_tokens:\\n",
    "        token_col = 'thinking_tokens'\\n",
    "        token_label = 'Thinking Tokens'\\n",
    "        model_a_tokens = model_a_moves[model_a_moves['thinking_tokens'].notna()]\\n",
    "        model_b_tokens = model_b_moves[model_b_moves['thinking_tokens'].notna()]\\n",
    "    else:\\n",
    "        token_col = None\\n",
    "        token_label = None\\n",
    "        model_a_tokens = pd.DataFrame()\\n",
    "        model_b_tokens = pd.DataFrame()\\n",
    "    \\n",
    "    if token_col and not model_a_tokens.empty and not model_b_tokens.empty:\\n",
    "        plt.subplot(2, 2, 1)\\n",
    "        plt.scatter(model_a_tokens[token_col], model_a_tokens['time_taken_seconds'], \\n",
    "                   alpha=0.7, color='blue', label=f\\"{{metadata['model_a']}} (Model A)\\")\\n",
    "        plt.scatter(model_b_tokens[token_col], model_b_tokens['time_taken_seconds'], \\n",
    "                   alpha=0.7, color='red', label=f\\"{{metadata['model_b']}} (Model B)\\")\\n",
    "        plt.xlabel(token_label)\\n",
    "        plt.ylabel('Time Taken (seconds)')\\n",
    "        plt.title(f'{{token_label}} vs Move Time')\\n",
    "        plt.legend()\\n",
    "        \\n",
    "        plt.subplot(2, 2, 2)\\n",
    "        plt.plot(model_a_tokens['move_number'], model_a_tokens[token_col], \\n",
    "                 'o-', label=f\\"{{metadata['model_a']}} (Model A)\\", color='blue', alpha=0.7)\\n",
    "        plt.plot(model_b_tokens['move_number'], model_b_tokens[token_col], \\n",
    "                 's-', label=f\\"{{metadata['model_b']}} (Model B)\\", color='red', alpha=0.7)\\n",
    "        plt.xlabel('Move Number')\\n",
    "        plt.ylabel(token_label)\\n",
    "        plt.title(f'{{token_label}} Throughout Game')\\n",
    "        plt.legend()\\n",
    "        \\n",
    "        plt.tight_layout()\\n",
    "        plt.show()\\n",
    "    else:\\n",
    "        print(\\"No thinking tokens data available\\")\\n",
    "        \\n",
    "    # Summary statistics\\n",
    "    print(\\"\\\\n=== GAME SUMMARY STATISTICS ===\\")\\n",
    "    print(f\\"Model A ({{metadata['model_a']}}) - Total moves: {{len(model_a_moves)}}\\")\\n",
    "    print(f\\"  Average move time: {{model_a_moves['time_taken_seconds'].mean():.2f}} seconds\\")\\n",
    "    print(f\\"  Total time used: {{model_a_moves['time_taken_seconds'].sum():.2f}} seconds\\")\\n",
    "    if has_thinking_tokens:\\n",
    "        avg_thinking = model_a_moves['thinking_tokens'].mean()\\n",
    "        if not np.isnan(avg_thinking):\\n",
    "            print(f\\"  Average thinking tokens: {{avg_thinking:.0f}}\\")\\n",
    "    \\n",
    "    print(f\\"\\\\nModel B ({{metadata['model_b']}}) - Total moves: {{len(model_b_moves)}}\\")\\n",
    "    print(f\\"  Average move time: {{model_b_moves['time_taken_seconds'].mean():.2f}} seconds\\")\\n",
    "    print(f\\"  Total time used: {{model_b_moves['time_taken_seconds'].sum():.2f}} seconds\\")\\n",
    "    if has_thinking_tokens:\\n",
    "        avg_thinking = model_b_moves['thinking_tokens'].mean()\\n",
    "        if not np.isnan(avg_thinking):\\n",
    "            print(f\\"  Average thinking tokens: {{avg_thinking:.0f}}\\")\\n",
    "else:\\n",
    "    print(\\"No move-level data available for analysis\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Move Quality Analysis (if available)\\n",
    "if move_analysis_df is not None and not move_analysis_df.empty:\\n",
    "    print(\\"\\\\n=== MOVE QUALITY ANALYSIS ===\\")\\n",
    "    print(f\\"Analyzing {{len(move_analysis_df)}} moves with Stockfish evaluation\\")\\n",
    "    \\n",
    "    # Separate move analysis by player\\n",
    "    model_a_analysis = move_analysis_df[move_analysis_df['player'] == 'Model A']\\n",
    "    model_b_analysis = move_analysis_df[move_analysis_df['player'] == 'Model B']\\n",
    "    \\n",
    "    if not model_a_analysis.empty and not model_b_analysis.empty:\\n",
    "        plt.figure(figsize=(12, 8))\\n",
    "        \\n",
    "        # Chart 1: Move Quality vs Time Taken\\n",
    "        plt.subplot(2, 2, 1)\\n",
    "        \\n",
    "        # Merge move analysis with timing data for the same game\\n",
    "        if game_moves_df is not None:\\n",
    "            # Merge on move_number and player to get timing data\\n",
    "            model_a_merged = pd.merge(model_a_analysis, \\n",
    "                                    game_moves_df[game_moves_df['who_played'] == 'Model A'], \\n",
    "                                    left_on='move_number', right_on='move_number', how='inner')\\n",
    "            model_b_merged = pd.merge(model_b_analysis, \\n",
    "                                    game_moves_df[game_moves_df['who_played'] == 'Model B'], \\n",
    "                                    left_on='move_number', right_on='move_number', how='inner')\\n",
    "            \\n",
    "            # Plot centipawn loss vs time taken\\n",
    "            plt.scatter(model_a_merged['time_taken_seconds'], model_a_merged['centipawn_loss'], \\n",
    "                       alpha=0.7, color='blue', s=50, label=f\\"{{metadata['model_a']}} (Model A)\\")\\n",
    "            plt.scatter(model_b_merged['time_taken_seconds'], model_b_merged['centipawn_loss'], \\n",
    "                       alpha=0.7, color='red', s=50, label=f\\"{{metadata['model_b']}} (Model B)\\")\\n",
    "            \\n",
    "            plt.xlabel('Time Taken (seconds)')\\n",
    "            plt.ylabel('Centipawn Loss')\\n",
    "            plt.title('Move Quality vs Time Taken')\\n",
    "            plt.legend()\\n",
    "            \\n",
    "            # Add quality reference lines\\n",
    "            plt.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Inaccuracy threshold')\\n",
    "            plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Mistake threshold')\\n",
    "            plt.axhline(y=100, color='darkred', linestyle='--', alpha=0.5, label='Blunder threshold')\\n",
    "        else:\\n",
    "            plt.text(0.5, 0.5, 'No timing data available for quality/time correlation', \\n",
    "                    ha='center', va='center', transform=plt.gca().transAxes)\\n",
    "        \\n",
    "        # Chart 2: Move Quality vs Turn Number\\n",
    "        plt.subplot(2, 2, 2)\\n",
    "        plt.plot(model_a_analysis['move_number'], model_a_analysis['centipawn_loss'], \\n",
    "                 'o-', label=f\\"{{metadata['model_a']}} (Model A)\\", color='blue', alpha=0.7, markersize=4)\\n",
    "        plt.plot(model_b_analysis['move_number'], model_b_analysis['centipawn_loss'], \\n",
    "                 's-', label=f\\"{{metadata['model_b']}} (Model B)\\", color='red', alpha=0.7, markersize=4)\\n",
    "        \\n",
    "        plt.xlabel('Move Number')\\n",
    "        plt.ylabel('Centipawn Loss')\\n",
    "        plt.title('Move Quality Throughout Game')\\n",
    "        plt.legend()\\n",
    "        \\n",
    "        # Add quality reference lines\\n",
    "        plt.axhline(y=25, color='orange', linestyle='--', alpha=0.5)\\n",
    "        plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)\\n",
    "        plt.axhline(y=100, color='darkred', linestyle='--', alpha=0.5)\\n",
    "        \\n",
    "        # Chart 3: Move Quality Distribution\\n",
    "        plt.subplot(2, 2, 3)\\n",
    "        plt.hist(model_a_analysis['centipawn_loss'], bins=20, alpha=0.7, \\n",
    "                 label=f\\"{{metadata['model_a']}} (Model A)\\", color='blue')\\n",
    "        plt.hist(model_b_analysis['centipawn_loss'], bins=20, alpha=0.7, \\n",
    "                 label=f\\"{{metadata['model_b']}} (Model B)\\", color='red')\\n",
    "        plt.xlabel('Centipawn Loss')\\n",
    "        plt.ylabel('Frequency')\\n",
    "        plt.title('Distribution of Move Quality')\\n",
    "        plt.legend()\\n",
    "        \\n",
    "        # Chart 4: Move Quality Summary Stats\\n",
    "        plt.subplot(2, 2, 4)\\n",
    "        plt.axis('off')\\n",
    "        \\n",
    "        # Calculate quality statistics\\n",
    "        def quality_stats(df):\\n",
    "            perfect = len(df[df['centipawn_loss'] == 0])\\n",
    "            good = len(df[(df['centipawn_loss'] > 0) & (df['centipawn_loss'] < 25)])\\n",
    "            inaccuracies = len(df[(df['centipawn_loss'] >= 25) & (df['centipawn_loss'] < 50)])\\n",
    "            mistakes = len(df[(df['centipawn_loss'] >= 50) & (df['centipawn_loss'] < 100)])\\n",
    "            blunders = len(df[df['centipawn_loss'] >= 100])\\n",
    "            return perfect, good, inaccuracies, mistakes, blunders\\n",
    "        \\n",
    "        a_perfect, a_good, a_inaccuracies, a_mistakes, a_blunders = quality_stats(model_a_analysis)\\n",
    "        b_perfect, b_good, b_inaccuracies, b_mistakes, b_blunders = quality_stats(model_b_analysis)\\n",
    "        \\n",
    "        stats_text = f\\\"\\\"\\\"MOVE QUALITY SUMMARY\\n",
    "\\n",
    "{{metadata['model_a']}} (Model A):\\n",
    "  Perfect moves (0 cp): {{a_perfect}} ({{a_perfect/len(model_a_analysis)*100:.1f}}%)\\n",
    "  Good moves (1-24 cp): {{a_good}} ({{a_good/len(model_a_analysis)*100:.1f}}%)\\n",
    "  Inaccuracies (25-49 cp): {{a_inaccuracies}} ({{a_inaccuracies/len(model_a_analysis)*100:.1f}}%)\\n",
    "  Mistakes (50-99 cp): {{a_mistakes}} ({{a_mistakes/len(model_a_analysis)*100:.1f}}%)\\n",
    "  Blunders (100+ cp): {{a_blunders}} ({{a_blunders/len(model_a_analysis)*100:.1f}}%)\\n",
    "  Avg centipawn loss: {{model_a_analysis['centipawn_loss'].mean():.1f}}\\n",
    "\\n",
    "{{metadata['model_b']}} (Model B):\\n",
    "  Perfect moves (0 cp): {{b_perfect}} ({{b_perfect/len(model_b_analysis)*100:.1f}}%)\\n",
    "  Good moves (1-24 cp): {{b_good}} ({{b_good/len(model_b_analysis)*100:.1f}}%)\\n",
    "  Inaccuracies (25-49 cp): {{b_inaccuracies}} ({{b_inaccuracies/len(model_b_analysis)*100:.1f}}%)\\n",
    "  Mistakes (50-99 cp): {{b_mistakes}} ({{b_mistakes/len(model_b_analysis)*100:.1f}}%)\\n",
    "  Blunders (100+ cp): {{b_blunders}} ({{b_blunders/len(model_b_analysis)*100:.1f}}%)\\n",
    "  Avg centipawn loss: {{model_b_analysis['centipawn_loss'].mean():.1f}}\\n",
    "\\\"\\\"\\\"\\n",
    "        \\n",
    "        plt.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top', \\n",
    "                fontfamily='monospace', transform=plt.gca().transAxes)\\n",
    "        \\n",
    "        plt.tight_layout()\\n",
    "        plt.show()\\n",
    "    else:\\n",
    "        print(\\"Insufficient move analysis data for visualization\\")\\n",
    "else:\\n",
    "    print(\\"\\\\n=== NO MOVE QUALITY ANALYSIS ===\\")\\n",
    "    print(\\"Move quality analysis not available.\\")\\n",
    "    print(\\"To generate move analysis, run:\\")\\n",
    "    print(\\"  python -m game_arena.blitz.move_analyzer <match_directory>\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Win Probability Analysis (if available)\\n",
    "if move_analysis_df is not None and not move_analysis_df.empty:\\n",
    "    # Check if win probability data is available\\n",
    "    has_win_prob = 'best_win_probability' in move_analysis_df.columns and 'played_win_probability' in move_analysis_df.columns\\n",
    "    \\n",
    "    if has_win_prob:\\n",
    "        print(\\"\\\\n=== WIN PROBABILITY ANALYSIS ===\\")\\n",
    "        print(f\\"Analyzing win likelihood for {{len(move_analysis_df)}} moves\\")\\n",
    "        \\n",
    "        # Separate win probability analysis by player\\n",
    "        model_a_win_analysis = move_analysis_df[move_analysis_df['player'] == 'Model A']\\n",
    "        model_b_win_analysis = move_analysis_df[move_analysis_df['player'] == 'Model B']\\n",
    "        \\n",
    "        if not model_a_win_analysis.empty and not model_b_win_analysis.empty:\\n",
    "            plt.figure(figsize=(15, 10))\\n",
    "            \\n",
    "            # Chart 1: Win Probability vs Time Taken\\n",
    "            plt.subplot(2, 3, 1)\\n",
    "            \\n",
    "            # Merge win probability with timing data\\n",
    "            if game_moves_df is not None:\\n",
    "                model_a_win_merged = pd.merge(model_a_win_analysis, \\n",
    "                                            game_moves_df[game_moves_df['who_played'] == 'Model A'], \\n",
    "                                            left_on='move_number', right_on='move_number', how='inner')\\n",
    "                model_b_win_merged = pd.merge(model_b_win_analysis, \\n",
    "                                            game_moves_df[game_moves_df['who_played'] == 'Model B'], \\n",
    "                                            left_on='move_number', right_on='move_number', how='inner')\\n",
    "                \\n",
    "                plt.scatter(model_a_win_merged['time_taken_seconds'], model_a_win_merged['played_win_probability'] * 100, \\n",
    "                           alpha=0.7, color='blue', s=50, label=f\\"{{metadata['model_a']}} (Model A)\\")\\n",
    "                plt.scatter(model_b_win_merged['time_taken_seconds'], model_b_win_merged['played_win_probability'] * 100, \\n",
    "                           alpha=0.7, color='red', s=50, label=f\\"{{metadata['model_b']}} (Model B)\\")\\n",
    "                \\n",
    "                plt.xlabel('Time Taken (seconds)')\\n",
    "                plt.ylabel('Win Probability (%)')\\n",
    "                plt.title('Win Probability vs Time Taken')\\n",
    "                plt.legend()\\n",
    "                plt.grid(True, alpha=0.3)\\n",
    "            else:\\n",
    "                plt.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=plt.gca().transAxes)\\n",
    "            \\n",
    "            # Chart 2: Win Probability Evolution Throughout Game\\n",
    "            plt.subplot(2, 3, 2)\\n",
    "            plt.plot(model_a_win_analysis['move_number'], model_a_win_analysis['played_win_probability'] * 100, \\n",
    "                     'o-', label=f\\"{{metadata['model_a']}} (Model A)\\", color='blue', alpha=0.7, markersize=4)\\n",
    "            plt.plot(model_b_win_analysis['move_number'], model_b_win_analysis['played_win_probability'] * 100, \\n",
    "                     's-', label=f\\"{{metadata['model_b']}} (Model B)\\", color='red', alpha=0.7, markersize=4)\\n",
    "            \\n",
    "            plt.xlabel('Move Number')\\n",
    "            plt.ylabel('Win Probability (%)')\\n",
    "            plt.title('Win Probability Throughout Game')\\n",
    "            plt.legend()\\n",
    "            plt.grid(True, alpha=0.3)\\n",
    "            plt.ylim(0, 100)\\n",
    "            \\n",
    "            # Chart 3: Win Probability Loss Distribution\\n",
    "            plt.subplot(2, 3, 3)\\n",
    "            plt.hist(model_a_win_analysis['win_probability_loss'] * 100, bins=20, alpha=0.7, \\n",
    "                     label=f\\"{{metadata['model_a']}} (Model A)\\", color='blue')\\n",
    "            plt.hist(model_b_win_analysis['win_probability_loss'] * 100, bins=20, alpha=0.7, \\n",
    "                     label=f\\"{{metadata['model_b']}} (Model B)\\", color='red')\\n",
    "            plt.xlabel('Win Probability Loss (%)')\\n",
    "            plt.ylabel('Frequency')\\n",
    "            plt.title('Distribution of Win Probability Loss')\\n",
    "            plt.legend()\\n",
    "            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='No loss')\\n",
    "            \\n",
    "            # Chart 4: Win Probability vs Centipawn Loss Correlation\\n",
    "            plt.subplot(2, 3, 4)\\n",
    "            plt.scatter(model_a_win_analysis['centipawn_loss'], model_a_win_analysis['win_probability_loss'] * 100, \\n",
    "                       alpha=0.7, color='blue', s=30, label=f\\"{{metadata['model_a']}} (Model A)\\")\\n",
    "            plt.scatter(model_b_win_analysis['centipawn_loss'], model_b_win_analysis['win_probability_loss'] * 100, \\n",
    "                       alpha=0.7, color='red', s=30, label=f\\"{{metadata['model_b']}} (Model B)\\")\\n",
    "            \\n",
    "            plt.xlabel('Centipawn Loss')\\n",
    "            plt.ylabel('Win Probability Loss (%)')\\n",
    "            plt.title('CP Loss vs Win Probability Loss')\\n",
    "            plt.legend()\\n",
    "            plt.grid(True, alpha=0.3)\\n",
    "            \\n",
    "            # Chart 5: Best vs Played Win Probability\\n",
    "            plt.subplot(2, 3, 5)\\n",
    "            plt.scatter(model_a_win_analysis['best_win_probability'] * 100, model_a_win_analysis['played_win_probability'] * 100, \\n",
    "                       alpha=0.7, color='blue', s=30, label=f\\"{{metadata['model_a']}} (Model A)\\")\\n",
    "            plt.scatter(model_b_win_analysis['best_win_probability'] * 100, model_b_win_analysis['played_win_probability'] * 100, \\n",
    "                       alpha=0.7, color='red', s=30, label=f\\"{{metadata['model_b']}} (Model B)\\")\\n",
    "            \\n",
    "            # Perfect play line (diagonal)\\n",
    "            plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect play')\\n",
    "            \\n",
    "            plt.xlabel('Best Win Probability (%)')\\n",
    "            plt.ylabel('Played Win Probability (%)')\\n",
    "            plt.title('Best vs Played Win Probability')\\n",
    "            plt.legend()\\n",
    "            plt.grid(True, alpha=0.3)\\n",
    "            plt.xlim(0, 100)\\n",
    "            plt.ylim(0, 100)\\n",
    "            \\n",
    "            # Chart 6: Win Probability Summary Statistics\\n",
    "            plt.subplot(2, 3, 6)\\n",
    "            plt.axis('off')\\n",
    "            \\n",
    "            # Calculate win probability statistics\\n",
    "            def win_prob_stats(df):\\n",
    "                avg_played_win_prob = df['played_win_probability'].mean() * 100\\n",
    "                avg_best_win_prob = df['best_win_probability'].mean() * 100\\n",
    "                avg_win_prob_loss = df['win_probability_loss'].mean() * 100\\n",
    "                max_win_prob_loss = df['win_probability_loss'].max() * 100\\n",
    "                big_losses = len(df[df['win_probability_loss'] > 0.1])  # >10% win prob loss\\n",
    "                return avg_played_win_prob, avg_best_win_prob, avg_win_prob_loss, max_win_prob_loss, big_losses\\n",
    "            \\n",
    "            a_avg_played, a_avg_best, a_avg_loss, a_max_loss, a_big_losses = win_prob_stats(model_a_win_analysis)\\n",
    "            b_avg_played, b_avg_best, b_avg_loss, b_max_loss, b_big_losses = win_prob_stats(model_b_win_analysis)\\n",
    "            \\n",
    "            stats_text = f\\\"\\\"\\\"WIN PROBABILITY SUMMARY\\n",
    "\\n",
    "{{metadata['model_a']}} (Model A):\\n",
    "  Avg played win prob: {{a_avg_played:.1f}}%\\n",
    "  Avg best win prob: {{a_avg_best:.1f}}%\\n",
    "  Avg win prob loss: {{a_avg_loss:.1f}}%\\n",
    "  Max win prob loss: {{a_max_loss:.1f}}%\\n",
    "  Major losses (>10%): {{a_big_losses}}\\n",
    "\\n",
    "{{metadata['model_b']}} (Model B):\\n",
    "  Avg played win prob: {{b_avg_played:.1f}}%\\n",
    "  Avg best win prob: {{b_avg_best:.1f}}%\\n",
    "  Avg win prob loss: {{b_avg_loss:.1f}}%\\n",
    "  Max win prob loss: {{b_max_loss:.1f}}%\\n",
    "  Major losses (>10%): {{b_big_losses}}\\n",
    "\\n",
    "Performance Comparison:\\n",
    "  Better avg accuracy: {{metadata['model_a'] if a_avg_loss < b_avg_loss else metadata['model_b']}}\\n",
    "  Fewer major losses: {{metadata['model_a'] if a_big_losses < b_big_losses else metadata['model_b'] if b_big_losses < a_big_losses else 'Tied'}}\\n",
    "\\\"\\\"\\\"\\n",
    "            \\n",
    "            plt.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top', \\n",
    "                    fontfamily='monospace', transform=plt.gca().transAxes)\\n",
    "            \\n",
    "            plt.tight_layout()\\n",
    "            plt.show()\\n",
    "        else:\\n",
    "            print(\\"Insufficient win probability data for visualization\\")\\n",
    "    else:\\n",
    "        print(\\"\\\\n=== WIN PROBABILITY DATA NOT AVAILABLE ===\\")\\n",
    "        print(\\"Win probability analysis requires move analysis with updated data format.\\")\\n",
    "        print(\\"Regenerate move analysis to include win probabilities.\\")\\n",
    "else:\\n",
    "    print(\\"\\\\n=== NO WIN PROBABILITY ANALYSIS ===\\")\\n",
    "    print(\\"Win probability analysis requires move quality data.\\")"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''
    
    notebook_path = Path(data_dir) / match_id / f"{match_id}_analysis.ipynb"
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print(f"📓 Analysis notebook created: {notebook_path}")


# Global data collector instance
_data_collector = None

def get_data_collector() -> BlitzDataCollector:
    """Get the global data collector instance."""
    global _data_collector
    if _data_collector is None:
        _data_collector = BlitzDataCollector()
    return _data_collector 