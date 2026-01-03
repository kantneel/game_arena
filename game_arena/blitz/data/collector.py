#!/usr/bin/env python3
"""Data collection for blitz chess matches."""

import csv
import datetime
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from game_arena.blitz.core.types import GameStats
from game_arena.blitz.data.types import MatchMetadata, GameRecord, GameMoveRecord
from game_arena.blitz.display.formatting import abbreviate_model_name


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
                   parser_choice: str,
                   dramatic_prompts_enabled: bool = False,
                   stateful_agents_enabled: bool = False,
                   dramatic_threshold_seconds: float = 60.0,
                   time_pressure_strategy: str = "none") -> str:
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
            parser_choice=parser_choice,
            dramatic_prompts_enabled=dramatic_prompts_enabled,
            stateful_agents_enabled=stateful_agents_enabled,
            dramatic_threshold_seconds=dramatic_threshold_seconds,
            time_pressure_strategy=time_pressure_strategy
        )
        
        # Save initial metadata so the match appears in the web UI immediately
        self._save_metadata_incremental()
        
        return match_id
    
    def record_move(self, game_number: int, who_played: str, move_played: str, 
                   board_state_before: str, time_taken: float, response_text: str,
                   time_at_turn_start: float, thinking_tokens: Optional[int],
                   output_tokens: Optional[int], total_tokens: Optional[int],
                   move_number: int, color: str, network_latency: float, 
                   retry_count: int,
                   # New time pressure parameters
                   time_pressure_level: str = "LOW",
                   used_dramatic_prompts: bool = False,
                   prompt_template_used: str = "NO_LEGAL_ACTIONS",
                   opponent_time_remaining: float = 0.0,
                   time_increment: int = 3,
                   previous_response_analysis_included: bool = False,
                   previous_move_time: Optional[float] = None,
                   previous_move_efficiency: Optional[float] = None) -> None:
        """Record a single move during gameplay."""
        if not self.current_match_id:
            raise ValueError("No active match. Call start_match() first.")
        
        # Calculate derived metrics
        reasoning_efficiency = None
        if thinking_tokens and time_taken > 0:
            reasoning_efficiency = thinking_tokens / time_taken
        
        # Determine time pressure category
        time_pressure_category = self._categorize_time_pressure(time_at_turn_start)
        
        # Determine time trend
        time_trend = self._calculate_time_trend(
            game_number, who_played, time_taken, previous_move_time
        )
        
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
            retry_count=retry_count,
            # New time pressure fields
            time_pressure_level=time_pressure_level,
            used_dramatic_prompts=used_dramatic_prompts,
            prompt_template_used=prompt_template_used,
            opponent_time_remaining=opponent_time_remaining,
            time_increment=time_increment,
            reasoning_efficiency=reasoning_efficiency,
            previous_response_analysis_included=previous_response_analysis_included,
            time_pressure_category=time_pressure_category,
            previous_move_time=previous_move_time,
            previous_move_efficiency=previous_move_efficiency,
            time_trend=time_trend
        )
        
        # Initialize game moves list if needed
        if game_number not in self.per_game_moves:
            self.per_game_moves[game_number] = []
        
        self.per_game_moves[game_number].append(move_record)
        
        # Save moves incrementally after each move for live viewing
        self._save_game_moves_csv(game_number)
        
        # Update heartbeat and current game after every move for live viewing
        if self.match_metadata:
            self.match_metadata.current_game = game_number
            self.match_metadata.last_updated = datetime.datetime.now()
            self._save_metadata_incremental()
    
    def _categorize_time_pressure(self, time_remaining: float) -> str:
        """Categorize time pressure based on time remaining."""
        if time_remaining < 30:
            return "under_30s"
        elif time_remaining < 60:
            return "under_60s"
        elif time_remaining < 120:
            return "under_120s"
        else:
            return "comfortable"
    
    def _calculate_time_trend(self, game_number: int, player: str, current_time: float, 
                             previous_time: Optional[float]) -> str:
        """Calculate whether player is speeding up or slowing down."""
        if previous_time is None:
            return "first_move"
        
        time_diff = current_time - previous_time
        threshold = 2.0  # seconds
        
        if abs(time_diff) < threshold:
            return "stable"
        elif time_diff > 0:
            return "slowing_down"
        else:
            return "speeding_up"
    
    def get_previous_move_data(self, game_number: int, player: str) -> tuple:
        """Get the previous move's time and efficiency for a player."""
        if game_number not in self.per_game_moves:
            return None, None
        
        moves = self.per_game_moves[game_number]
        player_moves = [m for m in moves if m.who_played == player]
        
        if len(player_moves) == 0:
            return None, None
        
        last_move = player_moves[-1]
        return last_move.time_taken_seconds, last_move.reasoning_efficiency
    
    def record_game(self, game_stats: GameStats, initial_time: float, increment: float) -> None:
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
    
    def _determine_termination_reason(self, game_stats: GameStats) -> str:
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
        
        # Also save metadata incrementally so web UI stays updated
        self._save_metadata_incremental()
    
    def _save_metadata_incremental(self) -> None:
        """Save metadata.json incrementally during match for live updates."""
        if not self.match_metadata or not self.current_match_dir:
            return
        
        # Update heartbeat timestamp
        self.match_metadata.last_updated = datetime.datetime.now()
        
        # Calculate current scores from game records
        model_a_wins = sum(1 for g in self.game_records if g.winner == "model_a")
        model_b_wins = sum(1 for g in self.game_records if g.winner == "model_b")
        draws = sum(1 for g in self.game_records if g.winner == "draw")
        
        # Update metadata with current scores
        self.match_metadata.total_games = len(self.game_records)
        self.match_metadata.model_a_wins = model_a_wins
        self.match_metadata.model_b_wins = model_b_wins
        self.match_metadata.draws = draws
        
        # Build metadata dict
        metadata_dict = asdict(self.match_metadata)
        
        # Convert datetime objects to strings
        for key, value in metadata_dict.items():
            if isinstance(value, datetime.datetime):
                metadata_dict[key] = value.isoformat()
        
        # Write to file
        with open(self.current_match_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def update_heartbeat(self, current_game: int = None) -> None:
        """Update the heartbeat timestamp (call this periodically during gameplay)."""
        if self.match_metadata:
            self.match_metadata.last_updated = datetime.datetime.now()
            if current_game is not None:
                self.match_metadata.current_game = current_game
            self._save_metadata_incremental()
    
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
        
        # Save time pressure analysis
        if self.match_metadata and (self.match_metadata.dramatic_prompts_enabled or 
                                   self.match_metadata.stateful_agents_enabled):
            time_pressure_analysis = self.generate_time_pressure_analysis()
            if time_pressure_analysis:
                with open(self.current_match_dir / "time_pressure_analysis.json", 'w') as f:
                    json.dump(time_pressure_analysis, f, indent=2)
                print(f"📊 Time pressure analysis saved")
        
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
    
    def generate_time_pressure_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive time pressure analysis."""
        if not self.per_game_moves:
            return {}
        
        all_moves = []
        for game_moves in self.per_game_moves.values():
            all_moves.extend(game_moves)
        
        analysis = {
            "time_pressure_distribution": self._analyze_time_pressure_distribution(all_moves),
            "dramatic_prompt_effectiveness": self._analyze_dramatic_prompt_effectiveness(all_moves),
            "stateful_analysis_impact": self._analyze_stateful_impact(all_moves),
            "time_trend_patterns": self._analyze_time_trends(all_moves),
            "reasoning_efficiency_by_pressure": self._analyze_efficiency_by_pressure(all_moves),
        }
        
        return analysis
    
    def _analyze_time_pressure_distribution(self, moves: List[GameMoveRecord]) -> Dict[str, Any]:
        """Analyze distribution of moves across time pressure categories."""
        pressure_counts = {"under_30s": 0, "under_60s": 0, "under_120s": 0, "comfortable": 0}
        pressure_avg_times = {"under_30s": [], "under_60s": [], "under_120s": [], "comfortable": []}
        
        for move in moves:
            category = move.time_pressure_category
            pressure_counts[category] += 1
            pressure_avg_times[category].append(move.time_taken_seconds)
        
        return {
            "move_counts_by_pressure": pressure_counts,
            "avg_time_by_pressure": {
                k: sum(v) / len(v) if v else 0 
                for k, v in pressure_avg_times.items()
            },
            "total_moves": len(moves)
        }
    
    def _analyze_dramatic_prompt_effectiveness(self, moves: List[GameMoveRecord]) -> Dict[str, Any]:
        """Analyze effectiveness of dramatic prompts vs normal prompts."""
        dramatic_moves = [m for m in moves if m.used_dramatic_prompts]
        normal_moves = [m for m in moves if not m.used_dramatic_prompts]
        
        if not dramatic_moves:
            return {"error": "No dramatic prompt moves found"}
        
        return {
            "dramatic_moves_count": len(dramatic_moves),
            "normal_moves_count": len(normal_moves),
            "dramatic_avg_time": sum(m.time_taken_seconds for m in dramatic_moves) / len(dramatic_moves),
            "normal_avg_time": sum(m.time_taken_seconds for m in normal_moves) / len(normal_moves) if normal_moves else 0,
        }
    
    def _analyze_stateful_impact(self, moves: List[GameMoveRecord]) -> Dict[str, Any]:
        """Analyze impact of stateful response analysis."""
        stateful_moves = [m for m in moves if m.previous_response_analysis_included]
        non_stateful_moves = [m for m in moves if not m.previous_response_analysis_included]
        
        if not stateful_moves:
            return {"error": "No stateful moves found"}
        
        return {
            "stateful_moves_count": len(stateful_moves),
            "non_stateful_moves_count": len(non_stateful_moves),
            "stateful_avg_time": sum(m.time_taken_seconds for m in stateful_moves) / len(stateful_moves),
            "non_stateful_avg_time": sum(m.time_taken_seconds for m in non_stateful_moves) / len(non_stateful_moves) if non_stateful_moves else 0,
        }
    
    def _analyze_time_trends(self, moves: List[GameMoveRecord]) -> Dict[str, Any]:
        """Analyze time trend patterns."""
        trend_counts = {"speeding_up": 0, "slowing_down": 0, "stable": 0, "first_move": 0}
        
        for move in moves:
            if move.time_trend:
                trend_counts[move.time_trend] += 1
        
        return trend_counts
    
    def _analyze_efficiency_by_pressure(self, moves: List[GameMoveRecord]) -> Dict[str, Any]:
        """Analyze reasoning efficiency by time pressure level."""
        pressure_efficiency = {"under_30s": [], "under_60s": [], "under_120s": [], "comfortable": []}
        
        for move in moves:
            if move.reasoning_efficiency is not None:
                pressure_efficiency[move.time_pressure_category].append(move.reasoning_efficiency)
        
        return {
            category: {
                "avg_efficiency": sum(effs) / len(effs) if effs else 0,
                "move_count": len(effs)
            }
            for category, effs in pressure_efficiency.items()
        }


# Global data collector instance
_data_collector = None


def get_data_collector() -> BlitzDataCollector:
    """Get the global data collector instance."""
    global _data_collector
    if _data_collector is None:
        _data_collector = BlitzDataCollector()
    return _data_collector

