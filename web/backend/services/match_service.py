#!/usr/bin/env python3
"""Match service for reading match data from filesystem."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from models.schemas import (
    MatchSummary,
    MatchDetail,
    GameSummary,
    GameDetail,
    MoveRecord,
)


class MatchService:
    """Service for loading and querying match data from the filesystem."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.matches: dict[str, dict] = {}  # match_id -> metadata
        self._games_cache: dict[str, pd.DataFrame] = {}  # match_id -> games_df
    
    def scan_results(self) -> None:
        """Scan the results directory and load all match metadata."""
        if not self.results_dir.exists():
            print(f"⚠️ Results directory not found: {self.results_dir}")
            return
        
        for match_dir in self.results_dir.iterdir():
            if not match_dir.is_dir():
                continue
            
            metadata_file = match_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    metadata["_dir"] = str(match_dir)
                    self.matches[metadata["match_id"]] = metadata
                except Exception as e:
                    print(f"⚠️ Failed to load {metadata_file}: {e}")
    
    def get_all_matches(self, limit: int = 50, offset: int = 0) -> list[MatchSummary]:
        """Get all matches, sorted by start time (newest first)."""
        sorted_matches = sorted(
            self.matches.values(),
            key=lambda m: m.get("start_time", ""),
            reverse=True
        )
        
        result = []
        for m in sorted_matches[offset:offset + limit]:
            result.append(self._to_match_summary(m))
        
        return result
    
    def get_match(self, match_id: str) -> Optional[MatchDetail]:
        """Get full details for a specific match."""
        if match_id not in self.matches:
            return None
        
        metadata = self.matches[match_id]
        match_dir = Path(metadata["_dir"])
        
        # Load games
        games = self._load_games(match_dir, metadata)
        
        # Current game is either from metadata, or inferred from completed games + 1
        current_game = metadata.get("current_game", 0)
        if not current_game:
            current_game = len(games) + 1 if not metadata.get("end_time") else len(games)
        
        return MatchDetail(
            match_id=match_id,
            model_a=metadata.get("model_a", ""),
            model_b=metadata.get("model_b", ""),
            model_a_score=metadata.get("model_a_wins", 0),
            model_b_score=metadata.get("model_b_wins", 0),
            draws=metadata.get("draws", 0),
            winner=metadata.get("final_winner", ""),
            total_games=metadata.get("total_games", 0),
            started_at=self._parse_datetime(metadata.get("start_time")),
            ended_at=self._parse_datetime(metadata.get("end_time")),
            time_control=metadata.get("time_control", ""),
            rethinking_enabled=metadata.get("rethinking_enabled", False),
            games=games,
            current_game=current_game,
            notes=metadata.get("notes"),
        )
    
    def get_game(self, match_id: str, game_number: int) -> Optional[GameDetail]:
        """Get full details for a specific game including moves."""
        if match_id not in self.matches:
            return None
        
        metadata = self.matches[match_id]
        match_dir = Path(metadata["_dir"])
        
        # Load games summary
        games_file = match_dir / "games_summary.csv"
        if not games_file.exists():
            return None
        
        games_df = pd.read_csv(games_file)
        game_row = games_df[games_df["game_number"] == game_number]
        if game_row.empty:
            return None
        
        game_data = game_row.iloc[0].to_dict()
        
        # Load moves
        moves_file = match_dir / f"game_{game_number}_moves.csv"
        moves = []
        
        # Load Stockfish analysis if available
        analysis_data = self._load_move_analysis(match_dir, game_number)
        
        if moves_file.exists():
            moves_df = pd.read_csv(moves_file)
            for _, row in moves_df.iterrows():
                move_num = int(row.get("move_number", 0))
                
                # Get analysis for this move if available
                analysis = analysis_data.get(move_num, {}) if analysis_data else {}
                
                # Calculate time remaining AFTER the move (time_before - time_taken + increment)
                time_before = float(row.get("time_available_at_turn_start", 0))
                time_taken = float(row.get("time_taken_seconds", 0))
                increment = float(row.get("time_increment", 3))
                time_after = max(0, time_before - time_taken + increment)
                
                moves.append(MoveRecord(
                    move_number=move_num,
                    player=row.get("who_played", ""),
                    color=row.get("color", ""),
                    move=row.get("move_played", ""),
                    fen_before=row.get("board_state_before_move", ""),
                    time_taken=time_taken,
                    time_remaining=time_after,
                    thinking_tokens=self._safe_int(row.get("thinking_tokens")),
                    centipawn_loss=self._safe_float(analysis.get("centipawn_loss")),
                    is_best_move=analysis.get("played_move_rank_among_top") == 1.0 if analysis.get("played_move_rank_among_top") is not None else None,
                    is_blunder=self._safe_float(analysis.get("centipawn_loss", 0)) >= 100 if analysis.get("centipawn_loss") is not None else None,
                    best_move=analysis.get("best_move_san"),
                    win_probability_loss=self._safe_float(analysis.get("win_probability_loss")),
                    num_legal_moves=self._safe_int(analysis.get("num_legal_moves")),
                    eval_sharpness=self._safe_int(analysis.get("eval_sharpness")),
                    position_eval_abs=self._safe_int(analysis.get("position_eval_abs")),
                ))
        
        # Determine white/black models
        model_a = metadata.get("model_a", "Model A")
        model_b = metadata.get("model_b", "Model B")
        model_a_color = game_data.get("model_a_color", "white")
        
        if model_a_color == "white":
            white_model, black_model = model_a, model_b
        else:
            white_model, black_model = model_b, model_a
        
        return GameDetail(
            game_number=game_number,
            match_id=match_id,
            white_model=white_model,
            black_model=black_model,
            result=game_data.get("result_string", ""),
            winner=game_data.get("winner", ""),
            termination=game_data.get("termination_reason", ""),
            total_moves=int(game_data.get("total_moves", 0)),
            duration_seconds=float(game_data.get("game_duration_seconds", 0)),
            moves=moves,
        )
    
    def get_live_game(self, match_id: str, game_number: int) -> Optional[GameDetail]:
        """Get game data for a live game (reads moves directly from moves file).
        
        This works even if the game hasn't been recorded in games_summary.csv yet.
        """
        if match_id not in self.matches:
            return None
        
        metadata = self.matches[match_id]
        match_dir = Path(metadata["_dir"])
        
        # First try to load from the completed game method
        existing_game = self.get_game(match_id, game_number)
        if existing_game and existing_game.moves:
            return existing_game
        
        # Load moves from file directly for live games
        moves_file = match_dir / f"game_{game_number}_moves.csv"
        moves = []
        
        # Load Stockfish analysis if available
        analysis_data = self._load_move_analysis(match_dir, game_number)
        
        if moves_file.exists():
            try:
                moves_df = pd.read_csv(moves_file)
                for _, row in moves_df.iterrows():
                    move_num = int(row.get("move_number", 0))
                    
                    # Get analysis for this move if available
                    analysis = analysis_data.get(move_num, {}) if analysis_data else {}
                    
                    # Calculate time remaining AFTER the move (time_before - time_taken + increment)
                    time_before = float(row.get("time_available_at_turn_start", 0))
                    time_taken = float(row.get("time_taken_seconds", 0))
                    increment = float(row.get("time_increment", 3))
                    time_after = max(0, time_before - time_taken + increment)
                    
                    moves.append(MoveRecord(
                        move_number=move_num,
                        player=row.get("who_played", ""),
                        color=row.get("color", ""),
                        move=row.get("move_played", ""),
                        fen_before=row.get("board_state_before_move", ""),
                        time_taken=time_taken,
                        time_remaining=time_after,
                        thinking_tokens=self._safe_int(row.get("thinking_tokens")),
                        centipawn_loss=self._safe_float(analysis.get("centipawn_loss")),
                        is_best_move=analysis.get("played_move_rank_among_top") == 1.0 if analysis.get("played_move_rank_among_top") is not None else None,
                        is_blunder=self._safe_float(analysis.get("centipawn_loss", 0)) >= 100 if analysis.get("centipawn_loss") is not None else None,
                        best_move=analysis.get("best_move_san"),
                        win_probability_loss=self._safe_float(analysis.get("win_probability_loss")),
                        num_legal_moves=self._safe_int(analysis.get("num_legal_moves")),
                        eval_sharpness=self._safe_int(analysis.get("eval_sharpness")),
                        position_eval_abs=self._safe_int(analysis.get("position_eval_abs")),
                    ))
            except Exception as e:
                print(f"Error reading moves file: {e}")
        
        # Determine white/black models from metadata
        model_a = metadata.get("model_a", "Model A")
        model_b = metadata.get("model_b", "Model B")
        
        # For odd game numbers, model_a is white; for even, model_b is white
        if game_number % 2 == 1:
            white_model, black_model = model_a, model_b
        else:
            white_model, black_model = model_b, model_a
        
        return GameDetail(
            game_number=game_number,
            match_id=match_id,
            white_model=white_model,
            black_model=black_model,
            result="in_progress",
            winner="",
            termination="",
            total_moves=len(moves),
            duration_seconds=0,
            moves=moves,
        )

    def _load_games(self, match_dir: Path, metadata: dict) -> list[GameSummary]:
        """Load game summaries from CSV."""
        games_file = match_dir / "games_summary.csv"
        if not games_file.exists():
            return []
        
        games_df = pd.read_csv(games_file)
        model_a = metadata.get("model_a", "Model A")
        model_b = metadata.get("model_b", "Model B")
        
        games = []
        for _, row in games_df.iterrows():
            model_a_color = row.get("model_a_color", "white")
            if model_a_color == "white":
                white_model, black_model = model_a, model_b
            else:
                white_model, black_model = model_b, model_a
            
            # Recalculate termination reason based on clock times (fixes old data bug)
            stored_termination = row.get("termination_reason", "")
            model_a_final = float(row.get("model_a_final_time", 999))
            model_b_final = float(row.get("model_b_final_time", 999))
            model_a_failures = int(row.get("model_a_parsing_failures", 0))
            model_b_failures = int(row.get("model_b_parsing_failures", 0))
            
            if model_a_final <= 0 or model_b_final <= 0:
                termination = "time_forfeit"
            elif model_a_failures >= 3 or model_b_failures >= 3:
                termination = "parsing_failure"
            else:
                termination = stored_termination
            
            games.append(GameSummary(
                game_number=int(row.get("game_number", 0)),
                white_model=white_model,
                black_model=black_model,
                result=row.get("result_string", ""),
                winner=row.get("winner", ""),
                termination=termination,
                total_moves=int(row.get("total_moves", 0)),
                duration_seconds=float(row.get("game_duration_seconds", 0)),
            ))
        
        return games
    
    def _to_match_summary(self, metadata: dict) -> MatchSummary:
        """Convert metadata dict to MatchSummary."""
        # Determine status: completed, live, or abandoned
        model_a_wins = metadata.get("model_a_wins", 0)
        model_b_wins = metadata.get("model_b_wins", 0)
        max_wins = max(model_a_wins, model_b_wins)
        
        # Get first_to from metadata, or infer from max wins for older matches
        first_to = metadata.get("first_to")
        if first_to is None:
            # For old matches without first_to, infer from the max wins
            # If someone has won, that's likely the first_to value
            first_to = max_wins if max_wins > 0 else 1
        
        # Check if match is complete based on score
        # A match is complete if one player has reached the first_to threshold
        is_score_complete = max_wins >= first_to or (max_wins >= 1 and metadata.get("final_winner"))
        
        if metadata.get("end_time") or is_score_complete:
            status = "completed"
            # Determine winner from score if not set
            if not metadata.get("final_winner") and model_a_wins != model_b_wins:
                if model_a_wins > model_b_wins:
                    metadata["final_winner"] = "model_a"
                else:
                    metadata["final_winner"] = "model_b"
        else:
            # Check if match is actually live (updated recently) or abandoned
            last_updated = metadata.get("last_updated") or metadata.get("start_time")
            if last_updated:
                try:
                    update_time = datetime.fromisoformat(last_updated)
                    time_since_update = (datetime.now() - update_time).total_seconds()
                    # If not updated in 30 minutes, consider it abandoned
                    status = "live" if time_since_update < 1800 else "completed"
                except (ValueError, TypeError):
                    status = "completed"
            else:
                status = "completed"
        
        return MatchSummary(
            match_id=metadata.get("match_id", ""),
            model_a=metadata.get("model_a", ""),
            model_b=metadata.get("model_b", ""),
            model_a_score=metadata.get("model_a_wins", 0),
            model_b_score=metadata.get("model_b_wins", 0),
            draws=metadata.get("draws", 0),
            winner=metadata.get("final_winner", ""),
            total_games=metadata.get("total_games", 0),
            started_at=self._parse_datetime(metadata.get("start_time")),
            ended_at=self._parse_datetime(metadata.get("end_time")),
            time_control=metadata.get("time_control", ""),
            status=status,
            notes=metadata.get("notes"),
        )
    
    def _parse_datetime(self, dt_str: Optional[str]) -> datetime:
        """Parse datetime string to datetime object."""
        if not dt_str:
            return datetime.now()
        try:
            return datetime.fromisoformat(dt_str)
        except ValueError:
            return datetime.now()
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert value to int, returning None if not possible."""
        if pd.isna(value):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float, returning None if not possible."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _load_move_analysis(self, match_dir: Path, game_number: int) -> Optional[dict]:
        """Load Stockfish move analysis for a specific game.
        
        Returns a dict mapping move_number -> analysis data.
        """
        # Try game-specific analysis file first
        analysis_file = match_dir / f"game_{game_number}_move_analysis.csv"
        
        # Fall back to complete analysis file
        if not analysis_file.exists():
            analysis_file = match_dir / "complete_move_analysis.csv"
        
        if not analysis_file.exists():
            return None
        
        try:
            df = pd.read_csv(analysis_file)
            
            # Filter to specific game if using complete file
            if "game_number" in df.columns:
                df = df[df["game_number"] == game_number]
            
            if df.empty:
                return None
            
            # Build lookup dict by move number
            result = {}
            for _, row in df.iterrows():
                move_num = int(row.get("move_number", 0))
                result[move_num] = row.to_dict()
            
            return result
        except Exception as e:
            print(f"Error loading move analysis: {e}")
            return None

