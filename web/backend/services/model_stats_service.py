#!/usr/bin/env python3
"""Aggregated model statistics service."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from services.analysis_service import AnalysisService, ModelPressureProfile


@dataclass
class ModelMatchRecord:
    """Record of a model's participation in a match."""
    match_id: str
    opponent: str
    wins: int
    losses: int
    draws: int
    result: str  # "win", "loss", "draw"
    date: datetime


@dataclass
class PressureLevelStats:
    """Aggregated stats for a pressure level across all matches."""
    pressure_level: str
    total_moves: int
    avg_move_time: float
    avg_thinking_tokens: Optional[float]
    avg_centipawn_loss: Optional[float]
    blunder_rate: float
    # Win rate when games reach this pressure level
    games_at_this_pressure: int
    win_rate_at_pressure: float


@dataclass
class ModelProfile:
    """Complete profile for a model across all matches."""
    model_id: str
    display_name: str
    
    # Overall stats
    total_matches: int
    total_games: int
    total_moves: int
    wins: int
    losses: int
    draws: int
    elo: int
    win_rate: float
    
    # Time management
    avg_move_time: float
    avg_thinking_tokens: Optional[float]
    
    # Pressure profile
    pressure_stats: list[PressureLevelStats]
    speed_adaptation_ratio: float
    quality_degradation_ratio: float
    thinking_reduction_ratio: float
    
    # Match history
    recent_matches: list[ModelMatchRecord]


class ModelStatsService:
    """Service for aggregating model statistics across matches."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.analysis_service = AnalysisService(results_dir)
        self._model_cache: dict[str, ModelProfile] = {}
    
    def get_all_models(self) -> list[dict]:
        """Get list of all models with basic stats."""
        models = {}
        
        for match_dir in self.results_dir.iterdir():
            if not match_dir.is_dir():
                continue
            
            metadata = self._load_metadata(match_dir)
            if not metadata:
                continue
            
            model_a = metadata.get("model_a", "")
            model_b = metadata.get("model_b", "")
            
            for model in [model_a, model_b]:
                if not model:
                    continue
                
                if model not in models:
                    models[model] = {
                        "model_id": model,
                        "display_name": self._format_display_name(model),
                        "matches": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                    }
                
                models[model]["matches"] += 1
                
                # Count wins/losses
                winner = metadata.get("final_winner", "")
                if winner == "model_a" and model == model_a:
                    models[model]["wins"] += 1
                elif winner == "model_b" and model == model_b:
                    models[model]["wins"] += 1
                elif winner == "draw":
                    models[model]["draws"] += 1
                elif winner:
                    models[model]["losses"] += 1
        
        # Calculate win rates and sort
        result = []
        for model_data in models.values():
            total = model_data["wins"] + model_data["losses"] + model_data["draws"]
            model_data["win_rate"] = model_data["wins"] / total if total > 0 else 0
            result.append(model_data)
        
        result.sort(key=lambda x: x["win_rate"], reverse=True)
        return result
    
    def get_model_profile(self, model_id: str) -> Optional[ModelProfile]:
        """Get complete profile for a specific model."""
        # Find all matches this model participated in
        match_records = []
        all_match_ids = []
        total_games = 0
        total_moves = 0
        wins = 0
        losses = 0
        draws = 0
        all_move_times = []
        all_thinking_tokens = []
        
        for match_dir in self.results_dir.iterdir():
            if not match_dir.is_dir():
                continue
            
            metadata = self._load_metadata(match_dir)
            if not metadata:
                continue
            
            model_a = metadata.get("model_a", "")
            model_b = metadata.get("model_b", "")
            
            if model_id not in [model_a, model_b]:
                continue
            
            match_id = metadata.get("match_id", match_dir.name)
            all_match_ids.append(match_id)
            
            # Determine opponent and result
            is_model_a = model_id == model_a
            opponent = model_b if is_model_a else model_a
            
            model_wins = metadata.get("model_a_wins", 0) if is_model_a else metadata.get("model_b_wins", 0)
            model_losses = metadata.get("model_b_wins", 0) if is_model_a else metadata.get("model_a_wins", 0)
            match_draws = metadata.get("draws", 0)
            
            winner = metadata.get("final_winner", "")
            if winner == "model_a":
                result = "win" if is_model_a else "loss"
            elif winner == "model_b":
                result = "loss" if is_model_a else "win"
            else:
                result = "draw"
            
            # Update totals
            total_games += model_wins + model_losses + match_draws
            wins += model_wins
            losses += model_losses
            draws += match_draws
            
            # Parse date
            try:
                date = datetime.fromisoformat(metadata.get("start_time", ""))
            except:
                date = datetime.now()
            
            match_records.append(ModelMatchRecord(
                match_id=match_id,
                opponent=opponent,
                wins=model_wins,
                losses=model_losses,
                draws=match_draws,
                result=result,
                date=date,
            ))
            
            # Load moves for this model
            moves = self._load_model_moves(match_dir, model_id)
            if not moves.empty:
                total_moves += len(moves)
                all_move_times.extend(moves["time_taken_seconds"].tolist())
                if "thinking_tokens" in moves.columns:
                    valid_tokens = moves["thinking_tokens"].dropna()
                    all_thinking_tokens.extend(valid_tokens.tolist())
        
        if not match_records:
            return None
        
        # Sort matches by date (most recent first)
        match_records.sort(key=lambda x: x.date, reverse=True)
        
        # Get aggregated pressure profile
        pressure_profile = self.analysis_service.analyze_model_aggregate(model_id, all_match_ids)
        
        # Build pressure level stats
        pressure_stats = []
        if pressure_profile:
            for ps in pressure_profile.pressure_stats:
                pressure_stats.append(PressureLevelStats(
                    pressure_level=ps.pressure_level,
                    total_moves=ps.move_count,
                    avg_move_time=ps.avg_move_time,
                    avg_thinking_tokens=ps.avg_thinking_tokens,
                    avg_centipawn_loss=ps.avg_centipawn_loss,
                    blunder_rate=ps.blunder_rate,
                    games_at_this_pressure=0,  # TODO: compute
                    win_rate_at_pressure=0.0,  # TODO: compute
                ))
        
        return ModelProfile(
            model_id=model_id,
            display_name=self._format_display_name(model_id),
            total_matches=len(match_records),
            total_games=total_games,
            total_moves=total_moves,
            wins=wins,
            losses=losses,
            draws=draws,
            elo=1500,  # TODO: get from ELO service
            win_rate=wins / total_games if total_games > 0 else 0,
            avg_move_time=sum(all_move_times) / len(all_move_times) if all_move_times else 0,
            avg_thinking_tokens=sum(all_thinking_tokens) / len(all_thinking_tokens) if all_thinking_tokens else None,
            pressure_stats=pressure_stats,
            speed_adaptation_ratio=pressure_profile.speed_adaptation_ratio if pressure_profile else 1.0,
            quality_degradation_ratio=pressure_profile.quality_degradation_ratio if pressure_profile else 1.0,
            thinking_reduction_ratio=pressure_profile.thinking_reduction_ratio if pressure_profile else 1.0,
            recent_matches=match_records[:10],
        )
    
    def get_model_comparison(self, model_ids: list[str]) -> dict:
        """Compare multiple models' pressure behavior."""
        comparison = {
            "models": [],
            "pressure_levels": ["comfortable", "medium", "high", "critical"],
        }
        
        for model_id in model_ids:
            profile = self.get_model_profile(model_id)
            if not profile:
                continue
            
            model_data = {
                "model_id": model_id,
                "display_name": profile.display_name,
                "speed_adaptation": profile.speed_adaptation_ratio,
                "quality_degradation": profile.quality_degradation_ratio,
                "thinking_reduction": profile.thinking_reduction_ratio,
                "pressure_data": {},
            }
            
            for ps in profile.pressure_stats:
                model_data["pressure_data"][ps.pressure_level] = {
                    "avg_move_time": ps.avg_move_time,
                    "avg_thinking_tokens": ps.avg_thinking_tokens,
                    "avg_centipawn_loss": ps.avg_centipawn_loss,
                    "blunder_rate": ps.blunder_rate,
                }
            
            comparison["models"].append(model_data)
        
        return comparison
    
    def _load_metadata(self, match_dir: Path) -> Optional[dict]:
        """Load match metadata."""
        import json
        metadata_file = match_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file) as f:
            return json.load(f)
    
    def _load_model_moves(self, match_dir: Path, model_id: str) -> pd.DataFrame:
        """Load all moves for a specific model from a match."""
        move_files = list(match_dir.glob("game_*_moves.csv"))
        if not move_files:
            return pd.DataFrame()
        
        dfs = []
        for f in move_files:
            try:
                df = pd.read_csv(f)
                # Filter to this model's moves
                model_moves = df[df["who_played"] == model_id]
                if not model_moves.empty:
                    dfs.append(model_moves)
            except Exception:
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    def _format_display_name(self, model_id: str) -> str:
        """Format model ID into display name."""
        return model_id.replace("-", " ").title()

