#!/usr/bin/env python3
"""Time pressure analysis service for chess matches."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class PressureStats:
    """Statistics for a specific pressure level."""
    pressure_level: str
    move_count: int
    avg_move_time: float
    std_move_time: float
    avg_thinking_tokens: Optional[float]
    avg_centipawn_loss: Optional[float]
    blunder_rate: float  # Moves with >100cp loss


@dataclass
class ModelPressureProfile:
    """Complete pressure profile for a model in a match."""
    model_name: str
    total_moves: int
    pressure_stats: list[PressureStats]
    # Adaptation metrics
    speed_adaptation_ratio: float  # How much faster at <60s vs >120s
    quality_degradation_ratio: float  # How much worse at <60s vs >120s
    thinking_reduction_ratio: float  # How much less thinking at <60s vs >120s


@dataclass
class MatchAnalysis:
    """Complete analysis for a match."""
    match_id: str
    model_a_profile: ModelPressureProfile
    model_b_profile: ModelPressureProfile
    insights: list[str]


# Pressure level thresholds (in seconds)
PRESSURE_THRESHOLDS = {
    "critical": (0, 30),
    "high": (30, 60),
    "medium": (60, 120),
    "comfortable": (120, float("inf")),
}


def categorize_pressure(time_remaining: float) -> str:
    """Categorize time remaining into pressure level."""
    for level, (low, high) in PRESSURE_THRESHOLDS.items():
        if low <= time_remaining < high:
            return level
    return "comfortable"


class AnalysisService:
    """Service for analyzing time pressure behavior in chess matches."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
    
    def analyze_match(self, match_id: str) -> Optional[MatchAnalysis]:
        """Perform complete time pressure analysis for a match."""
        match_dir = self.results_dir / match_id
        if not match_dir.exists():
            return None
        
        # Load metadata
        metadata = self._load_metadata(match_dir)
        if not metadata:
            return None
        
        # Load all moves from all games
        all_moves = self._load_all_moves(match_dir)
        if all_moves.empty:
            return None
        
        # Load move analysis if available (for centipawn loss)
        move_analysis = self._load_move_analysis(match_dir)
        if move_analysis is not None and not move_analysis.empty:
            all_moves = self._merge_with_analysis(all_moves, move_analysis)
        
        # Compute profiles for each model
        model_a = metadata.get("model_a", "Model A")
        model_b = metadata.get("model_b", "Model B")
        
        # Map "Model A" / "Model B" in CSV to actual model names
        all_moves["model_name"] = all_moves["who_played"].map({
            "Model A": model_a,
            "Model B": model_b,
            model_a: model_a,
            model_b: model_b,
        })
        
        model_a_moves = all_moves[all_moves["model_name"] == model_a]
        model_b_moves = all_moves[all_moves["model_name"] == model_b]
        
        model_a_profile = self._compute_pressure_profile(model_a, model_a_moves)
        model_b_profile = self._compute_pressure_profile(model_b, model_b_moves)
        
        # Generate insights
        insights = self._generate_insights(model_a_profile, model_b_profile)
        
        return MatchAnalysis(
            match_id=match_id,
            model_a_profile=model_a_profile,
            model_b_profile=model_b_profile,
            insights=insights,
        )
    
    def analyze_model_aggregate(self, model_name: str, match_ids: list[str]) -> Optional[ModelPressureProfile]:
        """Aggregate pressure profile across multiple matches for a model."""
        all_moves = []
        
        for match_id in match_ids:
            match_dir = self.results_dir / match_id
            if not match_dir.exists():
                continue
            
            metadata = self._load_metadata(match_dir)
            if not metadata:
                continue
            
            # Check if this model participated
            if model_name not in [metadata.get("model_a"), metadata.get("model_b")]:
                continue
            
            moves = self._load_all_moves(match_dir)
            if moves.empty:
                continue
            
            # Filter to this model's moves
            model_moves = moves[moves["who_played"] == model_name]
            
            # Add move analysis if available
            analysis = self._load_move_analysis(match_dir)
            if analysis is not None and not analysis.empty:
                model_moves = self._merge_with_analysis(model_moves, analysis)
            
            all_moves.append(model_moves)
        
        if not all_moves:
            return None
        
        combined = pd.concat(all_moves, ignore_index=True)
        return self._compute_pressure_profile(model_name, combined)
    
    def get_pressure_scatter_data(self, match_id: str) -> Optional[dict]:
        """Get data for time remaining vs move time scatter plot."""
        match_dir = self.results_dir / match_id
        if not match_dir.exists():
            return None
        
        metadata = self._load_metadata(match_dir)
        all_moves = self._load_all_moves(match_dir)
        
        if all_moves.empty:
            return None
        
        model_a = metadata.get("model_a", "Model A")
        model_b = metadata.get("model_b", "Model B")
        
        data = {
            "model_a": model_a,
            "model_b": model_b,
            "points": [],
        }
        
        # Map "Model A" / "Model B" in CSV to actual model names
        all_moves["model_name"] = all_moves["who_played"].map({
            "Model A": model_a,
            "Model B": model_b,
            model_a: model_a,
            model_b: model_b,
        })
        
        for _, row in all_moves.iterrows():
            data["points"].append({
                "model": row["model_name"],
                "time_remaining": row["time_available_at_turn_start"],
                "move_time": row["time_taken_seconds"],
                "game_number": row.get("game_number", 1),
                "move_number": row["move_number"],
                "thinking_tokens": row.get("thinking_tokens"),
            })
        
        return data
    
    def get_thinking_by_pressure(self, match_id: str) -> Optional[dict]:
        """Get thinking tokens grouped by pressure level."""
        match_dir = self.results_dir / match_id
        if not match_dir.exists():
            return None
        
        metadata = self._load_metadata(match_dir)
        all_moves = self._load_all_moves(match_dir)
        
        if all_moves.empty:
            return None
        
        model_a = metadata.get("model_a", "Model A")
        model_b = metadata.get("model_b", "Model B")
        
        # Map "Model A" / "Model B" in CSV to actual model names
        all_moves["model_name"] = all_moves["who_played"].map({
            "Model A": model_a,
            "Model B": model_b,
            model_a: model_a,
            model_b: model_b,
        })
        
        # Add pressure category
        all_moves["pressure"] = all_moves["time_available_at_turn_start"].apply(categorize_pressure)
        
        result = {"model_a": model_a, "model_b": model_b, "data": []}
        
        for pressure in ["comfortable", "medium", "high", "critical"]:
            pressure_moves = all_moves[all_moves["pressure"] == pressure]
            
            model_a_moves = pressure_moves[pressure_moves["model_name"] == model_a]
            model_b_moves = pressure_moves[pressure_moves["model_name"] == model_b]
            
            result["data"].append({
                "pressure": pressure,
                "model_a_avg_tokens": model_a_moves["thinking_tokens"].mean() if not model_a_moves.empty else 0,
                "model_b_avg_tokens": model_b_moves["thinking_tokens"].mean() if not model_b_moves.empty else 0,
                "model_a_avg_time": model_a_moves["time_taken_seconds"].mean() if not model_a_moves.empty else 0,
                "model_b_avg_time": model_b_moves["time_taken_seconds"].mean() if not model_b_moves.empty else 0,
                "model_a_count": len(model_a_moves),
                "model_b_count": len(model_b_moves),
            })
        
        return result
    
    def _load_metadata(self, match_dir: Path) -> Optional[dict]:
        """Load match metadata."""
        import json
        metadata_file = match_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file) as f:
            return json.load(f)
    
    def _load_all_moves(self, match_dir: Path) -> pd.DataFrame:
        """Load all moves from all games in a match."""
        move_files = list(match_dir.glob("game_*_moves.csv"))
        if not move_files:
            return pd.DataFrame()
        
        dfs = []
        for f in move_files:
            try:
                df = pd.read_csv(f)
                # Extract game number from filename
                game_num = int(f.stem.split("_")[1])
                df["game_number"] = game_num
                dfs.append(df)
            except Exception:
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    def _load_move_analysis(self, match_dir: Path) -> Optional[pd.DataFrame]:
        """Load Stockfish move analysis if available."""
        analysis_file = match_dir / "complete_move_analysis.csv"
        if not analysis_file.exists():
            return None
        
        try:
            return pd.read_csv(analysis_file)
        except Exception:
            return None
    
    def _merge_with_analysis(self, moves_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """Merge move data with Stockfish analysis."""
        # Merge on game_number and move_number
        if "game_number" in moves_df.columns and "game_number" in analysis_df.columns:
            merged = moves_df.merge(
                analysis_df[["game_number", "move_number", "centipawn_loss"]],
                on=["game_number", "move_number"],
                how="left"
            )
            return merged
        return moves_df
    
    def _compute_pressure_profile(self, model_name: str, moves_df: pd.DataFrame) -> ModelPressureProfile:
        """Compute pressure profile from move data."""
        if moves_df.empty:
            return ModelPressureProfile(
                model_name=model_name,
                total_moves=0,
                pressure_stats=[],
                speed_adaptation_ratio=1.0,
                quality_degradation_ratio=1.0,
                thinking_reduction_ratio=1.0,
            )
        
        # Add pressure category
        moves_df = moves_df.copy()
        moves_df["pressure"] = moves_df["time_available_at_turn_start"].apply(categorize_pressure)
        
        # Compute stats for each pressure level
        pressure_stats = []
        for level in ["comfortable", "medium", "high", "critical"]:
            level_moves = moves_df[moves_df["pressure"] == level]
            
            if level_moves.empty:
                continue
            
            has_cpl = "centipawn_loss" in level_moves.columns and level_moves["centipawn_loss"].notna().any()
            has_tokens = "thinking_tokens" in level_moves.columns and level_moves["thinking_tokens"].notna().any()
            
            stats = PressureStats(
                pressure_level=level,
                move_count=len(level_moves),
                avg_move_time=level_moves["time_taken_seconds"].mean(),
                std_move_time=level_moves["time_taken_seconds"].std(),
                avg_thinking_tokens=level_moves["thinking_tokens"].mean() if has_tokens else None,
                avg_centipawn_loss=level_moves["centipawn_loss"].mean() if has_cpl else None,
                blunder_rate=(level_moves["centipawn_loss"] > 100).mean() if has_cpl else 0.0,
            )
            pressure_stats.append(stats)
        
        # Compute adaptation ratios
        comfortable = moves_df[moves_df["pressure"] == "comfortable"]
        pressured = moves_df[moves_df["pressure"].isin(["high", "critical"])]
        
        speed_ratio = 1.0
        if not comfortable.empty and not pressured.empty:
            comfortable_time = comfortable["time_taken_seconds"].mean()
            pressured_time = pressured["time_taken_seconds"].mean()
            if comfortable_time > 0:
                speed_ratio = pressured_time / comfortable_time
        
        quality_ratio = 1.0
        if "centipawn_loss" in moves_df.columns:
            if not comfortable.empty and not pressured.empty:
                comfortable_cpl = comfortable["centipawn_loss"].mean()
                pressured_cpl = pressured["centipawn_loss"].mean()
                if comfortable_cpl > 0 and not pd.isna(comfortable_cpl) and not pd.isna(pressured_cpl):
                    quality_ratio = pressured_cpl / comfortable_cpl
        
        thinking_ratio = 1.0
        if "thinking_tokens" in moves_df.columns:
            if not comfortable.empty and not pressured.empty:
                comfortable_tokens = comfortable["thinking_tokens"].mean()
                pressured_tokens = pressured["thinking_tokens"].mean()
                if comfortable_tokens > 0 and not pd.isna(comfortable_tokens) and not pd.isna(pressured_tokens):
                    thinking_ratio = pressured_tokens / comfortable_tokens
        
        return ModelPressureProfile(
            model_name=model_name,
            total_moves=len(moves_df),
            pressure_stats=pressure_stats,
            speed_adaptation_ratio=speed_ratio,
            quality_degradation_ratio=quality_ratio,
            thinking_reduction_ratio=thinking_ratio,
        )
    
    def _generate_insights(
        self,
        profile_a: ModelPressureProfile,
        profile_b: ModelPressureProfile
    ) -> list[str]:
        """Generate human-readable insights from profiles."""
        insights = []
        
        # Speed adaptation
        if profile_a.speed_adaptation_ratio < 0.7:
            insights.append(
                f"{profile_a.model_name} significantly sped up under pressure "
                f"({(1 - profile_a.speed_adaptation_ratio) * 100:.0f}% faster)"
            )
        elif profile_a.speed_adaptation_ratio > 1.1:
            insights.append(
                f"{profile_a.model_name} surprisingly slowed down under pressure"
            )
        
        if profile_b.speed_adaptation_ratio < 0.7:
            insights.append(
                f"{profile_b.model_name} significantly sped up under pressure "
                f"({(1 - profile_b.speed_adaptation_ratio) * 100:.0f}% faster)"
            )
        elif profile_b.speed_adaptation_ratio > 1.1:
            insights.append(
                f"{profile_b.model_name} surprisingly slowed down under pressure"
            )
        
        # Quality comparison
        if profile_a.quality_degradation_ratio > 2.0:
            insights.append(
                f"{profile_a.model_name}'s move quality dropped significantly under pressure "
                f"({profile_a.quality_degradation_ratio:.1f}x worse)"
            )
        
        if profile_b.quality_degradation_ratio > 2.0:
            insights.append(
                f"{profile_b.model_name}'s move quality dropped significantly under pressure "
                f"({profile_b.quality_degradation_ratio:.1f}x worse)"
            )
        
        # Thinking reduction
        if profile_a.thinking_reduction_ratio < 0.5:
            insights.append(
                f"{profile_a.model_name} reduced thinking by "
                f"{(1 - profile_a.thinking_reduction_ratio) * 100:.0f}% under pressure"
            )
        
        if profile_b.thinking_reduction_ratio < 0.5:
            insights.append(
                f"{profile_b.model_name} reduced thinking by "
                f"{(1 - profile_b.thinking_reduction_ratio) * 100:.0f}% under pressure"
            )
        
        # Comparison insights
        if abs(profile_a.speed_adaptation_ratio - profile_b.speed_adaptation_ratio) > 0.3:
            faster_model = profile_a.model_name if profile_a.speed_adaptation_ratio < profile_b.speed_adaptation_ratio else profile_b.model_name
            insights.append(f"{faster_model} adapted to time pressure more quickly")
        
        if not insights:
            insights.append("Both models showed similar time pressure behavior")
        
        return insights

