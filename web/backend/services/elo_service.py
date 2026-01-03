#!/usr/bin/env python3
"""ELO calculation service for model rankings."""

from datetime import datetime
from typing import Optional

from models.schemas import ModelStats, LeaderboardResponse


class EloService:
    """Service for calculating and tracking ELO ratings."""
    
    K_FACTOR = 32  # Standard K-factor
    INITIAL_ELO = 1500
    
    def __init__(self):
        self.models: dict[str, ModelStats] = {}
    
    def get_or_create_model(self, model_id: str) -> ModelStats:
        """Get existing model stats or create new entry."""
        if model_id not in self.models:
            self.models[model_id] = ModelStats(
                model_id=model_id,
                display_name=self._format_display_name(model_id),
                elo=self.INITIAL_ELO,
                games_played=0,
                wins=0,
                losses=0,
                draws=0,
                win_rate=0.0,
                elo_change=0,
            )
        return self.models[model_id]
    
    def calculate_elo_change(
        self,
        winner_elo: int,
        loser_elo: int,
        is_draw: bool = False
    ) -> tuple[int, int]:
        """Calculate ELO changes for a match result.
        
        Returns:
            Tuple of (winner_change, loser_change)
        """
        expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
        expected_loser = 1 - expected_winner
        
        if is_draw:
            # Draw - both players get 0.5 score
            winner_change = round(self.K_FACTOR * (0.5 - expected_winner))
            loser_change = round(self.K_FACTOR * (0.5 - expected_loser))
        else:
            # Win/loss
            winner_change = round(self.K_FACTOR * (1 - expected_winner))
            loser_change = round(self.K_FACTOR * (0 - expected_loser))
        
        return winner_change, loser_change
    
    def record_match_result(
        self,
        model_a: str,
        model_b: str,
        model_a_wins: int,
        model_b_wins: int,
        draws: int,
    ) -> None:
        """Record a match result and update ELO ratings."""
        stats_a = self.get_or_create_model(model_a)
        stats_b = self.get_or_create_model(model_b)
        
        total_games = model_a_wins + model_b_wins + draws
        
        # Update game counts
        stats_a.games_played += total_games
        stats_b.games_played += total_games
        stats_a.wins += model_a_wins
        stats_a.losses += model_b_wins
        stats_a.draws += draws
        stats_b.wins += model_b_wins
        stats_b.losses += model_a_wins
        stats_b.draws += draws
        
        # Calculate ELO changes for each game
        total_a_change = 0
        total_b_change = 0
        
        for _ in range(model_a_wins):
            change_a, change_b = self.calculate_elo_change(stats_a.elo, stats_b.elo)
            total_a_change += change_a
            total_b_change += change_b
        
        for _ in range(model_b_wins):
            change_b, change_a = self.calculate_elo_change(stats_b.elo, stats_a.elo)
            total_a_change += change_a
            total_b_change += change_b
        
        for _ in range(draws):
            change_a, change_b = self.calculate_elo_change(stats_a.elo, stats_b.elo, is_draw=True)
            total_a_change += change_a
            total_b_change += change_b
        
        # Apply ELO changes
        stats_a.elo += total_a_change
        stats_b.elo += total_b_change
        stats_a.elo_change = total_a_change
        stats_b.elo_change = total_b_change
        
        # Update win rates
        if stats_a.games_played > 0:
            stats_a.win_rate = stats_a.wins / stats_a.games_played
        if stats_b.games_played > 0:
            stats_b.win_rate = stats_b.wins / stats_b.games_played
    
    def get_leaderboard(self) -> LeaderboardResponse:
        """Get current leaderboard sorted by ELO."""
        sorted_models = sorted(
            self.models.values(),
            key=lambda m: m.elo,
            reverse=True
        )
        
        return LeaderboardResponse(
            models=sorted_models,
            last_updated=datetime.now(),
        )
    
    def rebuild_from_matches(self, matches: list[dict]) -> None:
        """Rebuild ELO ratings from historical match data."""
        # Reset all models
        self.models = {}
        
        # Sort matches by start time
        sorted_matches = sorted(
            matches,
            key=lambda m: m.get("start_time", "")
        )
        
        # Process each match
        for match in sorted_matches:
            if not match.get("end_time"):
                continue  # Skip incomplete matches
            
            self.record_match_result(
                model_a=match.get("model_a", ""),
                model_b=match.get("model_b", ""),
                model_a_wins=match.get("model_a_wins", 0),
                model_b_wins=match.get("model_b_wins", 0),
                draws=match.get("draws", 0),
            )
    
    def _format_display_name(self, model_id: str) -> str:
        """Format model ID into display name."""
        # Simple formatting - capitalize and replace hyphens
        return model_id.replace("-", " ").title()

