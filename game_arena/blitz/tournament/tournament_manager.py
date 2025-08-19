#!/usr/bin/env python3
"""Tournament manager for single elimination blitz chess brackets."""

import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import dataclasses
from dataclasses import asdict
import pandas as pd

from game_arena.blitz import utils
from game_arena.blitz import data_collector 
from game_arena.blitz import blitz_match


@dataclasses.dataclass
class TournamentMetadata:
    """Metadata for an entire tournament."""
    tournament_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    tournament_type: str = "single_elimination"  # single_elimination, double_elimination, round_robin
    models: List[str] = dataclasses.field(default_factory=list)
    time_control: str = ""  # e.g., "300+3"
    rethinking_enabled: bool = False
    max_parsing_failures: int = 3
    max_rethinks: int = 2
    reasoning_budget: int = 8000
    parser_choice: str = ""
    winner: str = ""
    total_matches: int = 0
    total_games: int = 0
    tournament_duration_seconds: float = 0.0


@dataclasses.dataclass 
class MatchResult:
    """Result of a single best-of-7 match in the tournament."""
    match_id: str
    round_name: str  # "quarterfinals", "semifinals", "finals"
    match_number: int
    model1: str
    model2: str
    winner: str
    loser: str
    score_model1: int
    score_model2: int
    games_played: int
    match_duration_seconds: float
    data_directory: str


@dataclasses.dataclass
class TournamentBracket:
    """Represents the tournament bracket structure."""
    quarterfinals: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    semifinals: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    finals: Tuple[str, str] = None
    
    # Results
    quarterfinal_results: List[MatchResult] = dataclasses.field(default_factory=list)
    semifinal_results: List[MatchResult] = dataclasses.field(default_factory=list)
    final_result: Optional[MatchResult] = None


class BlitzTournamentManager:
    """Manages single elimination tournament with 8 models."""
    
    def __init__(self, models: List[str], data_dir: str = "tournament_data"):
        if len(models) != 8:
            raise ValueError(f"Tournament requires exactly 8 models, got {len(models)}")
        
        self.models = models
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.tournament_metadata: Optional[TournamentMetadata] = None
        self.bracket: Optional[TournamentBracket] = None
        self.match_results: List[MatchResult] = []
        
    def start_tournament(self, 
                        time_control_seconds: int,
                        increment_seconds: int,
                        rethinking_enabled: bool = False,
                        max_parsing_failures: int = 3,
                        max_rethinks: int = 2,
                        reasoning_budget: int = 8000,
                        parser_choice: str = "rule_then_soft") -> str:
        """Start a new tournament and return the tournament ID."""
        timestamp = datetime.datetime.now()
        tournament_id = f"blitz_tournament_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        self.tournament_id = tournament_id
        self.tournament_metadata = TournamentMetadata(
            tournament_id=tournament_id,
            start_time=timestamp,
            models=self.models.copy(),
            time_control=f"{time_control_seconds}+{increment_seconds}",
            rethinking_enabled=rethinking_enabled,
            max_parsing_failures=max_parsing_failures,
            max_rethinks=max_rethinks,
            reasoning_budget=reasoning_budget,
            parser_choice=parser_choice
        )
        
        # Create tournament directory
        tournament_dir = self.data_dir / tournament_id
        tournament_dir.mkdir(exist_ok=True)
        
        # Initialize bracket
        self.bracket = self._create_bracket()
        
        # Save initial tournament data
        self._save_tournament_metadata()
        
        print(f"🏆 Tournament started: {tournament_id}")
        print(f"📁 Tournament data directory: {tournament_dir}")
        
        return tournament_id
    
    def _create_bracket(self) -> TournamentBracket:
        """Create the tournament bracket with random seeding."""
        import random
        
        # Shuffle models for random seeding
        shuffled_models = self.models.copy()
        random.shuffle(shuffled_models)
        
        # Create quarterfinal matchups
        quarterfinals = [
            (shuffled_models[0], shuffled_models[1]),
            (shuffled_models[2], shuffled_models[3]),
            (shuffled_models[4], shuffled_models[5]),
            (shuffled_models[6], shuffled_models[7])
        ]
        
        bracket = TournamentBracket(quarterfinals=quarterfinals)
        
        print("🗂️ Tournament Bracket Created:")
        print("=" * 50)
        print("QUARTERFINALS:")
        for i, (model1, model2) in enumerate(quarterfinals, 1):
            print(f"  Match {i}: {model1} vs {model2}")
        print("=" * 50)
        
        return bracket
    
    def run_tournament(self, **match_flags) -> Dict[str, Any]:
        """Run the complete tournament."""
        if not self.tournament_metadata:
            raise ValueError("Tournament not started. Call start_tournament() first.")
        
        print(f"\n🚀 Starting Tournament: {self.tournament_id}")
        print(f"🎯 Format: Single Elimination (Best of 7)")
        print(f"⏰ Time Control: {self.tournament_metadata.time_control}")
        
        total_matches = 0
        total_games = 0
        
        # Run Quarterfinals
        print(f"\n{'='*20} QUARTERFINALS {'='*20}")
        semifinal_qualifiers = []
        
        for i, (model1, model2) in enumerate(self.bracket.quarterfinals, 1):
            print(f"\n🥊 Quarterfinal Match {i}: {model1} vs {model2}")
            
            result = self._run_best_of_7_match(
                model1, model2, f"quarterfinals", i, **match_flags
            )
            
            self.bracket.quarterfinal_results.append(result)
            self.match_results.append(result)
            semifinal_qualifiers.append(result.winner)
            total_matches += 1
            total_games += result.games_played
            
            print(f"✅ {result.winner} defeats {result.loser} ({result.score_model1}-{result.score_model2})")
        
        # Set up semifinals
        self.bracket.semifinals = [
            (semifinal_qualifiers[0], semifinal_qualifiers[1]),
            (semifinal_qualifiers[2], semifinal_qualifiers[3])
        ]
        
        # Run Semifinals
        print(f"\n{'='*20} SEMIFINALS {'='*20}")
        final_qualifiers = []
        
        for i, (model1, model2) in enumerate(self.bracket.semifinals, 1):
            print(f"\n🥊 Semifinal Match {i}: {model1} vs {model2}")
            
            result = self._run_best_of_7_match(
                model1, model2, f"semifinals", i, **match_flags
            )
            
            self.bracket.semifinal_results.append(result)
            self.match_results.append(result)
            final_qualifiers.append(result.winner)
            total_matches += 1
            total_games += result.games_played
            
            print(f"✅ {result.winner} defeats {result.loser} ({result.score_model1}-{result.score_model2})")
        
        # Set up finals
        self.bracket.finals = (final_qualifiers[0], final_qualifiers[1])
        
        # Run Finals
        print(f"\n{'='*20} FINALS {'='*20}")
        model1, model2 = self.bracket.finals
        print(f"\n🏆 FINAL MATCH: {model1} vs {model2}")
        
        result = self._run_best_of_7_match(
            model1, model2, "finals", 1, **match_flags
        )
        
        self.bracket.final_result = result
        self.match_results.append(result)
        total_matches += 1
        total_games += result.games_played
        
        # Update tournament metadata
        self.tournament_metadata.end_time = datetime.datetime.now()
        self.tournament_metadata.winner = result.winner
        self.tournament_metadata.total_matches = total_matches
        self.tournament_metadata.total_games = total_games
        self.tournament_metadata.tournament_duration_seconds = (
            self.tournament_metadata.end_time - self.tournament_metadata.start_time
        ).total_seconds()
        
        # Save final tournament data
        self._save_all_tournament_data()
        
        print(f"\n{'='*20} TOURNAMENT COMPLETE {'='*20}")
        print(f"🏆 CHAMPION: {result.winner}")
        print(f"🥈 Runner-up: {result.loser}")
        print(f"📊 Total matches: {total_matches}")
        print(f"📊 Total games: {total_games}")
        print(f"⏱️ Tournament duration: {self.tournament_metadata.tournament_duration_seconds/60:.1f} minutes")
        
        return {
            "tournament_id": self.tournament_id,
            "winner": result.winner,
            "runner_up": result.loser,
            "total_matches": total_matches,
            "total_games": total_games,
            "duration_minutes": self.tournament_metadata.tournament_duration_seconds / 60
        }
    
    def _run_best_of_7_match(self, model1: str, model2: str, round_name: str, 
                           match_number: int, **match_flags) -> MatchResult:
        """Run a single best-of-7 match between two models."""
        # This would integrate with the existing blitz_match.py
        # For now, we'll simulate the interface
        
        match_start = datetime.datetime.now()
        
        # TODO: Actually call the blitz match function with proper model setup
        # This is a placeholder that shows the intended interface
        
        # Simulate match result for now
        import random
        games_played = random.randint(4, 7)  # Best of 7, so 4-7 games
        if random.random() < 0.5:
            winner, loser = model1, model2
            score1, score2 = 4, games_played - 4
        else:
            winner, loser = model2, model1
            score1, score2 = games_played - 4, 4
            score1, score2 = score2, score1  # Swap to match model order
        
        match_end = datetime.datetime.now()
        duration = (match_end - match_start).total_seconds()
        
        # Create a dummy match_id and data directory
        match_id = f"{self.tournament_id}_{round_name}_match_{match_number}"
        data_dir = str(self.data_dir / self.tournament_id / f"{round_name}_match_{match_number}")
        
        return MatchResult(
            match_id=match_id,
            round_name=round_name,
            match_number=match_number,
            model1=model1,
            model2=model2,
            winner=winner,
            loser=loser,
            score_model1=score1,
            score_model2=score2,
            games_played=games_played,
            match_duration_seconds=duration,
            data_directory=data_dir
        )
    
    def _save_tournament_metadata(self):
        """Save tournament metadata to disk."""
        if not self.tournament_metadata:
            return
        
        tournament_dir = self.data_dir / self.tournament_id
        tournament_dir.mkdir(exist_ok=True)
        
        metadata_dict = asdict(self.tournament_metadata)
        # Convert datetime objects to strings
        for key, value in metadata_dict.items():
            if isinstance(value, datetime.datetime):
                metadata_dict[key] = value.isoformat() if value else None
        
        with open(tournament_dir / "tournament_metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _save_all_tournament_data(self):
        """Save all tournament data including results and bracket."""
        self._save_tournament_metadata()
        
        tournament_dir = self.data_dir / self.tournament_id
        
        # Save bracket structure
        bracket_dict = asdict(self.bracket)
        with open(tournament_dir / "bracket.json", 'w') as f:
            json.dump(bracket_dict, f, indent=2)
        
        # Save match results as CSV
        if self.match_results:
            results_df = pd.DataFrame([asdict(result) for result in self.match_results])
            results_df.to_csv(tournament_dir / "match_results.csv", index=False)
        
        # Create tournament analysis notebook
        self._create_tournament_analysis_notebook()
        
        print(f"📊 Tournament data saved to: {tournament_dir}")
    
    def _create_tournament_analysis_notebook(self):
        """Create a comprehensive analysis notebook for the tournament."""
        from game_arena.blitz.tournament.analysis import tournament_analysis
        tournament_analysis.create_tournament_analysis_notebook(
            self.tournament_id, str(self.data_dir)
        )


def load_tournament_data(tournament_id: str, data_dir: str = "tournament_data") -> Dict[str, Any]:
    """Load all data for a completed tournament."""
    tournament_dir = Path(data_dir) / tournament_id
    
    if not tournament_dir.exists():
        raise FileNotFoundError(f"Tournament data not found: {tournament_dir}")
    
    # Load metadata
    with open(tournament_dir / "tournament_metadata.json") as f:
        metadata = json.load(f)
    
    # Load bracket
    with open(tournament_dir / "bracket.json") as f:
        bracket = json.load(f)
    
    # Load match results
    match_results_df = pd.read_csv(tournament_dir / "match_results.csv")
    
    return {
        "metadata": metadata,
        "bracket": bracket,
        "match_results": match_results_df,
        "data_directory": str(tournament_dir)
    } 