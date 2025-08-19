#!/usr/bin/env python3
"""Integration layer between tournament system and blitz match execution."""

import datetime
import time
from typing import Dict, Any, Tuple
from pathlib import Path
import shutil
import tempfile

from game_arena.blitz import blitz_match
from game_arena.blitz.tournament import tournament_manager
from game_arena.blitz import verification
from game_arena.blitz import utils
from game_arena.blitz import flags as game_flags
from game_arena.blitz import data_collector


def run_tournament_match_integration(model1_name: str, model2_name: str, 
                                   round_name: str, match_number: int,
                                   tournament_id: str,
                                   **match_settings) -> tournament_manager.MatchResult:
    """
    Run a single best-of-7 match between two models with full integration.
    
    This function properly integrates with the existing blitz_match system
    and collects all the necessary data for tournament analysis.
    """
    
    print(f"\n🎯 Starting {round_name} Match {match_number}: {model1_name} vs {model2_name}")
    
    match_start = datetime.datetime.now()
    
    # Create match-specific data directory within tournament
    tournament_dir = Path("tournament_data") / tournament_id
    match_dir = tournament_dir / f"{round_name}_match_{match_number}"
    match_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporarily redirect the global data collector to our match directory
    original_data_dir = data_collector._data_collector.data_dir if data_collector._data_collector else None
    
    try:
        # Set up models (you'll need to implement proper model initialization here)
        # For now, we'll use the existing model setup from blitz_match
        
        # This is where you'd map model names to actual model instances
        # For demonstration, we'll use placeholder logic
        gemini_model, openai_model, strategy_arg1, strategy_arg2 = setup_models_for_tournament_match(
            model1_name, model2_name, **match_settings
        )
        
        # Set up data collection for this specific match
        collector = data_collector.BlitzDataCollector(str(match_dir))
        
        # Start the match data collection
        match_id = collector.start_match(
            gemini_model=model1_name if "gemini" in model1_name.lower() else model2_name,
            openai_model=model2_name if "gemini" in model1_name.lower() else model1_name,
            time_control_seconds=match_settings.get('time_control_seconds', game_flags._INITIAL_TIME_SECONDS.value),
            increment_seconds=match_settings.get('increment_seconds', game_flags._INCREMENT_SECONDS.value),
            rethinking_enabled=match_settings.get('rethinking_enabled', game_flags._USE_RETHINKING.value),
            max_parsing_failures=match_settings.get('max_parsing_failures', game_flags._MAX_PARSING_FAILURES.value),
            max_rethinks=match_settings.get('max_rethinks', game_flags._MAX_RETHINKS.value),
            reasoning_budget=match_settings.get('reasoning_budget', game_flags._REASONING_BUDGET.value),
            parser_choice=str(match_settings.get('parser_choice', game_flags._PARSER_CHOICE.value))
        )
        
        # Run the actual best-of-7 match
        match_result = run_best_of_7_with_data_collection(
            model1_name, model2_name, 
            gemini_model, openai_model,
            strategy_arg1, strategy_arg2,
            collector, **match_settings
        )
        
        # Finalize data collection
        collector.end_match({
            "gemini": match_result['score_model1'] if "gemini" in model1_name.lower() else match_result['score_model2'],
            "openai": match_result['score_model2'] if "gemini" in model1_name.lower() else match_result['score_model1'],
            "draws": 0  # Assuming no draws in best-of-7
        })
        
        # Create analysis notebook for this match
        data_collector.create_analysis_notebook(match_id, str(match_dir))
        
        match_end = datetime.datetime.now()
        duration = (match_end - match_start).total_seconds()
        
        return tournament_manager.MatchResult(
            match_id=match_id,
            round_name=round_name,
            match_number=match_number,
            model1=model1_name,
            model2=model2_name,
            winner=match_result['winner'],
            loser=match_result['loser'],
            score_model1=match_result['score_model1'],
            score_model2=match_result['score_model2'],
            games_played=match_result['games_played'],
            match_duration_seconds=duration,
            data_directory=str(match_dir)
        )
        
    except Exception as e:
        print(f"❌ Error in tournament match: {e}")
        # Return a failure result
        return tournament_manager.MatchResult(
            match_id=f"failed_{round_name}_{match_number}",
            round_name=round_name,
            match_number=match_number,
            model1=model1_name,
            model2=model2_name,
            winner="error",
            loser="error",
            score_model1=0,
            score_model2=0,
            games_played=0,
            match_duration_seconds=0,
            data_directory=str(match_dir)
        )
    
    finally:
        # Restore original data collector settings
        if original_data_dir and data_collector._data_collector:
            data_collector._data_collector.data_dir = original_data_dir


def setup_models_for_tournament_match(model1_name: str, model2_name: str, **settings) -> Tuple[Any, Any, Any, Any]:
    """
    Set up models for a tournament match.
    
    This function would need to be implemented to properly initialize
    the specific models based on their names. For now, it's a placeholder
    that uses the existing verification setup.
    """
    
    # TODO: Implement proper model mapping and initialization
    # This is where you'd map model names like "claude-3.5-sonnet" to actual model instances
    
    # For now, use the existing setup (which uses gemini and openai)
    # In a real implementation, you'd:
    # 1. Parse the model names
    # 2. Initialize the appropriate model clients
    # 3. Set up the correct samplers/parsers
    
    # Placeholder using existing verification setup
    return verification.setup_models_and_rethink_samplers(game_flags)


def run_best_of_7_with_data_collection(model1_name: str, model2_name: str,
                                      gemini_model, openai_model,
                                      strategy_arg1, strategy_arg2,
                                      collector: data_collector.BlitzDataCollector,
                                      **settings) -> Dict[str, Any]:
    """
    Run a best-of-7 match with proper data collection.
    
    This integrates with the existing blitz_match logic
    while ensuring all data is properly collected for tournament analysis.
    """
    
    # Set up model wrappers
    gemini_model_wrapper = utils.NoRetryModelWrapper(gemini_model)
    openai_model_wrapper = utils.NoRetryModelWrapper(openai_model)
    
    # Determine which model is which
    if "gemini" in model1_name.lower():
        white_model, black_model = gemini_model_wrapper, openai_model_wrapper
        model1_is_gemini = True
    else:
        white_model, black_model = openai_model_wrapper, gemini_model_wrapper
        model1_is_gemini = False
    
    # Set up strategy
    if settings.get('rethinking_enabled', False):
        strategy_arg1._model = gemini_model_wrapper
        strategy_arg2._model = openai_model_wrapper
        move_strategy = (strategy_arg1, strategy_arg2)
        use_rethinking = True
    else:
        move_strategy = strategy_arg1  # parser
        use_rethinking = False
    
    # Calibrate network latencies
    gemini_latency = utils.calibrate_network_latency(
        gemini_model_wrapper, 
        settings.get('calibration_rounds', 3)
    )
    openai_latency = utils.calibrate_network_latency(
        openai_model_wrapper, 
        settings.get('calibration_rounds', 3)
    )
    
    # Match tracking
    model1_wins = 0
    model2_wins = 0
    games_played = 0
    all_game_stats = []
    
    # Play games until one player wins 4
    while model1_wins < 4 and model2_wins < 4 and games_played < 7:
        games_played += 1
        
        # Alternate who plays white
        gemini_plays_white = (games_played % 2 == 1)
        
        # Set up model assignments
        if gemini_plays_white:
            white_model_actual, black_model_actual = gemini_model_wrapper, openai_model_wrapper
            white_latency, black_latency = gemini_latency, openai_latency
        else:
            white_model_actual, black_model_actual = openai_model_wrapper, gemini_model_wrapper
            white_latency, black_latency = openai_latency, gemini_latency
        
        # Run single game using existing logic
        game_stats = blitz_match.play_single_blitz_game(
            white_model_actual, black_model_actual, move_strategy,
            gemini_plays_white, games_played,
            white_latency, black_latency, use_rethinking
        )
        
        # Record game data
        collector.record_game(
            game_stats,
            initial_time=settings.get('time_control_seconds', game_flags._INITIAL_TIME_SECONDS.value),
            increment=settings.get('increment_seconds', game_flags._INCREMENT_SECONDS.value)
        )
        
        # Update scores based on which model won
        if game_stats.winner == "gemini":
            if model1_is_gemini:
                model1_wins += 1
            else:
                model2_wins += 1
        elif game_stats.winner == "openai":
            if not model1_is_gemini:
                model1_wins += 1
            else:
                model2_wins += 1
        
        all_game_stats.append(game_stats)
        
        print(f"Game {games_played}: {game_stats.winner} wins")
        print(f"Current score: {model1_name} {model1_wins} - {model2_wins} {model2_name}")
    
    # Determine overall winner
    if model1_wins > model2_wins:
        winner, loser = model1_name, model2_name
    else:
        winner, loser = model2_name, model1_name
    
    return {
        'winner': winner,
        'loser': loser,
        'score_model1': model1_wins,
        'score_model2': model2_wins,
        'games_played': games_played,
        'all_game_stats': all_game_stats
    }


# Monkey patch the tournament manager to use our integration
def patch_tournament_manager():
    """Patch the tournament manager to use our integration layer."""
    
    def patched_run_best_of_7_match(self, model1: str, model2: str, round_name: str, 
                                   match_number: int, **match_flags) -> tournament_manager.MatchResult:
        """Patched version that uses the integration layer."""
        return run_tournament_match_integration(
            model1, model2, round_name, match_number, 
            self.tournament_id, **match_flags
        )
    
    # Replace the method
    tournament_manager.BlitzTournamentManager._run_best_of_7_match = patched_run_best_of_7_match


# Apply the patch when this module is imported
patch_tournament_manager() 