#!/usr/bin/env python3
"""Main script to run blitz chess tournaments with 8 models."""

import datetime
import time
from typing import List
from pathlib import Path

from absl import app
from absl import flags
import termcolor

from game_arena.blitz.tournament import tournament_manager
from game_arena.blitz.tournament.analysis import tournament_analysis
from game_arena.blitz import blitz_match
from game_arena.blitz import flags as game_flags
from game_arena.blitz.tournament import tournament_integration  # Import the integration layer

colored = termcolor.colored

# Tournament-specific flags
flags.DEFINE_list("models", 
                 ["gemini-2.5-flash", "gpt-4.1", "claude-3.5-sonnet", "llama-3.1-70b",
                  "gemini-2.5-pro", "gpt-4.0", "claude-3-opus", "mistral-large"], 
                 "List of 8 models to compete in the tournament")

flags.DEFINE_string("tournament_name", "", 
                   "Optional custom name for the tournament (auto-generated if empty)")

flags.DEFINE_bool("create_visualizations", True,
                 "Whether to create visualization plots after the tournament")

FLAGS = flags.FLAGS


def validate_tournament_setup():
    """Validate that tournament setup is correct."""
    if len(FLAGS.models) != 8:
        raise ValueError(f"Tournament requires exactly 8 models, got {len(FLAGS.models)}")
    
    # Check for duplicate models
    if len(set(FLAGS.models)) != len(FLAGS.models):
        raise ValueError("Duplicate models found in tournament list")
    
    print(colored("✅ Tournament setup validated", "green"))


def run_complete_tournament(models: List[str]) -> dict:
    """Run a complete single elimination tournament."""
    
    # Initialize tournament manager
    tournament_mgr = tournament_manager.BlitzTournamentManager(models)
    
    # Start tournament with current flag settings
    tournament_id = tournament_mgr.start_tournament(
        time_control_seconds=game_flags._INITIAL_TIME_SECONDS.value,
        increment_seconds=game_flags._INCREMENT_SECONDS.value,
        rethinking_enabled=game_flags._USE_RETHINKING.value,
        max_parsing_failures=game_flags._MAX_PARSING_FAILURES.value,
        max_rethinks=game_flags._MAX_RETHINKS.value,
        reasoning_budget=game_flags._REASONING_BUDGET.value,
        parser_choice=str(game_flags._PARSER_CHOICE.value)
    )
    
    print(colored(f"🏆 Tournament Started: {tournament_id}", "magenta"))
    print(colored(f"⚔️ Models: {', '.join(models)}", "blue"))
    print(colored(f"⏰ Time Control: {game_flags._INITIAL_TIME_SECONDS.value}s + {game_flags._INCREMENT_SECONDS.value}s", "blue"))
    print(colored(f"🧠 Rethinking: {game_flags._USE_RETHINKING.value}", "blue"))
    
    # Run the tournament using the integrated tournament manager
    # The integration layer has patched the tournament manager to use real matches
    tournament_results = tournament_mgr.run_tournament(
        time_control_seconds=game_flags._INITIAL_TIME_SECONDS.value,
        increment_seconds=game_flags._INCREMENT_SECONDS.value,
        rethinking_enabled=game_flags._USE_RETHINKING.value,
        max_parsing_failures=game_flags._MAX_PARSING_FAILURES.value,
        max_rethinks=game_flags._MAX_RETHINKS.value,
        reasoning_budget=game_flags._REASONING_BUDGET.value,
        parser_choice=str(game_flags._PARSER_CHOICE.value),
        calibration_rounds=game_flags._CALIBRATION_ROUNDS.value
    )
    
    tournament_results["tournament_id"] = tournament_id
    return tournament_results


def main(_):
    """Main tournament execution function."""
    print(colored("🏁 BLITZ CHESS TOURNAMENT SYSTEM", "magenta"))
    print(colored("=" * 50, "magenta"))
    
    # Validate setup
    validate_tournament_setup()
    
    # Print tournament configuration
    print(colored("\n📋 TOURNAMENT CONFIGURATION:", "cyan"))
    print(f"Models: {FLAGS.models}")
    print(f"Time Control: {game_flags._INITIAL_TIME_SECONDS.value}s + {game_flags._INCREMENT_SECONDS.value}s")
    print(f"Rethinking: {game_flags._USE_RETHINKING.value}")
    print(f"Max Parsing Failures: {game_flags._MAX_PARSING_FAILURES.value}")
    print(f"Parser: {game_flags._PARSER_CHOICE.value}")
    
    start_time = time.time()
    
    # Run the tournament
    try:
        results = run_complete_tournament(FLAGS.models)
        
        # Calculate total duration
        total_duration = time.time() - start_time
        duration_str = str(datetime.timedelta(seconds=total_duration))
        
        print(colored(f"\n⏱️  Total Tournament Duration: {duration_str}", "blue"))
        print(colored(f"📊 Total Matches: {results.get('total_matches', 0)}", "blue"))
        
        # Generate analysis
        tournament_id = results["tournament_id"]
        
        if FLAGS.create_visualizations:
            print(colored("\n📊 Generating tournament analysis...", "yellow"))
            
            # Create analysis notebook
            tournament_analysis.create_tournament_analysis_notebook(tournament_id)
            
            # Generate text report
            report = tournament_analysis.generate_tournament_report(tournament_id)
            
            # Save report
            report_path = Path("tournament_data") / tournament_id / "tournament_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(colored(f"📄 Tournament report saved: {report_path}", "green"))
            print(colored(f"📓 Analysis notebook created for detailed insights", "green"))
        
        print(colored(f"\n🎉 Tournament {tournament_id} completed successfully!", "green"))
        
        # Print final summary
        print(colored("\n" + "="*60, "magenta"))
        print(colored("TOURNAMENT ANALYSIS SUMMARY", "magenta"))
        print(colored("="*60, "magenta"))
        print(f"🏆 Champion: {results.get('winner', 'Unknown')}")
        print(f"🥈 Runner-up: {results.get('runner_up', 'Unknown')}")
        print(f"📊 Total matches played: {results.get('total_matches', 0)}")
        print(f"📊 Total games played: {results.get('total_games', 0)}")
        print(f"⏱️  Tournament duration: {results.get('duration_minutes', 0):.1f} minutes")
        
        print(colored(f"\n📁 All data saved in: tournament_data/{tournament_id}/", "green"))
        print(colored("Data includes:", "green"))
        print(colored("  • Tournament bracket and results", "green"))
        print(colored("  • Move-by-move statistics for every game", "green"))
        print(colored("  • Time pressure behavior analysis", "green"))
        print(colored("  • Token usage and efficiency metrics", "green"))
        print(colored("  • Tactical time usage pattern detection", "green"))
        print(colored("  • Interactive Jupyter notebooks for analysis", "green"))
        
    except Exception as e:
        print(colored(f"❌ Tournament failed: {e}", "red"))
        raise


if __name__ == "__main__":
    app.run(main) 