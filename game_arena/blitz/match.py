#!/usr/bin/env python3
"""Configurable blitz chess match between AI models with time management.

This is the main entry point for running blitz chess matches.
"""

import datetime
import time
from typing import Optional

from absl import app
import termcolor

from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import prompts
from game_arena.harness import rethink
from game_arena.harness import tournament_util

# Import from new modular structure
from game_arena.blitz.core import GameState, GameStats
from game_arena.blitz.models import (
    NoRetryModelWrapper,
    calibrate_network_latency,
    handle_rethinking_move,
)
from game_arena.blitz.models.registry import get_model_from_registry
from game_arena.blitz.prompts import create_time_aware_prompt_substitutions
from game_arena.blitz.data import get_data_collector, create_analysis_notebook
from game_arena.blitz.display import (
    format_time,
    print_detailed_game_analysis,
    print_comprehensive_match_analysis,
    display_reasoning_traces,
)
from game_arena.blitz.core.parsing import handle_simple_parsing
from game_arena.blitz.analysis.stockfish import MoveQualityAnalyzer

# Import flags
from game_arena.blitz import flags as game_flags

colored = termcolor.colored


def run_automatic_move_analysis(match_id: str, collector) -> bool:
    """Run Stockfish move quality analysis on the completed match."""
    if not game_flags._RUN_MOVE_ANALYSIS.value:
        print(colored("🔬 Move quality analysis disabled (use --run_move_analysis=true to enable)", "yellow"))
        return False
    
    print(colored("\n🔬 Running automatic move quality analysis...", "cyan"))
    
    try:
        match_dir = collector.data_dir / match_id
        
        if not match_dir.exists():
            print(colored(f"❌ Match directory not found: {match_dir}", "red"))
            return False
        
        move_files = list(match_dir.glob("game_*_moves.csv"))
        if not move_files:
            print(colored(f"❌ No game move files found in {match_dir}", "red"))
            return False
        
        print(f"📁 Analyzing {len(move_files)} games with Stockfish...")
        print(f"⚙️  Analysis parameters: depth={game_flags._MOVE_ANALYSIS_DEPTH.value}, multipv={game_flags._MOVE_ANALYSIS_MULTIPV.value}")
        
        analyzer = MoveQualityAnalyzer(
            default_depth=game_flags._MOVE_ANALYSIS_DEPTH.value,
            default_multipv=game_flags._MOVE_ANALYSIS_MULTIPV.value
        )
        
        print(f"🏃 Using Stockfish at: {analyzer.engine_path}")
        
        results = analyzer.analyze_match_directory(
            match_dir,
            depth=game_flags._MOVE_ANALYSIS_DEPTH.value,
            multipv=game_flags._MOVE_ANALYSIS_MULTIPV.value,
            save_results=True
        )
        
        total_moves = sum(len(analyses) for analyses in results.values())
        print(colored(f"✅ Move quality analysis complete! Analyzed {total_moves} moves", "green"))
        print(colored(f"📊 Analysis saved to: {match_dir}/complete_move_analysis.csv", "green"))
        
        return True
        
    except FileNotFoundError as e:
        if "stockfish" in str(e).lower():
            print(colored("❌ Stockfish engine not found", "red"))
            print(colored("💡 To install Stockfish:", "yellow"))
            print(colored("   macOS: brew install stockfish", "yellow"))
            print(colored("   Ubuntu: sudo apt install stockfish", "yellow"))
        else:
            print(colored(f"❌ File error during move analysis: {e}", "red"))
        return False
        
    except Exception as e:
        print(colored(f"❌ Move quality analysis failed: {e}", "red"))
        return False


def setup_models_and_strategy():
    """Set up models and parsing/rethinking strategy."""
    from game_arena.harness import llm_parsers
    
    print(f"Setting up Model A: {game_flags._MODEL_A.value}")
    model_a = get_model_from_registry(game_flags._MODEL_A.value)
    
    print(f"Setting up Model B: {game_flags._MODEL_B.value}")
    model_b = get_model_from_registry(game_flags._MODEL_B.value)
    
    # Apply per-model reasoning budgets
    def apply_reasoning_budget(model, budget: int):
        """Apply reasoning budget to a model's options."""
        if not hasattr(model, '_model_options') or model._model_options is None:
            model._model_options = {}
        
        # For Anthropic models with thinking config
        if 'thinking' in model._model_options and isinstance(model._model_options['thinking'], dict):
            if model._model_options['thinking'].get('type') == 'enabled':
                model._model_options['thinking']['budget_tokens'] = budget
                print(f"  → Set Anthropic thinking budget: {budget}")
        else:
            # For Gemini and other models using thinking_budget
            model._model_options['thinking_budget'] = budget
            print(f"  → Set thinking budget: {budget}")
    
    # Determine per-model budgets (use per-model if set, else fall back to global)
    budget_a = game_flags._REASONING_BUDGET_A.value if game_flags._REASONING_BUDGET_A.value > 0 else game_flags._REASONING_BUDGET.value
    budget_b = game_flags._REASONING_BUDGET_B.value if game_flags._REASONING_BUDGET_B.value > 0 else game_flags._REASONING_BUDGET.value
    
    print(f"Reasoning budgets: Model A = {budget_a}, Model B = {budget_b}")
    apply_reasoning_budget(model_a, budget_a)
    apply_reasoning_budget(model_b, budget_b)
    
    # Set up parsers
    match game_flags._PARSER_CHOICE.value:
        case tournament_util.ParserChoice.RULE_THEN_SOFT:
            move_parser = parsers.RuleBasedMoveParser()
            legality_parser = parsers.SoftMoveParser("chess")
        case tournament_util.ParserChoice.LLM_ONLY:
            parser_model = get_model_from_registry("gemini-2.5-flash")
            move_parser = llm_parsers.LLMParser(
                model=parser_model,
                instruction_config=llm_parsers.OpenSpielChessInstructionConfig_V0,
            )
            legality_parser = parsers.SoftMoveParser("chess")
        case _:
            raise ValueError(f"Unsupported parser choice: {game_flags._PARSER_CHOICE.value}")
    
    # Create rethink samplers if enabled
    if game_flags._USE_RETHINKING.value:
        prompt_generator = prompt_generation.PromptGeneratorText()
        
        model_a_sampler = rethink.RethinkSampler(
            model=model_a,
            strategy=game_flags._RETHINK_STRATEGY.value,
            num_max_rethinks=game_flags._MAX_RETHINKS.value,
            move_parser=move_parser,
            legality_parser=legality_parser,
            game_short_name="chess",
            prompt_generator=prompt_generator,
            rethink_template=None,
        )
        
        model_b_sampler = rethink.RethinkSampler(
            model=model_b,
            strategy=game_flags._RETHINK_STRATEGY.value,
            num_max_rethinks=game_flags._MAX_RETHINKS.value,
            move_parser=move_parser,
            legality_parser=legality_parser,
            game_short_name="chess",
            prompt_generator=prompt_generator,
            rethink_template=None,
        )
        
        return model_a, model_b, model_a_sampler, model_b_sampler
    else:
        parser = parsers.ChainedMoveParser([move_parser, legality_parser])
        return model_a, model_b, parser, parser


def play_single_blitz_game(
    white_model_wrapper: NoRetryModelWrapper, 
    black_model_wrapper: NoRetryModelWrapper, 
    move_strategy,
    model_a_plays_white: bool,
    game_number: int,
    white_latency: float,
    black_latency: float,
    use_rethinking: bool = False
) -> GameStats:
    """Play a single blitz game."""
    strategy_label = "WITH RETHINKING" if use_rethinking else ""
    print(colored(f"\n=== BLITZ GAME {game_number} {strategy_label} ===", "cyan"))
    
    game_state = GameState(
        game_number=game_number,
        model_a_plays_white=model_a_plays_white,
        initial_time=game_flags._INITIAL_TIME_SECONDS.value
    )
    
    print(colored(f"{game_state.white_name} (White) vs {game_state.black_name} (Black)", "cyan"))
    
    model_a_name = game_flags._MODEL_A.value
    model_b_name = game_flags._MODEL_B.value
    if model_a_plays_white:
        white_model, black_model = white_model_wrapper, black_model_wrapper
    else:
        white_model, black_model = black_model_wrapper, white_model_wrapper
    
    print(colored(f"⏰ Starting time: {format_time(game_flags._INITIAL_TIME_SECONDS.value)} each", "blue"))
    print(colored(f"⏰ Increment: +{game_flags._INCREMENT_SECONDS.value}s per move", "blue"))
    
    if use_rethinking:
        white_sampler, black_sampler = move_strategy
        print(colored(f"🧠 Rethinking enabled: {game_flags._RETHINK_STRATEGY.value.value} (max {game_flags._MAX_RETHINKS.value} attempts)", "blue"))
    else:
        parser = move_strategy
    
    prompt_generator = prompt_generation.PromptGeneratorText()
    prompt_template = prompts.PromptTemplate.NO_LEGAL_ACTIONS
    
    # Main game loop
    while not game_state.pyspiel_state.is_terminal() and game_state.move_count < game_flags._MAX_MOVES_PER_GAME.value:
        player_info = game_state.get_current_player_info(white_model, black_model, white_latency, black_latency)
        
        time_forfeit_result = game_state.check_time_forfeit()
        if time_forfeit_result:
            return time_forfeit_result
        
        print(f"\nMove {game_state.move_count + 1}: {player_info['player_name']}'s turn")
        print(colored(f"⏰ {player_info['player_name']}: {format_time(player_info['player_clock'].time_remaining)} | "
                     f"Opponent: {format_time(player_info['opponent_clock'].time_remaining)}", "yellow"))
        
        board_state_before = game_state.pyspiel_state.to_string()
        time_at_turn_start = player_info['player_clock'].time_remaining
        
        player_info['player_clock'].start_move()
        
        prompt_substitutions = create_time_aware_prompt_substitutions(
            game_state.pyspiel_state, 
            player_info['player_clock'], 
            player_info['opponent_clock'], 
            game_flags._INCREMENT_SECONDS.value,
            is_blitz=True
        )
        
        if use_rethinking:
            current_sampler = white_sampler if player_info['is_white'] else black_sampler
            
            move_notation, should_continue, game_end_result, retry_info = handle_rethinking_move(
                game_state, player_info, current_sampler, prompt_substitutions,
                game_flags._MAX_PARSING_FAILURES.value, game_flags._MAX_RETHINKS.value
            )
            
            if game_end_result:
                return game_end_result
            
            if should_continue:
                continue
            
            thinking_time = player_info['player_clock'].end_move(player_info['network_latency'], game_flags._INCREMENT_SECONDS.value)
            print(colored(f"⏰ Thinking time: {thinking_time:.2f}s", "blue"))
            
            if game_flags._SHOW_REASONING_TRACES.value:
                response_obj = retry_info.get('response')
                display_reasoning_traces(response_obj, retry_info.get('generate_returns'))
            
            if move_notation:
                success = game_state.apply_move(
                    move_notation, player_info, player_info['network_latency'],
                    game_flags._INCREMENT_SECONDS.value, retry_info.get('response'),
                    retry_info.get('retry_count', 0), retry_info.get('total_retry_time', 0),
                    thinking_time, board_state_before, time_at_turn_start
                )
                if not success:
                    game_state.increment_parsing_failures(player_info['is_white'])
                    continue
        else:
            prompt = prompt_generator.generate_prompt_with_text_only(
                prompt_template=prompt_template,
                game_short_name="chess",
                **prompt_substitutions,
            )
            
            try:
                response, retry_count, total_retry_time = player_info['model'].generate_with_text_input(prompt)
                thinking_time = player_info['player_clock'].end_move(player_info['network_latency'], game_flags._INCREMENT_SECONDS.value)
                
                print(f"{player_info['player_name']} response: {response.main_response[:100]}...")
                print(colored(f"⏰ Thinking time: {thinking_time:.2f}s", "blue"))
                
                if game_flags._SHOW_REASONING_TRACES.value:
                    display_reasoning_traces(response)
            
            except Exception as e:
                thinking_time = player_info['player_clock'].end_move(player_info['network_latency'], game_flags._INCREMENT_SECONDS.value)
                print(colored(f"Error calling {player_info['player_name']} model: {e}", "red"))
                return game_state._create_game_stats("error", "*")
            
            move_notation, should_continue, game_end_result = handle_simple_parsing(
                game_state, player_info, response, parser, game_flags._MAX_PARSING_FAILURES.value
            )
            
            if game_end_result:
                return game_end_result
            
            if should_continue:
                continue
            
            if move_notation:
                print(f"Parsed move: {move_notation}")
                success = game_state.apply_move(
                    move_notation, player_info, player_info['network_latency'],
                    game_flags._INCREMENT_SECONDS.value, response, retry_count, total_retry_time,
                    thinking_time, board_state_before, time_at_turn_start
                )
                if not success:
                    game_state.increment_parsing_failures(player_info['is_white'])
                    continue
    
    return game_state.calculate_final_result()


def main(_) -> None:
    first_to = game_flags._FIRST_TO.value
    max_games = first_to * 2 - 1
    
    print(colored(f"=== BLITZ CHESS MATCH (FIRST TO {first_to}) ===", "magenta"))
    print(colored(f"⏰ Time Control: {game_flags._INITIAL_TIME_SECONDS.value}s + {game_flags._INCREMENT_SECONDS.value}s increment", "blue"))
    print(colored(f"🧠 Rethinking: {game_flags._USE_RETHINKING.value}", "blue"))
    print(colored(f"🤖 Model A: {game_flags._MODEL_A.value}", "blue"))
    print(colored(f"🤖 Model B: {game_flags._MODEL_B.value}", "blue"))
    
    start_time = time.time()
    
    # Initialize data collection
    collector = get_data_collector()
    match_id = collector.start_match(
        model_a=game_flags._MODEL_A.value,
        model_b=game_flags._MODEL_B.value,
        time_control_seconds=game_flags._INITIAL_TIME_SECONDS.value,
        increment_seconds=game_flags._INCREMENT_SECONDS.value,
        rethinking_enabled=game_flags._USE_RETHINKING.value,
        max_parsing_failures=game_flags._MAX_PARSING_FAILURES.value,
        max_rethinks=game_flags._MAX_RETHINKS.value,
        reasoning_budget=game_flags._REASONING_BUDGET.value,
        parser_choice=str(game_flags._PARSER_CHOICE.value)
    )
    
    print(colored(f"📊 Data collection started - Match ID: {match_id}", "green"))
    
    model_a_name = game_flags._MODEL_A.value
    model_b_name = game_flags._MODEL_B.value
    
    # Set up models and strategy
    model_a, model_b, strategy_arg1, strategy_arg2 = setup_models_and_strategy()
    
    model_a_wrapper = NoRetryModelWrapper(model_a)
    model_b_wrapper = NoRetryModelWrapper(model_b)
    
    if game_flags._USE_RETHINKING.value:
        strategy_arg1._model = model_a_wrapper
        strategy_arg2._model = model_b_wrapper
        move_strategy = (strategy_arg1, strategy_arg2)
        use_rethinking = True
    else:
        move_strategy = strategy_arg1
        use_rethinking = False
    
    # Calibrate network latency
    print(colored("\n🌐 Calibrating network latencies...", "yellow"))
    model_a_latency = calibrate_network_latency(model_a_wrapper, game_flags._CALIBRATION_ROUNDS.value)
    model_b_latency = calibrate_network_latency(model_b_wrapper, game_flags._CALIBRATION_ROUNDS.value)
    
    # Match tracking
    model_a_wins = 0
    model_b_wins = 0
    draws = 0
    games_played = 0
    all_game_stats = []
    
    # Play games
    while model_a_wins < first_to and model_b_wins < first_to and games_played < max_games:
        games_played += 1
        model_a_plays_white = (games_played % 2 == 1)
        
        if model_a_plays_white:
            white_model, black_model = model_a_wrapper, model_b_wrapper
            white_latency, black_latency = model_a_latency, model_b_latency
        else:
            white_model, black_model = model_b_wrapper, model_a_wrapper
            white_latency, black_latency = model_b_latency, model_a_latency
        
        game_stats = play_single_blitz_game(
            white_model, black_model, move_strategy,
            model_a_plays_white, games_played,
            white_latency, black_latency,
            use_rethinking
        )
        
        collector.record_game(
            game_stats, 
            initial_time=game_flags._INITIAL_TIME_SECONDS.value,
            increment=game_flags._INCREMENT_SECONDS.value
        )
        
        if game_stats.winner == "model_a":
            model_a_wins += 1
        elif game_stats.winner == "model_b":
            model_b_wins += 1
        else:
            draws += 1
        
        all_game_stats.append(game_stats)
        print_detailed_game_analysis(game_stats)
        
        print(colored(f"\nSCORE AFTER GAME {games_played}:", "magenta"))
        print(colored(f"{model_a_name}: {model_a_wins} wins", "blue"))
        print(colored(f"{model_b_name}: {model_b_wins} wins", "blue"))
        print(colored(f"Draws: {draws}", "yellow"))
        
        if model_a_wins == first_to:
            print(colored(f"\n🎉 MATCH WINNER: {model_a_name.upper()}! ({model_a_wins}-{model_b_wins})", "green"))
            break
        elif model_b_wins == first_to:
            print(colored(f"\n🎉 MATCH WINNER: {model_b_name.upper()}! ({model_b_wins}-{model_a_wins})", "green"))
            break
    
    # End data collection
    final_scores = {"model_a": model_a_wins, "model_b": model_b_wins, "draws": draws}
    collector.end_match(final_scores)
    
    # Run move analysis
    run_automatic_move_analysis(match_id, collector)
    
    # Create analysis notebook
    create_analysis_notebook(match_id)
    
    # Final summary
    end_time = time.time()
    duration = datetime.timedelta(seconds=end_time - start_time)
    
    print(colored("\n" + "="*70, "magenta"))
    print(colored(f"FINAL BLITZ MATCH RESULTS", "magenta"))
    print(colored("="*70, "magenta"))
    print(f"Games played: {games_played}")
    print(f"{model_a_name} wins: {model_a_wins}")
    print(f"{model_b_name} wins: {model_b_wins}")
    print(f"Draws: {draws}")
    print(f"Match duration: {duration}")
    
    print_comprehensive_match_analysis(all_game_stats)
    
    print(colored(f"\n📊 Match data saved to: _results/{match_id}/", "green"))


if __name__ == "__main__":
    app.run(main)

