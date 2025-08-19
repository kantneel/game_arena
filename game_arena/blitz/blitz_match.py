#!/usr/bin/env python3
"""Configurable blitz chess match between AI models with time management."""

import datetime
import time
from typing import Dict, List, Tuple, Optional
import dataclasses
import logging

from absl import app
from absl import flags
from game_arena.harness import game_notation_examples
from game_arena.harness import llm_parsers
from game_arena.harness import model_generation
from game_arena.harness import model_generation_sdk
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import prompts
from game_arena.harness import rethink
from game_arena.harness import samplers
from game_arena.harness import tournament_util
import termcolor
import pyspiel
import tenacity

# Import our modules
from game_arena.blitz import utils
from game_arena.blitz import flags as game_flags
from game_arena.blitz import verification
from game_arena.blitz import game_engine
from game_arena.blitz import data_collector
from game_arena.blitz.move_analysis import move_analyzer

colored = termcolor.colored


def run_automatic_move_analysis(match_id: str, collector: data_collector.BlitzDataCollector) -> bool:
    """
    Run Stockfish move quality analysis on the completed match.
    
    Args:
        match_id: The match identifier
        collector: Data collector instance to get match directory
        
    Returns:
        True if analysis completed successfully, False otherwise
    """
    if not game_flags._RUN_MOVE_ANALYSIS.value:
        print(colored("🔬 Move quality analysis disabled (use --run_move_analysis=true to enable)", "yellow"))
        return False
    
    print(colored("\n🔬 Running automatic move quality analysis...", "cyan"))
    
    try:
        # Get match directory from collector
        match_dir = collector.data_dir / match_id
        
        if not match_dir.exists():
            print(colored(f"❌ Match directory not found: {match_dir}", "red"))
            return False
        
        # Check for game move files
        move_files = list(match_dir.glob("game_*_moves.csv"))
        if not move_files:
            print(colored(f"❌ No game move files found in {match_dir}", "red"))
            return False
        
        print(f"📁 Analyzing {len(move_files)} games with Stockfish...")
        print(f"⚙️  Analysis parameters: depth={game_flags._MOVE_ANALYSIS_DEPTH.value}, multipv={game_flags._MOVE_ANALYSIS_MULTIPV.value}")
        
        # Initialize move analyzer
        analyzer = move_analyzer.MoveQualityAnalyzer(
            default_depth=game_flags._MOVE_ANALYSIS_DEPTH.value,
            default_multipv=game_flags._MOVE_ANALYSIS_MULTIPV.value
        )
        
        print(f"🏃 Using Stockfish at: {analyzer.engine_path}")
        
        # Run analysis
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
            print(colored("   Or download from: https://stockfishchess.org/", "yellow"))
        else:
            print(colored(f"❌ File error during move analysis: {e}", "red"))
        return False
        
    except Exception as e:
        print(colored(f"❌ Move quality analysis failed: {e}", "red"))
        print(colored("💡 Analysis will be skipped, but match data is still available", "yellow"))
        return False


def display_reasoning_traces(response_obj, generate_returns=None):
    """Display reasoning traces from a response object if enabled and available."""
    if not game_flags._SHOW_REASONING_TRACES.value:
        return
    
    # Handle rethinking case with multiple responses
    if generate_returns:
        for i, gen_return in enumerate(generate_returns):
            if hasattr(gen_return, 'main_response_and_thoughts'):
                prefix = f"🧠 Attempt {i+1} - " if len(generate_returns) > 1 else "🧠 "
                _display_single_reasoning_trace(gen_return, prefix)
        return
    
    # Handle single response case
    if not response_obj or not hasattr(response_obj, 'main_response_and_thoughts'):
        return
    
    _display_single_reasoning_trace(response_obj, "🧠 ")


def _display_single_reasoning_trace(response_obj, prefix):
    """Display reasoning trace for a single response object."""
    full_text = response_obj.main_response_and_thoughts
    main_resp = getattr(response_obj, 'main_response', '')
    # Extract reasoning part (everything after main response)
    if full_text and len(full_text) > len(main_resp):
        reasoning_only = full_text[len(main_resp):].strip()
        if reasoning_only.startswith('\n\n'):
            reasoning_only = reasoning_only[2:]
        
        if reasoning_only:  # Only show if there's actual reasoning content
            words = reasoning_only.split()
            if len(words) > 100:  # If more than 100 words total
                first_50 = ' '.join(words[:50])
                last_50 = ' '.join(words[-50:])
                print(colored(f"{prefix}Reasoning (first 50 words): {first_50}...", "magenta"))
                print(colored(f"{prefix}Reasoning (last 50 words): ...{last_50}", "magenta"))
            else:
                print(colored(f"{prefix}Reasoning: {reasoning_only}", "magenta"))
    else:
        # Fallback: show first/last 50 words of full response if no separate reasoning
        words = full_text.split()
        if len(words) > 100:
            first_50 = ' '.join(words[:50])
            last_50 = ' '.join(words[-50:])
            print(colored(f"{prefix}Response (first 50 words): {first_50}...", "magenta"))
            print(colored(f"{prefix}Response (last 50 words): ...{last_50}", "magenta"))
        elif len(words) > 0:
            print(colored(f"{prefix}Response: {full_text}", "magenta"))


class BlitzModelWrapper:
    """Wrapper for models used with rethink samplers to track retry info."""
    
    def __init__(self, wrapped_model):
        self.wrapped_model = wrapped_model
        self.total_retry_time = 0
        self.retry_count = 0
    
    def generate_with_text_input(self, model_input):
        response, retry_count, retry_time = self.wrapped_model.generate_with_text_input(model_input)
        self.retry_count += retry_count
        self.total_retry_time += retry_time
        return response


def handle_rethinking_move(game_state: game_engine.GameState, player_info: dict, 
                          sampler, prompt_substitutions: dict, max_failures: int, 
                          max_rethinks: int) -> Tuple[Optional[str], bool, Optional[utils.GameStats], dict]:
    """
    Handle move generation with rethinking.
    Returns: (move_notation, should_continue, game_stats_if_ended, retry_info)
    """
    # Create wrapper to track retry info
    blitz_model = BlitzModelWrapper(player_info['model'])
    
    # Temporarily replace the sampler's model
    original_model = sampler._model
    sampler._model = blitz_model
    
    try:
        # Use the rethink sampler
        sampler_output = sampler.sample_action_with_text_and_state_input(
            state=game_state.pyspiel_state,
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            **prompt_substitutions
        )
        
        # Restore original model
        sampler._model = original_model
        
        print(f"{player_info['player_name']} rethink result: {sampler_output.move_type.value}")
        if sampler_output.auxiliary_outputs:
            num_attempts = len([k for k in sampler_output.auxiliary_outputs.keys() if k.startswith("parsed_action_attempt_")])
            print(colored(f"🧠 Rethinking attempts: {num_attempts}", "blue"))
        
        # Check if we got a legal move
        if sampler_output.move_type == tournament_util.MoveType.LEGAL and sampler_output.action:
            print(f"Final move: {sampler_output.action}")
            
            # Count total parsing failures from auxiliary outputs
            current_failures = sum(1 for k, v in sampler_output.auxiliary_outputs.items() 
                                 if k.startswith("maybe_legal_action_attempt_") and v is None)
            
            game_state.increment_parsing_failures(player_info['is_white'], current_failures)
            
            # Return successful move with aggregated response info
            # Combine all responses for full text (including thoughts)
            full_responses = []
            for i, gr in enumerate(sampler_output.generate_returns):
                if hasattr(gr, 'main_response_and_thoughts') and gr.main_response_and_thoughts:
                    full_responses.append(f"Attempt {i+1}: {gr.main_response_and_thoughts}")
                elif hasattr(gr, 'main_response') and gr.main_response:
                    full_responses.append(f"Attempt {i+1}: {gr.main_response}")
            
            combined_response_text = "\n\n".join(full_responses) if full_responses else ""
            
            aggregated_response = type('obj', (object,), {
                'reasoning_tokens': sum(gr.reasoning_tokens or 0 for gr in sampler_output.generate_returns),
                'generation_tokens': sum(gr.generation_tokens or 0 for gr in sampler_output.generate_returns),
                'prompt_tokens': sum(gr.prompt_tokens or 0 for gr in sampler_output.generate_returns),
                'main_response_and_thoughts': combined_response_text,
                'main_response': sampler_output.action  # The final move that was chosen
            })
            
            retry_info = {
                'retry_count': blitz_model.retry_count,
                'total_retry_time': blitz_model.total_retry_time,
                'response': aggregated_response,
                'generate_returns': sampler_output.generate_returns  # Add the actual responses for reasoning traces
            }
            
            return sampler_output.action, False, None, retry_info
            
        else:
            # Rethinking failed to produce a legal move
            print(colored(f"🚫 {player_info['player_name']} failed to produce legal move after rethinking", "red"))
            
            # Count all attempts as failures
            total_attempts = len([k for k in sampler_output.auxiliary_outputs.keys() if k.startswith("parsed_action_attempt_")])
            game_state.increment_parsing_failures(player_info['is_white'], total_attempts)
            current_total_failures = game_state.get_parsing_failures(player_info['is_white'])
            
            print(colored(f"⚠️  Total parsing failures for {player_info['player_name']}: {current_total_failures}/{max_failures}", "yellow"))
            
            failure_result = game_state.check_parsing_failure_limit(player_info['is_white'], max_failures)
            if failure_result:
                return None, False, failure_result, {}
            
            print(colored(f"🔄 Skipping {player_info['player_name']}'s turn, continuing game...", "yellow"))
            return None, True, None, {}
            
    except Exception as e:
        # Restore original model
        sampler._model = original_model
        
        print(colored(f"Error during rethinking for {player_info['player_name']}: {e}", "red"))
        
        # Treat as max failures
        game_state.increment_parsing_failures(player_info['is_white'], max_rethinks + 1)
        
        # Return error result
        game_duration = time.time() - game_state.game_start_time
        winner = "error"
        error_stats = utils.GameStats(
            game_number=game_state.game_number,
            winner=winner,
            result_string="*",
            model_a_color="white" if game_state.model_a_plays_white else "black",
            total_moves=game_state.move_count,
            duration=game_duration,
            move_stats=game_state.move_stats,
            model_a_final_time=game_state.white_clock.time_remaining if game_state.model_a_plays_white else game_state.black_clock.time_remaining,
            model_b_final_time=game_state.black_clock.time_remaining if game_state.model_a_plays_white else game_state.white_clock.time_remaining,
            model_a_parsing_failures=game_state.white_parsing_failures if game_state.model_a_plays_white else game_state.black_parsing_failures,
            model_b_parsing_failures=game_state.black_parsing_failures if game_state.model_a_plays_white else game_state.white_parsing_failures
        )
        return None, False, error_stats, {}


def play_single_blitz_game(
    white_model_wrapper: utils.NoRetryModelWrapper, 
    black_model_wrapper: utils.NoRetryModelWrapper, 
    move_strategy,  # Either a parser or tuple of (white_sampler, black_sampler)
    model_a_plays_white: bool,
    game_number: int,
    white_latency: float,
    black_latency: float,
    use_rethinking: bool = False
) -> utils.GameStats:
    """
    Play a single blitz game with unified logic for both simple parsing and rethinking.
    """
    strategy_label = "WITH RETHINKING" if use_rethinking else ""
    print(colored(f"\n=== BLITZ GAME {game_number} {strategy_label} ===", "cyan"))
    
    # Initialize game state
    game_state = game_engine.GameState(
        game_number=game_number,
        model_a_plays_white=model_a_plays_white,
        initial_time=game_flags._INITIAL_TIME_SECONDS.value
    )
    
    print(colored(f"{game_state.white_name} (White) vs {game_state.black_name} (Black)", "cyan"))
    
    # Set up models based on color assignments
    model_a_name = game_flags._MODEL_A.value
    model_b_name = game_flags._MODEL_B.value
    if model_a_plays_white:
        white_model, black_model = white_model_wrapper, black_model_wrapper
        white_name, black_name = model_a_name, model_b_name
    else:
        white_model, black_model = black_model_wrapper, white_model_wrapper
        white_name, black_name = model_b_name, model_a_name
    
    print(colored(f"⏰ Starting time: {utils.format_time(game_flags._INITIAL_TIME_SECONDS.value)} each", "blue"))
    print(colored(f"⏰ Increment: +{game_flags._INCREMENT_SECONDS.value}s per move", "blue"))
    print(colored(f"🔧 Max parsing failures per player: {game_flags._MAX_PARSING_FAILURES.value}", "blue"))
    
    if use_rethinking:
        white_sampler, black_sampler = move_strategy
        print(colored(f"🧠 Rethinking enabled: {game_flags._RETHINK_STRATEGY.value.value} (max {game_flags._MAX_RETHINKS.value} attempts)", "blue"))
    else:
        parser = move_strategy
    
    # Set up prompt generator
    prompt_generator = prompt_generation.PromptGeneratorText()
    prompt_template = prompts.PromptTemplate.NO_LEGAL_ACTIONS
    
    # Main game loop
    while not game_state.pyspiel_state.is_terminal() and game_state.move_count < game_flags._MAX_MOVES_PER_GAME.value:
        # Get current player info
        player_info = game_state.get_current_player_info(white_model, black_model, white_latency, black_latency)
        
        # Check for time forfeit
        time_forfeit_result = game_state.check_time_forfeit()
        if time_forfeit_result:
            return time_forfeit_result
        
        print(f"\nMove {game_state.move_count + 1}: {player_info['player_name']}'s turn")
        print(colored(f"⏰ {player_info['player_name']}: {utils.format_time(player_info['player_clock'].time_remaining)} | "
                     f"Opponent: {utils.format_time(player_info['opponent_clock'].time_remaining)}", "yellow"))
        
        # Capture board state before the move and time at turn start
        board_state_before = game_state.pyspiel_state.to_string()
        time_at_turn_start = player_info['player_clock'].time_remaining
        
        # Start the move clock
        player_info['player_clock'].start_move()
        
        # Generate time-aware prompt
        prompt_substitutions = game_engine.create_time_aware_prompt_substitutions(
            game_state.pyspiel_state, 
            player_info['player_clock'], 
            player_info['opponent_clock'], 
            game_flags._INCREMENT_SECONDS.value,
            is_blitz=True
        )
        
        # Handle move generation based on strategy
        if use_rethinking:
            # Select appropriate sampler
            current_sampler = white_sampler if player_info['is_white'] else black_sampler
            
            move_notation, should_continue, game_end_result, retry_info = handle_rethinking_move(
                game_state, player_info, current_sampler, prompt_substitutions,
                game_flags._MAX_PARSING_FAILURES.value, game_flags._MAX_RETHINKS.value
            )
            
            if game_end_result:
                return game_end_result
            
            if should_continue:
                continue
            
            # Show timing info
            thinking_time = player_info['player_clock'].end_move(player_info['network_latency'], game_flags._INCREMENT_SECONDS.value)
            print(colored(f"⏰ Thinking time: {thinking_time:.2f}s", "blue"))
            if retry_info.get('retry_count', 0) > 0:
                print(colored(f"🔄 Network retries: {retry_info['retry_count']} attempts, {retry_info['total_retry_time']:.1f}s retry time (not counted)", "yellow"))
            
            # Show reasoning traces if enabled
            response_obj = retry_info.get('response')
            display_reasoning_traces(response_obj, retry_info.get('generate_returns'))
            
            # Apply the move with rethinking response info
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
            # Simple parsing approach
            prompt = prompt_generator.generate_prompt_with_text_only(
                prompt_template=prompt_template,
                game_short_name="chess",
                **prompt_substitutions,
            )
            
            # Call the model
            try:
                response, retry_count, total_retry_time = player_info['model'].generate_with_text_input(prompt)
                thinking_time = player_info['player_clock'].end_move(player_info['network_latency'], game_flags._INCREMENT_SECONDS.value)
                
                print(f"{player_info['player_name']} response: {response.main_response[:100]}...")
                print(colored(f"⏰ Thinking time: {thinking_time:.2f}s", "blue"))
                if retry_count > 0:
                    print(colored(f"🔄 Retries: {retry_count} attempts, {total_retry_time:.1f}s retry time (not counted)", "yellow"))
                
                # Show reasoning traces if enabled
                display_reasoning_traces(response)
            
            except Exception as e:
                thinking_time = player_info['player_clock'].end_move(player_info['network_latency'], game_flags._INCREMENT_SECONDS.value)
                print(colored(f"Error calling {player_info['player_name']} model: {e}", "red"))
                
                # Return error result
                return game_state._create_game_stats("error", "*")
            
            # Parse the move
            move_notation, should_continue, game_end_result = game_engine.handle_simple_parsing(
                game_state, player_info, response, parser, game_flags._MAX_PARSING_FAILURES.value
            )
            
            if game_end_result:
                return game_end_result
            
            if should_continue:
                continue
            
            # Apply the move
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
    
    # Calculate final result
    return game_state.calculate_final_result()


def main(_) -> None:
    if game_flags._VERIFICATION_ONLY.value:
        verification.verify_retry_wrapper_functionality(game_flags)
        return
        
    first_to = game_flags._FIRST_TO.value
    max_games = first_to * 2 - 1
    print(colored(f"=== BLITZ CHESS MATCH (FIRST TO {first_to}) ===", "magenta"))
    print(colored(f"⏰ Time Control: {game_flags._INITIAL_TIME_SECONDS.value}s + {game_flags._INCREMENT_SECONDS.value}s increment", "blue"))
    print(colored(f"🧠 Rethinking: {game_flags._USE_RETHINKING.value} ({game_flags._RETHINK_STRATEGY.value.value if game_flags._USE_RETHINKING.value else 'disabled'})", "blue"))
    if game_flags._USE_RETHINKING.value:
        print(colored(f"🧠 Max rethinks: {game_flags._MAX_RETHINKS.value}", "blue"))
        print(colored(f"🧠 Reasoning budget: {game_flags._REASONING_BUDGET.value} tokens", "blue"))
        print(colored(f"🧠 Show reasoning traces: {game_flags._SHOW_REASONING_TRACES.value}", "blue"))
    print(colored(f"🔧 Max parsing failures per player: {game_flags._MAX_PARSING_FAILURES.value}", "blue"))
    print(colored(f"🔬 Move quality analysis: {game_flags._RUN_MOVE_ANALYSIS.value}", "blue"))
    if game_flags._RUN_MOVE_ANALYSIS.value:
        print(colored(f"🔬 Analysis depth: {game_flags._MOVE_ANALYSIS_DEPTH.value}, multipv: {game_flags._MOVE_ANALYSIS_MULTIPV.value}", "blue"))
    print(colored(f"🤖 Competing Models:", "blue"))
    print(colored(f"   Model A: {game_flags._MODEL_A.value}", "blue"))
    print(colored(f"   Model B: {game_flags._MODEL_B.value}", "blue"))
    print(colored(f"Parser: {game_flags._PARSER_CHOICE.value}", "blue"))
    
    start_time = time.time()
    
    # Initialize data collection
    collector = data_collector.get_data_collector()
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
    
    # Store model names for consistent display
    model_a_name = game_flags._MODEL_A.value
    model_b_name = game_flags._MODEL_B.value
    print(colored(f"📋 Model mapping: A={model_a_name}, B={model_b_name}", "blue"))
    
    # Set up models and strategy
    model_a, model_b, strategy_arg1, strategy_arg2 = verification.setup_models_and_rethink_samplers(game_flags)
    
    # Wrap models with NoRetryModelWrapper
    model_a_wrapper = utils.NoRetryModelWrapper(model_a)
    model_b_wrapper = utils.NoRetryModelWrapper(model_b)
    
    if game_flags._USE_RETHINKING.value:
        # Update samplers to use wrapped models
        strategy_arg1._model = model_a_wrapper  # model_a_sampler
        strategy_arg2._model = model_b_wrapper  # model_b_sampler
        move_strategy = (strategy_arg1, strategy_arg2)  # (model_a_sampler, model_b_sampler)
        use_rethinking = True
    else:
        move_strategy = strategy_arg1  # parser
        use_rethinking = False
    
    # Calibrate network latency for both models
    print(colored("\n🌐 Calibrating network latencies...", "yellow"))
    model_a_latency = utils.calibrate_network_latency(model_a_wrapper, game_flags._CALIBRATION_ROUNDS.value)
    model_b_latency = utils.calibrate_network_latency(model_b_wrapper, game_flags._CALIBRATION_ROUNDS.value)
    
    # Match tracking
    model_a_wins = 0
    model_b_wins = 0
    draws = 0
    games_played = 0
    all_game_stats = []
    
    # Play games until one player reaches first_to wins
    while model_a_wins < first_to and model_b_wins < first_to and games_played < max_games:
        games_played += 1
        
        # Alternate who plays white ({model_a_name} starts as white in game 1)
        model_a_plays_white = (games_played % 2 == 1)
        
        # Set up model assignments (white, black based on model_a_plays_white)
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
        
        # Record game data for analysis
        collector.record_game(
            game_stats, 
            initial_time=game_flags._INITIAL_TIME_SECONDS.value,
            increment=game_flags._INCREMENT_SECONDS.value
        )
        
        # Update scores
        if game_stats.winner == "model_a":
            model_a_wins += 1
        elif game_stats.winner == "model_b":
            model_b_wins += 1
        elif game_stats.winner == "draw":
            draws += 1
        
        all_game_stats.append(game_stats)
        
        # Print detailed analysis
        utils.print_detailed_game_analysis(game_stats)
        
        # Print current score
        print(colored(f"\nSCORE AFTER GAME {games_played}:", "magenta"))
        print(colored(f"{model_a_name}: {model_a_wins} wins", "blue"))
        print(colored(f"{model_b_name}: {model_b_wins} wins", "blue"))
        print(colored(f"Draws: {draws}", "yellow"))
        
        # Check if match is decided
        if model_a_wins == first_to:
            print(colored(f"\n🎉 MATCH WINNER: {model_a_name.upper()}! ({model_a_wins}-{model_b_wins})", "green"))
            break
        elif model_b_wins == first_to:
            print(colored(f"\n🎉 MATCH WINNER: {model_b_name.upper()}! ({model_b_wins}-{model_a_wins})", "green"))
            break
    
    # End data collection
    final_scores = {
        "model_a": model_a_wins,
        "model_b": model_b_wins,
        "draws": draws
    }
    collector.end_match(final_scores)
    
    # Run automatic move quality analysis
    move_analysis_success = run_automatic_move_analysis(match_id, collector)
    
    # Create analysis notebook (with move analysis data if available)
    data_collector.create_analysis_notebook(match_id)
    
    # Final results and comprehensive analysis
    end_time = time.time()
    duration = datetime.timedelta(seconds=end_time - start_time)
    
    print(colored("\n" + "="*70, "magenta"))
    print(colored(f"FINAL BLITZ MATCH RESULTS & ANALYSIS (FIRST TO {first_to})", "magenta"))
    print(colored("="*70, "magenta"))
    print(f"Games played: {games_played}")
    print(f"{model_a_name} wins: {model_a_wins}")
    print(f"{model_b_name} wins: {model_b_wins}")
    print(f"Draws: {draws}")
    print(f"Match duration: {duration}")
    
    # Time management analysis
    print(colored("\n⏰ TIME MANAGEMENT ANALYSIS:", "cyan"))
    
    model_a_avg_final_time = sum(g.model_a_final_time for g in all_game_stats) / len(all_game_stats)
    model_b_avg_final_time = sum(g.model_b_final_time for g in all_game_stats) / len(all_game_stats)
    
    print(f"Average final time - {model_a_name}: {utils.format_time(model_a_avg_final_time)}")
    print(f"Average final time - {model_b_name}: {utils.format_time(model_b_avg_final_time)}")
    
    # Rethinking analysis if enabled
    if game_flags._USE_RETHINKING.value:
        print(colored("\n🧠 RETHINKING ANALYSIS:", "cyan"))
        total_model_a_failures = sum(g.model_a_parsing_failures for g in all_game_stats)
        total_model_b_failures = sum(g.model_b_parsing_failures for g in all_game_stats)
        print(f"Total parsing failures - {model_a_name}: {total_model_a_failures}")
        print(f"Total parsing failures - {model_b_name}: {total_model_b_failures}")
        print(f"Average failures per game - {model_a_name}: {total_model_a_failures/len(all_game_stats):.1f}")
        print(f"Average failures per game - {model_b_name}: {total_model_b_failures/len(all_game_stats):.1f}")
    
    # Comprehensive analysis
    utils.print_comprehensive_match_analysis(all_game_stats)
    
    print(colored("\nGAME BY GAME BREAKDOWN:", "cyan"))
    for game in all_game_stats:
        color_info = f"({model_a_name}: {game.model_a_color})"
        time_info = f"Times: {model_a_name}:{utils.format_time(game.model_a_final_time)} {model_b_name}:{utils.format_time(game.model_b_final_time)}"
        failure_info = f"Failures: {model_a_name}:{game.model_a_parsing_failures} {model_b_name}:{game.model_b_parsing_failures}" if game_flags._USE_RETHINKING.value else ""
        print(f"Game {game.game_number}: {game.winner} wins {color_info} - {game.result_string} "
              f"({game.total_moves} moves, {game.duration:.0f}s) {time_info} {failure_info}")
    
    print(colored(f"\n📊 Complete match data saved for analysis in: _data/{match_id}/", "green"))
    if move_analysis_success:
        print(colored(f"✅ Move quality analysis completed and integrated into notebook", "green"))
    else:
        print(colored(f"⚠️  Move quality analysis skipped (enable with --run_move_analysis=true)", "yellow"))
    print(colored(f"📓 Jupyter notebook created for detailed analysis", "green"))


if __name__ == "__main__":
    app.run(main) 