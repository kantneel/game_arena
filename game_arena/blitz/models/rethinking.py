#!/usr/bin/env python3
"""Rethinking move handler for blitz chess matches."""

import time
from typing import Tuple, Optional

import termcolor

from game_arena.harness import tournament_util
from game_arena.harness import prompts
from game_arena.blitz.core.types import GameStats
from game_arena.blitz.models.wrappers import BlitzModelWrapper

colored = termcolor.colored


def handle_rethinking_move(game_state, player_info: dict, 
                          sampler, prompt_substitutions: dict, max_failures: int, 
                          max_rethinks: int) -> Tuple[Optional[str], bool, Optional[GameStats], dict]:
    """
    Handle move generation with rethinking.
    
    Args:
        game_state: Current game state
        player_info: Dictionary with player model, clock, etc.
        sampler: RethinkSampler for the current player
        prompt_substitutions: Prompt template substitutions
        max_failures: Maximum parsing failures allowed
        max_rethinks: Maximum rethinking attempts
        
    Returns: 
        Tuple of (move_notation, should_continue, game_stats_if_ended, retry_info)
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
        error_stats = GameStats(
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

