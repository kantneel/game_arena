#!/usr/bin/env python3
"""Network latency calibration for blitz chess matches."""

import time

import termcolor

from game_arena.harness import tournament_util
from game_arena.blitz.models.wrappers import NoRetryModelWrapper

colored = termcolor.colored


def calibrate_network_latency(model_wrapper: NoRetryModelWrapper, calibration_rounds: int) -> float:
    """Calibrate network latency by making minimal token requests.
    
    Args:
        model_wrapper: Wrapped model to calibrate
        calibration_rounds: Number of calibration rounds to run
        
    Returns:
        Average network latency in seconds
    """
    print(colored("Calibrating network latency...", "yellow"))
    latencies = []
    
    # Check if this is a Gemini model (more prone to rate limiting during rapid calls)
    is_gemini_model = 'gemini' in model_wrapper.model_name.lower()
    
    # Use minimal prompt and slightly higher token limit for stability
    minimal_text = "Hi, what's your name?"
    
    # Temporarily set max tokens to 5 instead of 1 for better stability
    original_max_tokens = None
    if hasattr(model_wrapper.wrapped_model, '_model_options') and model_wrapper.wrapped_model._model_options is not None:
        original_max_tokens = model_wrapper.wrapped_model._model_options.get('max_output_tokens')
        model_wrapper.wrapped_model._model_options['max_output_tokens'] = 5  # Changed from 1 to 5
    elif hasattr(model_wrapper.wrapped_model, '_model_options'):
        model_wrapper.wrapped_model._model_options = {'max_output_tokens': 5}
    else:
        model_wrapper.wrapped_model._model_options = {'max_output_tokens': 5}
    
    if is_gemini_model:
        print(colored("  Detected Gemini model - using conservative calibration settings", "yellow"))
    
    for i in range(calibration_rounds):
        try:
            # Add delay between calibration calls to avoid rate limiting
            if i > 0:
                # Longer delay for Gemini to avoid rate limiting
                delay = 1.0 if is_gemini_model else 0.5
                time.sleep(delay)
            
            # Change beginning of prompt to ideally avoid prompt caching
            minimal_prompt = tournament_util.ModelTextInput(prompt_text=f"Round {i+1}: Hi, what's your name?")
            start_time = time.time()
            response, retry_count, retry_time = model_wrapper.generate_with_text_input(minimal_prompt)
            end_time = time.time()
            
            # Only count the actual API call time, not retry time
            latency = (end_time - start_time) - retry_time
            latencies.append(latency)
            
            if retry_count > 0:
                print(f"  Round {i+1}: {latency:.3f}s (after {retry_count} retries)")
            else:
                print(f"  Round {i+1}: {latency:.3f}s")
            
        except Exception as e:
            print(colored(f"  Round {i+1} failed: {e}", "red"))
            # Use a more conservative fallback that accounts for retry delays
            latencies.append(1.0)  # More conservative fallback (was 0.5)
            
            # For Gemini, add extra delay after failures to avoid cascading rate limits
            if is_gemini_model and i < calibration_rounds - 1:
                print(colored("  Adding extra delay for Gemini after failure...", "yellow"))
                time.sleep(2.0)
    
    # Restore original max tokens
    try:
        if hasattr(model_wrapper.wrapped_model, '_model_options'):
            if original_max_tokens is not None:
                model_wrapper.wrapped_model._model_options['max_output_tokens'] = original_max_tokens
            else:
                if 'max_output_tokens' in model_wrapper.wrapped_model._model_options:
                    del model_wrapper.wrapped_model._model_options['max_output_tokens']
    except:
        pass
    
    avg_latency = sum(latencies) / len(latencies)
    print(colored(f"Average network latency: {avg_latency:.3f}s", "green"))
    return avg_latency

