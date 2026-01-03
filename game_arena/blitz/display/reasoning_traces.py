#!/usr/bin/env python3
"""Reasoning trace display functions for blitz chess matches."""

import termcolor

colored = termcolor.colored


def display_reasoning_traces(response_obj, generate_returns=None, show_traces: bool = True):
    """Display reasoning traces from a response object if enabled and available.
    
    Args:
        response_obj: Response object with main_response_and_thoughts attribute
        generate_returns: Optional list of generate return objects for rethinking case
        show_traces: Whether to actually show the traces (controlled by flag)
    """
    if not show_traces:
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


def _display_single_reasoning_trace(response_obj, prefix: str):
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
        words = full_text.split() if full_text else []
        if len(words) > 100:
            first_50 = ' '.join(words[:50])
            last_50 = ' '.join(words[-50:])
            print(colored(f"{prefix}Response (first 50 words): {first_50}...", "magenta"))
            print(colored(f"{prefix}Response (last 50 words): ...{last_50}", "magenta"))
        elif len(words) > 0:
            print(colored(f"{prefix}Response: {full_text}", "magenta"))

