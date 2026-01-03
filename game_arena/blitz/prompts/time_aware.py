#!/usr/bin/env python3
"""Time-aware prompt generation for blitz chess matches."""

from typing import Dict

from game_arena.harness import game_notation_examples
from game_arena.harness import tournament_util
from game_arena.blitz.core.clock import PlayerClock
from game_arena.blitz.display.formatting import format_time
from game_arena.blitz.prompts.dramatic import (
    create_dramatic_time_pressure_text,
    create_dramatic_instruction_text,
)


def create_time_aware_prompt_substitutions(
    pyspiel_state, 
    player_clock: PlayerClock,
    opponent_clock: PlayerClock,
    increment_seconds: int,
    is_blitz: bool = True,
    use_dramatic_pressure: bool = False,
    previous_response_analysis: str = ""
) -> Dict[str, str]:
    """Create prompt substitutions including time information.
    
    Args:
        pyspiel_state: Current game state from pyspiel
        player_clock: Current player's clock
        opponent_clock: Opponent's clock
        increment_seconds: Time increment per move
        is_blitz: Whether this is a blitz game with time pressure
        use_dramatic_pressure: Whether to use dramatic time pressure prompts
        previous_response_analysis: Optional analysis of previous response for stateful feedback
        
    Returns:
        Dictionary of prompt substitutions
    """
    base_substitutions = {
        "readable_state_str": tournament_util.convert_to_readable_state(
            game_short_name="chess",
            state_str=pyspiel_state.to_string(),
            current_player=pyspiel_state.current_player(),
        ),
        "move_history": (
            tournament_util.get_action_string_history(pyspiel_state) or "None"
        ),
        "player_name": game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"][
            "player_map"
        ][pyspiel_state.current_player()],
        "move_notation": game_notation_examples.GAME_SPECIFIC_NOTATIONS[
            "chess"
        ]["move_notation"],
        "notation": game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"][
            "state_notation"
        ],
    }
    
    if is_blitz:
        if use_dramatic_pressure:
            # Use dramatic time pressure mode
            dramatic_pressure = create_dramatic_time_pressure_text(
                player_clock.time_remaining,
                opponent_clock.time_remaining,
                increment_seconds
            )
            dramatic_instruction = create_dramatic_instruction_text(player_clock.time_remaining)
            base_substitutions["dramatic_time_pressure"] = dramatic_pressure
            base_substitutions["dramatic_instruction"] = dramatic_instruction
            base_substitutions["time_info"] = ""  # Dramatic pressure replaces time_info
        else:
            # Use normal time_info format
            time_info = f"""
BLITZ CHESS TIME INFORMATION:
⏰ Your remaining time: {format_time(player_clock.time_remaining)}
⏰ Opponent's remaining time: {format_time(opponent_clock.time_remaining)}
⏰ Time increment per move: +{increment_seconds} seconds

⚠️  CRITICAL TIME RULES:
- This is REAL WALL CLOCK TIME - your thinking/reasoning time directly consumes your clock
- You lose immediately if your time runs out (time forfeit)
- Longer reasoning traces = more time consumed = higher risk of time forfeit
- You must balance move quality vs. time management
- Each move adds {increment_seconds} seconds to your clock after you play it
- Consider quick, good moves over perfect moves that consume too much time

Current time pressure level: {"🔴 HIGH" if player_clock.time_remaining < 60 else "🟡 MEDIUM" if player_clock.time_remaining < 120 else "🟢 LOW"}
"""
            base_substitutions["time_info"] = time_info
            base_substitutions["dramatic_time_pressure"] = ""
            base_substitutions["dramatic_instruction"] = "Reason step by step to come up with your move, then output your final answer in the format \"Final Answer: X\" where X is your chosen move in algebraic notation."
    else:
        base_substitutions["time_info"] = ""
        base_substitutions["dramatic_time_pressure"] = ""
        base_substitutions["dramatic_instruction"] = "Reason step by step to come up with your move, then output your final answer in the format \"Final Answer: X\" where X is your chosen move in algebraic notation."
    
    # Add stateful previous response analysis
    base_substitutions["previous_response_analysis"] = previous_response_analysis
    
    return base_substitutions

