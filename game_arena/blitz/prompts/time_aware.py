#!/usr/bin/env python3
"""Time-aware prompt generation for blitz chess matches."""

from dataclasses import dataclass
from typing import Dict, Optional

from game_arena.harness import game_notation_examples
from game_arena.harness import tournament_util
from game_arena.blitz.core.clock import PlayerClock
from game_arena.blitz.display.formatting import format_time
from game_arena.blitz.prompts.dramatic import (
    create_dramatic_time_pressure_text,
    create_dramatic_instruction_text,
)


def get_move_history_in_format(
    pyspiel_state, 
    notation_format: str = "san",
    model_a_name: str = "White",
    model_b_name: str = "Black",
) -> str:
    """Get move history in the specified notation format.
    
    Args:
        pyspiel_state: Current game state
        notation_format: One of "san", "lan", or "pgn"
        model_a_name: Name for white player (used in PGN headers)
        model_b_name: Name for black player (used in PGN headers)
        
    Returns:
        Move history string in the requested format
    """
    game_name = pyspiel_state.get_game().get_type().short_name
    if not game_name.startswith("chess"):
        return tournament_util.get_action_string_history(pyspiel_state) or "None"
    
    pgn_game = tournament_util.get_pgn(pyspiel_state)
    
    if notation_format == "pgn":
        # Full PGN format with headers
        import io
        from datetime import datetime
        
        # Set headers
        pgn_game.headers["Event"] = "Blitz Chess Match"
        pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        pgn_game.headers["White"] = model_a_name if len(pyspiel_state.history()) % 2 == 0 or len(pyspiel_state.history()) == 0 else model_b_name
        pgn_game.headers["Black"] = model_b_name if len(pyspiel_state.history()) % 2 == 0 or len(pyspiel_state.history()) == 0 else model_a_name
        pgn_game.headers["Result"] = "*"  # Game in progress
        
        # Export full PGN
        exporter = io.StringIO()
        exporter.write(str(pgn_game))
        return exporter.getvalue().strip()
        
    elif notation_format == "lan":
        # Long Algebraic Notation (e2e4 instead of e4)
        return tournament_util.format_chess_movetext(
            pgn_game,
            numbering_scheme="default",
            use_lan=True,
            add_current_fen=False,
        )
    else:
        # Standard Algebraic Notation (default)
        return tournament_util.format_chess_movetext(
            pgn_game,
            numbering_scheme="default",
            use_lan=False,
            add_current_fen=False,
        )


@dataclass
class PreviousResponseData:
    """Data about the model's previous response for feedback."""
    time_taken_seconds: float
    thinking_tokens: int
    output_tokens: int
    tokens_per_second: float
    time_remaining_after: float


def create_response_feedback_text(
    prev_data: Optional[PreviousResponseData],
    current_time_remaining: float,
    include_efficiency_guidance: bool = False
) -> str:
    """Generate feedback about the model's previous response.
    
    This enables "recurrent" awareness where the model can adapt based on
    how quickly it's actually generating tokens.
    
    Args:
        prev_data: Data from the previous response, or None if first move
        current_time_remaining: Current time on the clock
        include_efficiency_guidance: Whether to include token rate guidance
        
    Returns:
        Feedback text to include in prompt, or empty string
    """
    if prev_data is None:
        return ""
    
    feedback_parts = [
        "\n📊 YOUR PREVIOUS RESPONSE ANALYSIS:",
        f"• Your last move took {prev_data.time_taken_seconds:.1f} seconds",
        f"• You used {prev_data.thinking_tokens:,} thinking tokens",
        f"• Your thinking speed: ~{prev_data.tokens_per_second:.0f} tokens/second",
    ]
    
    if include_efficiency_guidance:
        # Calculate how many tokens the model can "afford" given remaining time
        # Leave buffer for safety
        safe_time_budget = min(current_time_remaining * 0.5, 30)  # Use at most 50% of time or 30s
        affordable_tokens = int(safe_time_budget * prev_data.tokens_per_second)
        
        feedback_parts.extend([
            "",
            "⚡ EFFICIENCY GUIDANCE:",
            f"• At your current speed, generating {affordable_tokens:,} tokens would take ~{safe_time_budget:.0f}s",
            f"• You have {format_time(current_time_remaining)} remaining",
        ])
        
        # Check critical first (more specific), then low time
        if current_time_remaining < 30:
            feedback_parts.append("• 🚨 CRITICAL: Minimize thinking tokens immediately!")
        elif current_time_remaining < 60:
            feedback_parts.append("• ⚠️ Consider shorter reasoning to preserve time!")
    
    return "\n".join(feedback_parts) + "\n"


def create_time_aware_prompt_substitutions(
    pyspiel_state, 
    player_clock: PlayerClock,
    opponent_clock: PlayerClock,
    increment_seconds: int,
    is_blitz: bool = True,
    use_dramatic_pressure: bool = False,
    previous_response_analysis: str = "",
    # New experiment flags
    enable_time_pressure_prompt: bool = True,
    previous_response_data: Optional[PreviousResponseData] = None,
    enable_response_feedback: bool = False,
    enable_efficiency_guidance: bool = False,
    # Move notation format
    move_notation_format: str = "san",
    model_a_name: str = "White",
    model_b_name: str = "Black",
) -> Dict[str, str]:
    """Create prompt substitutions including time information.
    
    Args:
        pyspiel_state: Current game state from pyspiel
        player_clock: Current player's clock
        opponent_clock: Opponent's clock
        increment_seconds: Time increment per move
        is_blitz: Whether this is a blitz game with time pressure
        use_dramatic_pressure: Whether to use dramatic time pressure prompts
        previous_response_analysis: Optional analysis of previous response (legacy)
        enable_time_pressure_prompt: Whether to include time pressure info (for ablation)
        previous_response_data: Data about previous response for feedback
        enable_response_feedback: Whether to include response feedback
        enable_efficiency_guidance: Whether to include token rate calculations
        move_notation_format: Format for move history ("san", "lan", or "pgn")
        model_a_name: Name of model A for PGN headers
        model_b_name: Name of model B for PGN headers
        
    Returns:
        Dictionary of prompt substitutions
    """
    # Get move history in the specified format
    move_history = get_move_history_in_format(
        pyspiel_state,
        notation_format=move_notation_format,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
    ) or "None"
    
    base_substitutions = {
        "readable_state_str": tournament_util.convert_to_readable_state(
            game_short_name="chess",
            state_str=pyspiel_state.to_string(),
            current_player=pyspiel_state.current_player(),
        ),
        "move_history": move_history,
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
    
    # Default instruction - allows model to choose reasoning depth based on time pressure
    default_instruction = 'Reason as much as you think is necessary for this position (could be extensive analysis or none at all depending on time pressure and position complexity), then output your final answer in the format "Final Answer: X" where X is your chosen move in algebraic notation.'
    
    if is_blitz and enable_time_pressure_prompt:
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
            base_substitutions["time_info"] = ""
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
            base_substitutions["dramatic_instruction"] = default_instruction
    else:
        # No time pressure prompts (ablation experiment)
        base_substitutions["time_info"] = ""
        base_substitutions["dramatic_time_pressure"] = ""
        base_substitutions["dramatic_instruction"] = default_instruction
    
    # Add response feedback if enabled
    if enable_response_feedback and previous_response_data:
        response_feedback = create_response_feedback_text(
            previous_response_data,
            player_clock.time_remaining,
            enable_efficiency_guidance
        )
        base_substitutions["response_feedback"] = response_feedback
    else:
        base_substitutions["response_feedback"] = ""
    
    # Legacy field for backwards compatibility
    base_substitutions["previous_response_analysis"] = previous_response_analysis
    
    return base_substitutions
