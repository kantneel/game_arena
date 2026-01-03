#!/usr/bin/env python3
"""Move parsing utilities for blitz chess matches."""

from typing import Optional, Tuple

import termcolor

from game_arena.harness import parsers
from game_arena.blitz.core.types import GameStats

colored = termcolor.colored


def handle_simple_parsing(game_state, player_info: dict, response, parser, 
                         max_failures: int) -> Tuple[Optional[str], bool, Optional[GameStats]]:
    """
    Handle parsing with simple parser (no rethinking).
    Returns: (move_notation, should_continue, game_stats_if_ended)
    """
    parser_input = parsers.TextParserInput(
        text=response.main_response,
        state_str=game_state.pyspiel_state.to_string(),
        legal_moves=parsers.get_legal_action_strings(game_state.pyspiel_state),
        player_number=game_state.pyspiel_state.current_player(),
    )
    
    try:
        parser_output = parser.parse(parser_input)
        if parser_output is None:
            # Parsing failed
            game_state.increment_parsing_failures(player_info['is_white'])
            current_failures = game_state.get_parsing_failures(player_info['is_white'])
            
            print(colored(f"⚠️  Parser failed for {player_info['player_name']} (failure {current_failures}/{max_failures})", "yellow"))
            
            failure_result = game_state.check_parsing_failure_limit(player_info['is_white'], max_failures)
            if failure_result:
                return None, False, failure_result
            
            print(colored(f"🔄 Skipping {player_info['player_name']}'s turn due to parsing failure. Continuing game...", "yellow"))
            return None, True, None  # Continue without making a move
        else:
            return parser_output, False, None  # Successful parse
            
    except Exception as e:
        # Treat exceptions as parsing failures
        game_state.increment_parsing_failures(player_info['is_white'])
        current_failures = game_state.get_parsing_failures(player_info['is_white'])
        
        print(colored(f"⚠️  Error parsing/applying move for {player_info['player_name']}: {e} (failure {current_failures}/{max_failures})", "yellow"))
        
        failure_result = game_state.check_parsing_failure_limit(player_info['is_white'], max_failures)
        if failure_result:
            return None, False, failure_result
        
        print(colored(f"🔄 Skipping {player_info['player_name']}'s turn due to parsing error. Continuing game...", "yellow"))
        return None, True, None

