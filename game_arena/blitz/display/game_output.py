#!/usr/bin/env python3
"""Game-level output and display functions."""

import termcolor

from game_arena.blitz.core.types import GameStats
from game_arena.blitz.display.formatting import format_time

colored = termcolor.colored


def print_detailed_game_analysis(game_stats: GameStats):
    """Print detailed analysis of the game."""
    print(colored(f"\n📊 DETAILED ANALYSIS - GAME {game_stats.game_number}", "magenta"))
    print(f"Duration: {game_stats.duration:.1f}s, Moves: {game_stats.total_moves}")
    
    # Get unique player names from move stats
    unique_players = list(set(m.player for m in game_stats.move_stats))
    
    # Determine which player is which based on model_a_color
    if len(unique_players) == 2:
        if game_stats.model_a_color == "white":
            # Find white and black players
            white_moves = [m for m in game_stats.move_stats if m.move_number % 2 == 1]
            black_moves = [m for m in game_stats.move_stats if m.move_number % 2 == 0]
            model_a_name = white_moves[0].player if white_moves else unique_players[0]
            model_b_name = black_moves[0].player if black_moves else unique_players[1]
        else:  # model_a is black
            black_moves = [m for m in game_stats.move_stats if m.move_number % 2 == 1]
            white_moves = [m for m in game_stats.move_stats if m.move_number % 2 == 0]
            model_a_name = black_moves[0].player if black_moves else unique_players[0]
            model_b_name = white_moves[0].player if white_moves else unique_players[1]
    else:
        # Fallback if we can't determine from moves
        model_a_name = unique_players[0] if unique_players else "Model A"
        model_b_name = unique_players[1] if len(unique_players) > 1 else "Model B"
    
    print(f"Final times - {model_a_name}: {format_time(game_stats.model_a_final_time)}, "
          f"{model_b_name}: {format_time(game_stats.model_b_final_time)}")
    
    # Display parsing failure information
    if game_stats.model_a_parsing_failures > 0 or game_stats.model_b_parsing_failures > 0:
        print(colored(f"⚠️  Parsing failures - {model_a_name}: {game_stats.model_a_parsing_failures}, "
                     f"{model_b_name}: {game_stats.model_b_parsing_failures}", "yellow"))
    else:
        print(colored("✅ No parsing failures in this game", "green"))
    
    # Aggregate statistics by player
    model_a_moves = [m for m in game_stats.move_stats if m.player == model_a_name]
    model_b_moves = [m for m in game_stats.move_stats if m.player == model_b_name]
    
    for player_name, moves in [(model_a_name, model_a_moves), (model_b_name, model_b_moves)]:
        if not moves:
            continue
            
        avg_thinking_time = sum(m.thinking_time for m in moves) / len(moves)
        total_thinking_time = sum(m.thinking_time for m in moves)
        avg_reasoning_tokens = sum(m.reasoning_tokens or 0 for m in moves) / len(moves)
        
        # Calculate retry statistics
        total_retries = sum(m.retry_count for m in moves)
        total_retry_time = sum(m.total_retry_time for m in moves)
        moves_with_retries = len([m for m in moves if m.retry_count > 0])
        
        print(f"\n{player_name} stats:")
        print(f"  Moves played: {len(moves)}")
        print(f"  Avg thinking time: {avg_thinking_time:.2f}s")
        print(f"  Total thinking time: {total_thinking_time:.1f}s")
        print(f"  Avg reasoning tokens: {avg_reasoning_tokens:.0f}")
        
        # Retry information
        if total_retries > 0:
            print(colored(f"  🔄 Total retries: {total_retries} across {moves_with_retries} moves", "yellow"))
            print(colored(f"  🔄 Total retry time: {total_retry_time:.1f}s (excluded from clock)", "yellow"))
            print(colored(f"  💾 Time saved by excluding retries: {total_retry_time:.1f}s", "green"))
        else:
            print(colored(f"  ✅ No retries needed - all API calls successful", "green"))
        
        # Find slowest and fastest moves
        if moves:
            slowest = max(moves, key=lambda m: m.thinking_time)
            fastest = min(moves, key=lambda m: m.thinking_time)
            print(f"  Slowest move: {slowest.move_notation} ({slowest.thinking_time:.2f}s)")
            if slowest.retry_count > 0:
                print(f"    └─ Had {slowest.retry_count} retries ({slowest.total_retry_time:.1f}s excluded)")
            print(f"  Fastest move: {fastest.move_notation} ({fastest.thinking_time:.2f}s)")
            if fastest.retry_count > 0:
                print(f"    └─ Had {fastest.retry_count} retries ({fastest.total_retry_time:.1f}s excluded)")

