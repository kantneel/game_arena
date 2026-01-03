#!/usr/bin/env python3
"""Match-level output and display functions."""

from typing import List

import termcolor

from game_arena.blitz.core.types import GameStats

colored = termcolor.colored


def print_comprehensive_match_analysis(all_game_stats: List[GameStats]):
    """Print comprehensive analysis across all games."""
    print(colored("\n🧠 REASONING EFFICIENCY ANALYSIS:", "cyan"))
    
    # Get model names from the first game's move stats
    model_a_name = "Model A"  # default fallback
    model_b_name = "Model B"  # default fallback
    
    if all_game_stats and all_game_stats[0].move_stats:
        # Determine model names from first game
        first_game = all_game_stats[0]
        unique_players = list(set(m.player for m in first_game.move_stats))
        
        if len(unique_players) == 2:
            if first_game.model_a_color == "white":
                white_moves = [m for m in first_game.move_stats if m.move_number % 2 == 1]
                black_moves = [m for m in first_game.move_stats if m.move_number % 2 == 0]
                model_a_name = white_moves[0].player if white_moves else unique_players[0]
                model_b_name = black_moves[0].player if black_moves else unique_players[1]
            else:  # model_a is black
                black_moves = [m for m in first_game.move_stats if m.move_number % 2 == 1]
                white_moves = [m for m in first_game.move_stats if m.move_number % 2 == 0]
                model_a_name = black_moves[0].player if black_moves else unique_players[0]
                model_b_name = white_moves[0].player if white_moves else unique_players[1]
    
    # Calculate overall parsing failure statistics
    total_model_a_parsing_failures = sum(g.model_a_parsing_failures for g in all_game_stats)
    total_model_b_parsing_failures = sum(g.model_b_parsing_failures for g in all_game_stats)
    games_with_model_a_failures = len([g for g in all_game_stats if g.model_a_parsing_failures > 0])
    games_with_model_b_failures = len([g for g in all_game_stats if g.model_b_parsing_failures > 0])
    
    print(colored(f"\n⚠️  PARSING FAILURE ANALYSIS:", "yellow"))
    print(f"Total parsing failures - {model_a_name}: {total_model_a_parsing_failures} across {games_with_model_a_failures} games")
    print(f"Total parsing failures - {model_b_name}: {total_model_b_parsing_failures} across {games_with_model_b_failures} games")
    
    all_model_a_moves = []
    all_model_b_moves = []
    
    for game in all_game_stats:
        all_model_a_moves.extend([m for m in game.move_stats if m.player == model_a_name])
        all_model_b_moves.extend([m for m in game.move_stats if m.player == model_b_name])
    
    for player_name, moves in [(model_a_name, all_model_a_moves), (model_b_name, all_model_b_moves)]:
        if not moves:
            continue
            
        avg_thinking = sum(m.thinking_time for m in moves) / len(moves)
        avg_reasoning_tokens = sum(m.reasoning_tokens or 0 for m in moves) / len(moves)
        
        # Calculate reasoning efficiency (tokens per second)
        efficiency_scores = []
        for move in moves:
            if move.thinking_time > 0 and move.reasoning_tokens:
                efficiency_scores.append(move.reasoning_tokens / move.thinking_time)
        
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
        
        # Calculate retry statistics
        total_retries = sum(m.retry_count for m in moves)
        total_retry_time = sum(m.total_retry_time for m in moves)
        moves_with_retries = len([m for m in moves if m.retry_count > 0])
        retry_rate = (moves_with_retries / len(moves)) * 100 if moves else 0
        
        print(f"\n{player_name} overall performance:")
        print(f"  Total moves: {len(moves)}")
        print(f"  Avg thinking time: {avg_thinking:.2f}s")
        print(f"  Avg reasoning tokens: {avg_reasoning_tokens:.0f}")
        print(f"  Reasoning efficiency: {avg_efficiency:.1f} tokens/second")
        
        # Retry analysis
        if total_retries > 0:
            print(colored(f"  🔄 API reliability: {retry_rate:.1f}% moves needed retries", "yellow"))
            print(colored(f"  🔄 Total retry overhead: {total_retry_time:.1f}s across {total_retries} retries", "yellow"))
            print(colored(f"  💾 Total time saved by excluding retries: {total_retry_time:.1f}s", "green"))
        else:
            print(colored(f"  ✅ Perfect API reliability: 0% moves needed retries", "green"))
        
        # Time pressure analysis
        time_pressure_moves = [m for m in moves if m.time_remaining_after < 60]
        if time_pressure_moves:
            avg_pressure_thinking = sum(m.thinking_time for m in time_pressure_moves) / len(time_pressure_moves)
            pressure_retries = sum(m.retry_count for m in time_pressure_moves)
            print(f"  Under time pressure (<60s): {len(time_pressure_moves)} moves, avg {avg_pressure_thinking:.2f}s thinking")
            if pressure_retries > 0:
                print(colored(f"    └─ Had {pressure_retries} retries under pressure (time saved!)", "green"))

