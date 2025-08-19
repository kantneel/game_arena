#!/usr/bin/env python3
"""Utility classes and functions for blitz chess matches with accurate timing."""

import time
from typing import Dict, List, Tuple, Optional
import dataclasses
import logging

from game_arena.harness import model_generation
from game_arena.harness import model_generation_sdk
from game_arena.harness import tournament_util
import termcolor

colored = termcolor.colored

@dataclasses.dataclass
class PlayerClock:
    """Tracks time remaining for a player."""
    time_remaining: float  # seconds
    is_active: bool = False
    move_start_time: Optional[float] = None
    total_moves: int = 0
    total_thinking_time: float = 0.0
    
    def start_move(self):
        """Start timing a move."""
        self.is_active = True
        self.move_start_time = time.time()
    
    def end_move(self, network_latency: float = 0.0, increment_seconds: float = 0.0) -> float:
        """End timing a move and return actual thinking time."""
        if not self.is_active or self.move_start_time is None:
            return 0.0
        
        move_end_time = time.time()
        total_move_time = move_end_time - self.move_start_time
        thinking_time = max(0.0, total_move_time - network_latency)
        
        self.time_remaining -= thinking_time
        self.time_remaining += increment_seconds  # Add increment
        self.total_thinking_time += thinking_time
        self.total_moves += 1
        self.is_active = False
        self.move_start_time = None
        
        return thinking_time

@dataclasses.dataclass 
class MoveStats:
    """Statistics for a single move."""
    player: str
    move_number: int
    move_notation: str
    thinking_time: float
    time_remaining_after: float
    reasoning_tokens: Optional[int]
    total_tokens: Optional[int]
    network_latency: float
    retry_count: int = 0
    total_retry_time: float = 0.0
    had_parsing_failure: bool = False

@dataclasses.dataclass
class GameStats:
    """Statistics for a complete game."""
    game_number: int
    winner: str
    result_string: str
    model_a_color: str
    total_moves: int
    duration: float
    move_stats: List[MoveStats]
    model_a_final_time: float
    model_b_final_time: float
    model_a_parsing_failures: int = 0
    model_b_parsing_failures: int = 0

class NoRetryModelWrapper:
    """Wrapper that disables automatic retries and handles them manually for accurate timing."""
    
    def __init__(self, wrapped_model, max_retries: int = 3, base_delay: float = 1.0):
        self.wrapped_model = wrapped_model
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # We'll call the underlying _generate method directly to bypass the retry decorator
        # This avoids the automatic retry logic entirely
        self._direct_generate = wrapped_model._generate
    
    @property
    def model_name(self) -> str:
        return self.wrapped_model.model_name
        
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should be retried."""
        return not isinstance(exception, model_generation.DoNotRetryError)
    
    def generate_with_text_input(self, model_input: tournament_util.ModelTextInput) -> Tuple[tournament_util.GenerateReturn, int, float]:
        """
        Generate with retry logic that doesn't count retry time.
        
        Returns:
            Tuple of (GenerateReturn, retry_count, total_retry_time)
        """
        retry_count = 0
        total_retry_time = 0.0
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Time only the successful call - call _generate directly to bypass retry decorator
                call_start = time.time()
                
                # Convert ModelTextInput to the format expected by _generate
                if hasattr(self.wrapped_model, '_generate'):
                    if isinstance(self.wrapped_model, model_generation_sdk.OpenAIChatCompletionsModel):
                        # OpenAI format
                        content = [{"type": "text", "text": model_input.prompt_text}]
                        result = self._direct_generate(content, model_input.system_instruction)
                    elif isinstance(self.wrapped_model, model_generation_sdk.AIStudioModel):
                        # AI Studio format (e.g., Gemini models)
                        contents = [model_input.prompt_text]
                        result = self._direct_generate(contents, model_input.system_instruction)
                    else:
                        # Fallback - try the original method but catch exceptions ourselves
                        result = self.wrapped_model.generate_with_text_input(model_input)
                else:
                    # Fallback - use the wrapped method
                    result = self.wrapped_model.generate_with_text_input(model_input)
                
                call_end = time.time()
                
                # Only count the time of the successful call
                actual_call_time = call_end - call_start
                
                return result, retry_count, total_retry_time
                
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e) or attempt >= self.max_retries:
                    # Don't retry or max attempts reached
                    break
                
                # Calculate retry delay
                retry_delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                retry_delay = min(retry_delay, 60.0)  # Cap at 60 seconds
                
                print(colored(f"API call failed (attempt {attempt + 1}), retrying in {retry_delay:.1f}s: {e}", "yellow"))
                
                retry_start = time.time()
                time.sleep(retry_delay)
                retry_end = time.time()
                
                retry_count += 1
                total_retry_time += (retry_end - retry_start)
        
        # If we get here, all retries failed
        raise last_exception

def format_time(seconds: float) -> str:
    """Format time as MM:SS.s"""
    if seconds < 0:
        return "00:00.0"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"

def calibrate_network_latency(model_wrapper: NoRetryModelWrapper, calibration_rounds: int) -> float:
    """Calibrate network latency by making minimal token requests."""
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
            
            # change beginning of prompt to ideally avoid prompt caching
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