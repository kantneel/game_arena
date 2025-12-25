#!/usr/bin/env python3
"""Core game engine for blitz chess matches."""

import time
from typing import Dict, Tuple, Optional
import termcolor
import pyspiel
from game_arena.harness import game_notation_examples
from game_arena.harness import tournament_util
from game_arena.harness import prompt_generation
from game_arena.harness import prompts
from game_arena.harness import parsers
from game_arena.blitz import utils
from game_arena.blitz import data_collector

colored = termcolor.colored


def create_time_aware_prompt_substitutions(
    pyspiel_state, 
    player_clock: utils.PlayerClock,
    opponent_clock: utils.PlayerClock,
    increment_seconds: int,
    is_blitz: bool = True,
    use_dramatic_pressure: bool = False,
    previous_response_analysis: str = ""
) -> Dict[str, str]:
    """Create prompt substitutions including time information."""
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
                player_clock, opponent_clock, increment_seconds
            )
            dramatic_instruction = create_dramatic_instruction_text(player_clock.time_remaining)
            base_substitutions["dramatic_time_pressure"] = dramatic_pressure
            base_substitutions["dramatic_instruction"] = dramatic_instruction
            base_substitutions["time_info"] = ""  # Dramatic pressure replaces time_info
        else:
            # Use normal time_info format
            time_info = f"""
BLITZ CHESS TIME INFORMATION:
⏰ Your remaining time: {utils.format_time(player_clock.time_remaining)}
⏰ Opponent's remaining time: {utils.format_time(opponent_clock.time_remaining)}
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


def create_dramatic_time_pressure_text(
    player_clock: utils.PlayerClock,
    opponent_clock: utils.PlayerClock,
    increment_seconds: int
) -> str:
    """Generate dramatic time pressure text with ALL CAPS emphasis."""
    
    time_remaining = player_clock.time_remaining
    
    if time_remaining < 30:
        urgency_level = "EXTREME"
        pressure_color = "🔴🔴🔴"
        dramatic_text = """
🚨🚨🚨 CRITICAL TIME EMERGENCY!!! 🚨🚨🚨

⚠️⚠️⚠️ YOU HAVE LESS THAN 30 SECONDS LEFT!!! ⚠️⚠️⚠️

🔥🔥🔥 EVERY SECOND COUNTS - MOVE IMMEDIATELY OR LOSE!!! 🔥🔥🔥

💥 DO NOT OVERTHINK! ANY REASONABLE MOVE IS BETTER THAN TIMING OUT! 💥
💥 EVEN A RANDOM LEGAL MOVE BEATS RUNNING OUT OF TIME! 💥
💥 SPEED IS MORE IMPORTANT THAN PERFECTION RIGHT NOW! 💥

⏰ THE CLOCK IS YOUR BIGGEST ENEMY - NOT YOUR OPPONENT! ⏰"""
    elif time_remaining < 60:
        urgency_level = "HIGH"
        pressure_color = "🔴🔴"
        dramatic_text = """
🚨🚨 TIME PRESSURE ALERT!!! 🚨🚨

⚠️ LESS THAN 1 MINUTE REMAINING! ⚠️

🔥 THINK FAST - EVERY SECOND MATTERS! 🔥
🔥 QUICK GOOD MOVES BEAT SLOW PERFECT MOVES! 🔥
🔥 TIME FORFEIT = INSTANT LOSS! 🔥

⏰ PRIORITIZE SPEED OVER DEEP ANALYSIS! ⏰"""
    elif time_remaining < 120:
        urgency_level = "MEDIUM"
        pressure_color = "🟡🟡"
        dramatic_text = """
⚠️ TIME PRESSURE BUILDING! ⚠️

🔥 UNDER 2 MINUTES - START MOVING FASTER! 🔥
⏰ BALANCE QUALITY WITH SPEED! ⏰
💭 LIMIT YOUR THINKING TIME PER MOVE! 💭"""
    else:
        urgency_level = "LOW"
        pressure_color = "🟢"
        dramatic_text = """
✅ COMFORTABLE TIME CUSHION ✅
💭 You can afford some analysis, but don't waste time! 💭"""

    return f"""
{dramatic_text}

{pressure_color} TIME PRESSURE LEVEL: {urgency_level} {pressure_color}
⏰ YOUR TIME: {utils.format_time(time_remaining)}
⏰ OPPONENT TIME: {utils.format_time(opponent_clock.time_remaining)}
⏰ INCREMENT: +{increment_seconds}s per move

🎯 REMEMBER: Running out of time = AUTOMATIC LOSS!!!
🎯 A mediocre move in 5 seconds beats a brilliant move in 65 seconds when you only have 60 seconds left!
"""


def create_dramatic_instruction_text(time_remaining: float) -> str:
    """Generate dramatic instruction text based on time remaining."""
    if time_remaining < 30:
        return "MOVE NOW!!! Think for maximum 3 seconds, then output your answer! Time is running out!!!"
    elif time_remaining < 60:
        return "Think quickly and decide fast! Limit yourself to 10 seconds of analysis maximum!"
    elif time_remaining < 120:
        return "Be efficient with your thinking time. Quick analysis, then move!"
    else:
        return "Reason step by step to come up with your move, then output your final answer in the format \"Final Answer: X\" where X is your chosen move in algebraic notation."


class GameState:
    """Tracks the state of a single game."""
    
    def __init__(self, game_number: int, model_a_plays_white: bool, initial_time: int, 
                 model_a_name: str = "Model A", model_b_name: str = "Model B"):
        self.game_number = game_number
        self.model_a_plays_white = model_a_plays_white
        
        # Player assignments
        if model_a_plays_white:
            self.white_name, self.black_name = model_a_name, model_b_name
        else:
            self.white_name, self.black_name = model_b_name, model_a_name
        
        # Clocks
        self.white_clock = utils.PlayerClock(time_remaining=initial_time)
        self.black_clock = utils.PlayerClock(time_remaining=initial_time)
        
        # Game tracking
        self.pyspiel_game = pyspiel.load_game("chess")
        self.pyspiel_state = self.pyspiel_game.new_initial_state()
        self.move_stats = []
        self.move_count = 0
        self.game_start_time = time.time()
        
        # Parsing failure tracking
        self.white_parsing_failures = 0
        self.black_parsing_failures = 0
    
    def get_current_player_info(self, white_model, black_model, white_latency, black_latency):
        """Get current player's model, clock, and info."""
        current_player = self.pyspiel_state.current_player()
        
        if current_player == 0:  # Black
            return {
                'model': black_model,
                'player_name': self.black_name,
                'player_clock': self.black_clock,
                'opponent_clock': self.white_clock,
                'network_latency': black_latency,
                'is_white': False
            }
        else:  # White
            return {
                'model': white_model,
                'player_name': self.white_name,
                'player_clock': self.white_clock,
                'opponent_clock': self.black_clock,
                'network_latency': white_latency,
                'is_white': True
            }
    
    def check_time_forfeit(self) -> Optional[utils.GameStats]:
        """Check if current player has run out of time."""
        current_player = self.pyspiel_state.current_player()
        player_info = self.get_current_player_info(None, None, 0, 0)  # Just need clock info
        
        if player_info['player_clock'].time_remaining <= 0:
            print(colored(f"⏰ TIME FORFEIT! {player_info['player_name']} ran out of time!", "red"))
            
            # Current player lost, so opponent wins
            # Map the winning player to model_a or model_b based on color
            winner = self._map_winning_color_to_model_id(current_player == 0)  # True if black wins, False if white wins
            result_string = "0-1" if current_player == 1 else "1-0"
            
            return self._create_game_stats(winner, result_string)
        
        return None
    
    def increment_parsing_failures(self, is_white: bool, count: int = 1):
        """Increment parsing failure count for the specified player."""
        if is_white:
            self.white_parsing_failures += count
        else:
            self.black_parsing_failures += count
    
    def get_parsing_failures(self, is_white: bool) -> int:
        """Get parsing failure count for the specified player."""
        return self.white_parsing_failures if is_white else self.black_parsing_failures
    
    def check_parsing_failure_limit(self, is_white: bool, max_failures: int) -> Optional[utils.GameStats]:
        """Check if player has exceeded max parsing failures."""
        current_failures = self.get_parsing_failures(is_white)
        
        if current_failures >= max_failures:
            player_name = self.white_name if is_white else self.black_name
            print(colored(f"❌ {player_name} exceeded max parsing failures ({max_failures}), game ends", "red"))
            
            # Opponent wins (if white failed, black wins; if black failed, white wins)
            winner = self._map_winning_color_to_model_id(not is_white)  # True if black wins, False if white wins
            result_string = "0-1" if is_white else "1-0"
            return self._create_game_stats(winner, result_string)
        
        return None
    
    def record_move_to_collector(self, move_notation: str, player_info: dict, 
                               response, thinking_time: float, time_at_turn_start: float,
                               network_latency: float, retry_count: int, 
                               board_state_before: str,
                               prompt_substitutions: dict = None) -> None:
        """Record a move to the data collector for per-game CSV generation."""
        collector = data_collector.get_data_collector()
        
        # Get full response text with thoughts
        response_text = ""
        if hasattr(response, 'main_response_and_thoughts') and response.main_response_and_thoughts:
            response_text = response.main_response_and_thoughts
        elif hasattr(response, 'main_response') and response.main_response:
            response_text = response.main_response
        else:
            response_text = str(response) if response else ""
        
        # Determine color
        color = "white" if player_info['is_white'] else "black"
        
        # Get token information
        thinking_tokens = getattr(response, 'reasoning_tokens', None)
        output_tokens = getattr(response, 'generation_tokens', None)
        prompt_tokens = getattr(response, 'prompt_tokens', None)
        total_tokens = None
        if output_tokens is not None and prompt_tokens is not None:
            total_tokens = output_tokens + prompt_tokens
        
        # Extract time pressure information from prompt substitutions
        time_pressure_level = "LOW"
        used_dramatic_prompts = False
        prompt_template_used = "NO_LEGAL_ACTIONS"
        previous_response_analysis_included = False
        
        if prompt_substitutions:
            # Determine time pressure level
            if time_at_turn_start < 30:
                time_pressure_level = "EXTREME"
            elif time_at_turn_start < 60:
                time_pressure_level = "HIGH"
            elif time_at_turn_start < 120:
                time_pressure_level = "MEDIUM"
            else:
                time_pressure_level = "LOW"
            
            # Check if dramatic prompts were used
            used_dramatic_prompts = bool(prompt_substitutions.get("dramatic_time_pressure", "").strip())
            
            # Check if stateful analysis was included
            previous_response_analysis_included = bool(prompt_substitutions.get("previous_response_analysis", "").strip())
        
        # Get previous move data for trend analysis
        previous_move_time, previous_move_efficiency = collector.get_previous_move_data(
            self.game_number, player_info['player_name']
        )
        
        collector.record_move(
            game_number=self.game_number,
            who_played=player_info['player_name'],
            move_played=move_notation,
            board_state_before=board_state_before,
            time_taken=thinking_time,
            response_text=response_text,
            time_at_turn_start=time_at_turn_start,
            thinking_tokens=thinking_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            move_number=self.move_count + 1,
            color=color,
            network_latency=network_latency,
            retry_count=retry_count,
            # New time pressure parameters
            time_pressure_level=time_pressure_level,
            used_dramatic_prompts=used_dramatic_prompts,
            prompt_template_used=prompt_template_used,
            opponent_time_remaining=player_info['opponent_clock'].time_remaining,
            time_increment=3,  # This should be passed in, but defaulting for now
            previous_response_analysis_included=previous_response_analysis_included,
            previous_move_time=previous_move_time,
            previous_move_efficiency=previous_move_efficiency
        )
    
    def apply_move(self, move_notation: str, player_info: dict, network_latency: float, 
                   increment: int, response, retry_count: int, total_retry_time: float, 
                   thinking_time: float, board_state_before: str = None, 
                   time_at_turn_start: float = None, prompt_substitutions: dict = None) -> bool:
        """Apply a move and record statistics. Returns True if successful."""
        try:
            # Record move to data collector if board state and turn start time provided
            if board_state_before is not None and time_at_turn_start is not None:
                self.record_move_to_collector(
                    move_notation, player_info, response, thinking_time, 
                    time_at_turn_start, network_latency, retry_count, board_state_before,
                    prompt_substitutions
                )
            
            # Record move statistics (keeping for backward compatibility)
            move_stat = utils.MoveStats(
                player=player_info['player_name'],
                move_number=self.move_count + 1,
                move_notation=move_notation,
                thinking_time=thinking_time,  # Use the provided thinking time
                time_remaining_after=player_info['player_clock'].time_remaining,  # Current time remaining
                reasoning_tokens=getattr(response, 'reasoning_tokens', None),
                total_tokens=(getattr(response, 'generation_tokens', 0) or 0) + (getattr(response, 'prompt_tokens', 0) or 0),
                network_latency=network_latency,
                retry_count=retry_count,
                total_retry_time=total_retry_time
            )
            
            # Apply the move
            self.pyspiel_state.apply_action(self.pyspiel_state.string_to_action(move_notation))
            self.move_count += 1
            
            # Store the move stat
            self.move_stats.append(move_stat)
            return True
            
        except Exception as e:
            print(colored(f"Error applying move {move_notation} for {player_info['player_name']}: {e}", "red"))
            return False
    
    def calculate_final_result(self) -> utils.GameStats:
        """Calculate the final game result."""
        if self.pyspiel_state.is_terminal():
            returns = self.pyspiel_state.returns()  # [black_return, white_return]
            black_score, white_score = returns[0], returns[1]
            
            # Map scores to result string (PGN format: white-black)
            score_map = {-1: "0", 0: "1/2", 1: "1"}
            result_string = f"{score_map[white_score]}-{score_map[black_score]}"
            
            # Determine winner
            if white_score > black_score:
                winner = self._map_winning_color_to_model_id(False)  # White wins
                print(colored(f"Game {self.game_number} result: {self.white_name} wins! ({result_string})", "green"))
            elif black_score > white_score:
                winner = self._map_winning_color_to_model_id(True)  # Black wins
                print(colored(f"Game {self.game_number} result: {self.black_name} wins! ({result_string})", "green"))
            else:
                winner = "draw"
                print(colored(f"Game {self.game_number} result: Draw ({result_string})", "yellow"))
        else:
            # Game hit move limit
            winner = "draw"
            result_string = "1/2-1/2"
            print(colored(f"Game {self.game_number} result: Draw by move limit ({result_string})", "yellow"))
        
        print(colored(f"⏰ Final times - {self.white_name}: {utils.format_time(self.white_clock.time_remaining)}, "
                     f"{self.black_name}: {utils.format_time(self.black_clock.time_remaining)}", "blue"))
        
        return self._create_game_stats(winner, result_string)
    
    def _map_winning_color_to_model_id(self, black_wins: bool) -> str:
        """
        Map the winning color to model_a or model_b based on color assignments.
        Returns:
            "model_a" or "model_b"
        """
        # model_a wins if: (black wins and model_a plays black) OR (white wins and model_a plays white)
        # This simplifies to: black_wins XOR model_a_plays_white
        return "model_a" if black_wins != self.model_a_plays_white else "model_b"
    
    def _create_game_stats(self, winner: str, result_string: str) -> utils.GameStats:
        """Create GameStats object with current state."""
        game_duration = time.time() - self.game_start_time
        
        return utils.GameStats(
            game_number=self.game_number,
            winner=winner,
            result_string=result_string,
            model_a_color="white" if self.model_a_plays_white else "black",
            total_moves=self.move_count,
            duration=game_duration,
            move_stats=self.move_stats,
            model_a_final_time=self.white_clock.time_remaining if self.model_a_plays_white else self.black_clock.time_remaining,
            model_b_final_time=self.black_clock.time_remaining if self.model_a_plays_white else self.white_clock.time_remaining,
            model_a_parsing_failures=self.white_parsing_failures if self.model_a_plays_white else self.black_parsing_failures,
            model_b_parsing_failures=self.black_parsing_failures if self.model_a_plays_white else self.white_parsing_failures
        )


def handle_simple_parsing(game_state: GameState, player_info: dict, response, parser, 
                         max_failures: int) -> Tuple[Optional[str], bool, Optional[utils.GameStats]]:
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