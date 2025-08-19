#!/usr/bin/env python3
"""Flag definitions for blitz chess match."""

from absl import flags
from game_arena.harness import tournament_util

_MODEL_A = flags.DEFINE_string(
    "model_a",
    "claude-sonnet-4",
    "Model A to use.",
)

_MODEL_B = flags.DEFINE_string(
    "model_b",
    "claude-opus-4",
    "Model B to use.",
)

_PARSER_CHOICE = flags.DEFINE_enum_class(
    "parser_choice",
    tournament_util.ParserChoice.RULE_THEN_SOFT,
    tournament_util.ParserChoice,
    "Move parser to use.",
)

_MAX_MOVES_PER_GAME = flags.DEFINE_integer(
    "max_moves_per_game",
    200,
    "Maximum number of moves per game to prevent infinite games.",
)

_INITIAL_TIME_SECONDS = flags.DEFINE_integer(
    "initial_time_seconds",
    300,  # 5 minutes
    "Initial time per player in seconds (blitz format).",
)

_INCREMENT_SECONDS = flags.DEFINE_integer(
    "increment_seconds",
    3,
    "Time increment per move in seconds.",
)

_CALIBRATION_ROUNDS = flags.DEFINE_integer(
    "calibration_rounds",
    3,
    "Number of calibration rounds to measure network latency.",
)

_SKIP_CALIBRATION = flags.DEFINE_bool(
    "skip_calibration",
    False,
    "Skip network latency calibration and use default latency estimates.",
)

_MAX_PARSING_FAILURES = flags.DEFINE_integer(
    "max_parsing_failures",
    3,
    "Maximum number of parsing failures allowed per player before ending the game.",
)

_USE_RETHINKING = flags.DEFINE_bool(
    "use_rethinking",
    True,
    "Enable rethinking when parsing fails.",
)

_RETHINK_STRATEGY = flags.DEFINE_enum_class(
    "rethink_strategy",
    tournament_util.SamplerChoice.RETHINK_WITH_ENV,
    tournament_util.SamplerChoice,
    "Rethinking strategy to use.",
)

_MAX_RETHINKS = flags.DEFINE_integer(
    "max_rethinks",
    2,
    "Maximum number of rethinking attempts (kept low for blitz).",
)

_SHOW_REASONING_TRACES = flags.DEFINE_bool(
    "show_reasoning_traces",
    False,
    "Show model reasoning traces during games.",
)

_REASONING_BUDGET = flags.DEFINE_integer(
    "reasoning_budget",
    8000,
    "Token budget for model reasoning (Gemini only, kept low for blitz).",
)

_FIRST_TO = flags.DEFINE_integer(
    "first_to",
    1,
    "Number of games needed to win the match (e.g., 4 for best-of-7).",
)

_VERIFICATION_ONLY = flags.DEFINE_bool(
    "verification_only",
    False,
    "Run verification test only instead of full match.",
)

_RUN_MOVE_ANALYSIS = flags.DEFINE_bool(
    "run_move_analysis",
    True,
    "Automatically run Stockfish move quality analysis after games complete.",
)

_MOVE_ANALYSIS_DEPTH = flags.DEFINE_integer(
    "move_analysis_depth",
    15,
    "Stockfish search depth for move analysis (higher = more accurate but slower).",
)

_MOVE_ANALYSIS_MULTIPV = flags.DEFINE_integer(
    "move_analysis_multipv",
    3,
    "Number of principal variations to analyze for move ranking.",
) 