#!/usr/bin/env python3
"""
Blitz Chess Arena - LLM vs LLM real-time chess with time management.

This module provides infrastructure for running blitz chess matches between
AI models, with comprehensive time management, data collection, and analysis.

Submodules:
    core: Core game mechanics (clock, game state, types)
    models: Model handling (wrappers, registry, calibration)
    prompts: Prompt engineering (time-aware, dramatic prompts)
    data: Data collection and storage
    display: Output and visualization
    analysis: Post-game analysis (Stockfish integration)
    tournament: Multi-model tournament system
"""

# Re-export commonly used classes and functions for convenience
from game_arena.blitz.core import (
    PlayerClock,
    GameState,
    MoveStats,
    GameStats,
    handle_simple_parsing,
)

from game_arena.blitz.models import (
    NoRetryModelWrapper,
    BlitzModelWrapper,
    get_model_from_registry,
    calibrate_network_latency,
    handle_rethinking_move,
)

from game_arena.blitz.prompts import (
    create_time_aware_prompt_substitutions,
    create_dramatic_time_pressure_text,
    create_dramatic_instruction_text,
)

from game_arena.blitz.data import (
    BlitzDataCollector,
    get_data_collector,
    create_analysis_notebook,
    MatchMetadata,
    GameRecord,
    GameMoveRecord,
)

from game_arena.blitz.display import (
    format_time,
    abbreviate_model_name,
    print_detailed_game_analysis,
    print_comprehensive_match_analysis,
    display_reasoning_traces,
)

__all__ = [
    # Core
    "PlayerClock",
    "GameState", 
    "MoveStats",
    "GameStats",
    "handle_simple_parsing",
    # Models
    "NoRetryModelWrapper",
    "BlitzModelWrapper",
    "get_model_from_registry",
    "calibrate_network_latency",
    "handle_rethinking_move",
    # Prompts
    "create_time_aware_prompt_substitutions",
    "create_dramatic_time_pressure_text",
    "create_dramatic_instruction_text",
    # Data
    "BlitzDataCollector",
    "get_data_collector",
    "create_analysis_notebook",
    "MatchMetadata",
    "GameRecord",
    "GameMoveRecord",
    # Display
    "format_time",
    "abbreviate_model_name",
    "print_detailed_game_analysis",
    "print_comprehensive_match_analysis",
    "display_reasoning_traces",
]
