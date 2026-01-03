#!/usr/bin/env python3
"""Analysis tools for blitz chess matches."""

# Re-export from stockfish submodule
from game_arena.blitz.analysis.stockfish import MoveQualityAnalyzer, MoveAnalysis

__all__ = [
    "MoveQualityAnalyzer",
    "MoveAnalysis",
]

