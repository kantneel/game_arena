#!/usr/bin/env python3
"""Notebook generation for blitz chess match analysis."""

import json
from pathlib import Path


def create_analysis_notebook(match_id: str, data_dir: str = "_results") -> None:
    """Create a Jupyter notebook for analyzing match data.
    
    Args:
        match_id: The match identifier
        data_dir: Directory where match data is stored
    """
    notebook = _build_notebook_structure(match_id)
    
    notebook_path = Path(data_dir) / match_id / f"{match_id}_analysis.ipynb"
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"📓 Analysis notebook created: {notebook_path}")


def _build_notebook_structure(match_id: str) -> dict:
    """Build the notebook JSON structure."""
    return {
        "cells": [
            _markdown_cell([
                f"# Blitz Chess Game Analysis: {match_id}\n",
                "\n",
                "This notebook provides turn-level analysis of individual games from the blitz chess match."
            ]),
            _code_cell(_get_setup_code()),
            _code_cell(_get_timing_analysis_code()),
            _code_cell(_get_token_analysis_code()),
            _code_cell(_get_move_quality_code()),
            _code_cell(_get_win_probability_code()),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def _markdown_cell(source_lines: list) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }


def _code_cell(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n')
    }


def _get_setup_code() -> str:
    """Get the setup and data loading code."""
    return '''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Load data
data_dir = Path('')

# Load metadata
with open(data_dir / 'metadata.json') as f:
    metadata = json.load(f)

# Load games summary data
games_df = pd.read_csv(data_dir / 'games_summary.csv')

# Load all available game move files
move_files = list(data_dir.glob('game_*_moves.csv'))
print(f"Found {len(move_files)} game move files")

# Check for move quality analysis data
move_analysis_file = data_dir / 'complete_move_analysis.csv'
has_move_analysis = move_analysis_file.exists()
print(f"Move quality analysis available: {has_move_analysis}")

# Load moves from first game for detailed analysis
if move_files:
    game_moves_df = pd.read_csv(move_files[0])
    game_num = move_files[0].stem.split('_')[1]
    print(f"Analyzing Game {game_num} in detail")
else:
    game_moves_df = None
    print("No individual game move data available")

# Load move quality analysis if available
if has_move_analysis:
    move_analysis_df = pd.read_csv(move_analysis_file)
    print(f"Loaded move analysis for {len(move_analysis_df)} moves")
else:
    move_analysis_df = None

print(f"Match: {metadata['match_id']}")
print(f"Models: {metadata['model_a']} vs {metadata['model_b']}")
print(f"Total Games: {metadata['total_games']}")'''


def _get_timing_analysis_code() -> str:
    """Get the timing analysis visualization code."""
    return '''if game_moves_df is not None:
    # Turn-level duration and move analysis
    plt.figure(figsize=(15, 10))
    
    # Separate data by player
    model_a_moves = game_moves_df[game_moves_df['who_played'] == 'Model A']
    model_b_moves = game_moves_df[game_moves_df['who_played'] == 'Model B']
    
    # Move time over the course of the game
    plt.subplot(2, 2, 1)
    plt.plot(model_a_moves['move_number'], model_a_moves['time_taken_seconds'], 
             'o-', label=f"{metadata['model_a']} (Model A)", color='blue', alpha=0.7)
    plt.plot(model_b_moves['move_number'], model_b_moves['time_taken_seconds'], 
             's-', label=f"{metadata['model_b']} (Model B)", color='red', alpha=0.7)
    plt.xlabel('Move Number')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Move Time Throughout Game')
    plt.legend()
    
    # Time remaining over the course of the game
    plt.subplot(2, 2, 2)
    plt.plot(model_a_moves['move_number'], model_a_moves['time_available_at_turn_start'], 
             'o-', label=f"{metadata['model_a']} (Model A)", color='blue', alpha=0.7)
    plt.plot(model_b_moves['move_number'], model_b_moves['time_available_at_turn_start'], 
             's-', label=f"{metadata['model_b']} (Model B)", color='red', alpha=0.7)
    plt.xlabel('Move Number')
    plt.ylabel('Time Remaining (seconds)')
    plt.title('Time Bank Throughout Game')
    plt.legend()
    
    # 2D scatter: Time remaining vs time taken
    plt.subplot(2, 2, 3)
    plt.scatter(model_a_moves['time_available_at_turn_start'], 
                model_a_moves['time_taken_seconds'], 
                alpha=0.7, color='blue', s=50, label=f"{metadata['model_a']}")
    plt.scatter(model_b_moves['time_available_at_turn_start'], 
                model_b_moves['time_taken_seconds'], 
                alpha=0.7, color='red', s=50, label=f"{metadata['model_b']}")
    plt.xlabel('Time Remaining at Turn Start (seconds)')
    plt.ylabel('Time Taken for Move (seconds)')
    plt.title('Time Remaining vs Time Taken')
    plt.legend()
    
    # Box plot comparison
    plt.subplot(2, 2, 4)
    plt.boxplot([model_a_moves['time_taken_seconds'], model_b_moves['time_taken_seconds']], 
               labels=[f"{metadata['model_a']}", f"{metadata['model_b']}"])
    plt.ylabel('Time Taken per Move (seconds)')
    plt.title('Move Time Distribution Comparison')
    
    plt.tight_layout()
    plt.show()
else:
    print("No move-level data available for detailed analysis")'''


def _get_token_analysis_code() -> str:
    """Get the token analysis code."""
    return '''if game_moves_df is not None:
    # Token analysis
    has_thinking_tokens = 'thinking_tokens' in game_moves_df.columns and game_moves_df['thinking_tokens'].notna().any()
    has_output_tokens = 'output_tokens' in game_moves_df.columns and game_moves_df['output_tokens'].notna().any()
    
    if has_thinking_tokens or has_output_tokens:
        plt.figure(figsize=(12, 6))
        
        token_col = 'thinking_tokens' if has_thinking_tokens else 'output_tokens'
        token_label = 'Thinking Tokens' if has_thinking_tokens else 'Output Tokens'
        
        model_a_tokens = model_a_moves[model_a_moves[token_col].notna()]
        model_b_tokens = model_b_moves[model_b_moves[token_col].notna()]
        
        if not model_a_tokens.empty and not model_b_tokens.empty:
            plt.subplot(1, 2, 1)
            plt.scatter(model_a_tokens[token_col], model_a_tokens['time_taken_seconds'], 
                       alpha=0.7, color='blue', label=f"{metadata['model_a']}")
            plt.scatter(model_b_tokens[token_col], model_b_tokens['time_taken_seconds'], 
                       alpha=0.7, color='red', label=f"{metadata['model_b']}")
            plt.xlabel(token_label)
            plt.ylabel('Time Taken (seconds)')
            plt.title(f'{token_label} vs Move Time')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(model_a_tokens['move_number'], model_a_tokens[token_col], 
                     'o-', label=f"{metadata['model_a']}", color='blue', alpha=0.7)
            plt.plot(model_b_tokens['move_number'], model_b_tokens[token_col], 
                     's-', label=f"{metadata['model_b']}", color='red', alpha=0.7)
            plt.xlabel('Move Number')
            plt.ylabel(token_label)
            plt.title(f'{token_label} Throughout Game')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    else:
        print("No token data available")
    
    # Summary statistics
    print("\\n=== GAME SUMMARY STATISTICS ===")
    print(f"Model A ({metadata['model_a']}) - Total moves: {len(model_a_moves)}")
    print(f"  Average move time: {model_a_moves['time_taken_seconds'].mean():.2f} seconds")
    print(f"  Total time used: {model_a_moves['time_taken_seconds'].sum():.2f} seconds")
    
    print(f"\\nModel B ({metadata['model_b']}) - Total moves: {len(model_b_moves)}")
    print(f"  Average move time: {model_b_moves['time_taken_seconds'].mean():.2f} seconds")
    print(f"  Total time used: {model_b_moves['time_taken_seconds'].sum():.2f} seconds")'''


def _get_move_quality_code() -> str:
    """Get the move quality analysis code."""
    return '''# Move Quality Analysis (if available)
if move_analysis_df is not None and not move_analysis_df.empty:
    print("\\n=== MOVE QUALITY ANALYSIS ===")
    print(f"Analyzing {len(move_analysis_df)} moves with Stockfish evaluation")
    
    model_a_analysis = move_analysis_df[move_analysis_df['player'] == 'Model A']
    model_b_analysis = move_analysis_df[move_analysis_df['player'] == 'Model B']
    
    if not model_a_analysis.empty and not model_b_analysis.empty:
        plt.figure(figsize=(12, 8))
        
        # Centipawn loss over time
        plt.subplot(2, 2, 1)
        plt.plot(model_a_analysis['move_number'], model_a_analysis['centipawn_loss'], 
                 'o-', label=f"{metadata['model_a']}", color='blue', alpha=0.7)
        plt.plot(model_b_analysis['move_number'], model_b_analysis['centipawn_loss'], 
                 's-', label=f"{metadata['model_b']}", color='red', alpha=0.7)
        plt.xlabel('Move Number')
        plt.ylabel('Centipawn Loss')
        plt.title('Move Quality Throughout Game')
        plt.legend()
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.5)
        
        # Distribution
        plt.subplot(2, 2, 2)
        plt.hist(model_a_analysis['centipawn_loss'], bins=20, alpha=0.7, 
                 label=f"{metadata['model_a']}", color='blue')
        plt.hist(model_b_analysis['centipawn_loss'], bins=20, alpha=0.7, 
                 label=f"{metadata['model_b']}", color='red')
        plt.xlabel('Centipawn Loss')
        plt.ylabel('Frequency')
        plt.title('Distribution of Move Quality')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\\n{metadata['model_a']} - Avg centipawn loss: {model_a_analysis['centipawn_loss'].mean():.1f}")
        print(f"{metadata['model_b']} - Avg centipawn loss: {model_b_analysis['centipawn_loss'].mean():.1f}")
else:
    print("\\nMove quality analysis not available")
    print("To generate, run: python -m game_arena.blitz.analysis.stockfish.analyzer <match_directory>")'''


def _get_win_probability_code() -> str:
    """Get the win probability analysis code."""
    return '''# Win Probability Analysis
if move_analysis_df is not None and 'played_win_probability' in move_analysis_df.columns:
    print("\\n=== WIN PROBABILITY ANALYSIS ===")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(model_a_analysis['move_number'], model_a_analysis['played_win_probability'] * 100, 
             'o-', label=f"{metadata['model_a']}", color='blue', alpha=0.7)
    plt.plot(model_b_analysis['move_number'], model_b_analysis['played_win_probability'] * 100, 
             's-', label=f"{metadata['model_b']}", color='red', alpha=0.7)
    plt.xlabel('Move Number')
    plt.ylabel('Win Probability (%)')
    plt.title('Win Probability Throughout Game')
    plt.legend()
    plt.ylim(0, 100)
    
    plt.subplot(1, 2, 2)
    plt.hist(model_a_analysis['win_probability_loss'] * 100, bins=20, alpha=0.7, 
             label=f"{metadata['model_a']}", color='blue')
    plt.hist(model_b_analysis['win_probability_loss'] * 100, bins=20, alpha=0.7, 
             label=f"{metadata['model_b']}", color='red')
    plt.xlabel('Win Probability Loss (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Win Probability Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("\\nWin probability data not available")'''

