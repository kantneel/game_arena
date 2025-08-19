#!/usr/bin/env python3
"""Advanced analysis tools for blitz chess tournaments."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional
import datetime

from game_arena.blitz import data_collector


def analyze_time_pressure_behavior(moves_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how models respond to time pressure."""
    
    # Categorize moves by time pressure
    time_pressure_analysis = {}
    
    for pressure_level in ['LOW', 'MEDIUM', 'HIGH']:
        pressure_moves = moves_df[moves_df['time_pressure_level'] == pressure_level]
        
        if len(pressure_moves) == 0:
            continue
            
        time_pressure_analysis[pressure_level] = {
            'avg_thinking_time': pressure_moves['thinking_time'].mean(),
            'avg_total_tokens': pressure_moves['total_tokens'].mean(),
            'avg_reasoning_tokens': pressure_moves['reasoning_tokens'].mean(),
            'move_count': len(pressure_moves),
            'avg_time_remaining': pressure_moves['time_remaining_after'].mean(),
            'token_efficiency': pressure_moves['total_tokens'].sum() / pressure_moves['thinking_time'].sum() if pressure_moves['thinking_time'].sum() > 0 else 0
        }
    
    return time_pressure_analysis


def analyze_tactical_time_usage(moves_df: pd.DataFrame, games_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze whether models use time tactically to pressure opponents."""
    
    tactical_analysis = {}
    
    # Group moves by game and analyze sequences
    for game_num in moves_df['game_number'].unique():
        game_moves = moves_df[moves_df['game_number'] == game_num].sort_values('move_in_game')
        
        if len(game_moves) < 4:  # Need at least 4 moves to analyze patterns
            continue
        
        # Analyze if players deliberately use more time when opponent is under pressure
        for player in game_moves['player'].unique():
            player_moves = game_moves[game_moves['player'] == player]
            opponent_moves = game_moves[game_moves['player'] != player]
            
            # Check if player uses more time when opponent has less time remaining
            correlations = []
            
            for i, move in player_moves.iterrows():
                # Find the opponent's previous move
                prev_opponent_moves = opponent_moves[opponent_moves['move_in_game'] < move['move_in_game']]
                if len(prev_opponent_moves) > 0:
                    last_opponent_move = prev_opponent_moves.iloc[-1]
                    
                    # Correlation between opponent's time pressure and player's time usage
                    opponent_time_pressure = 1.0 / max(last_opponent_move['time_remaining_after'], 1.0)  # Inverse of time = pressure
                    player_time_usage = move['thinking_time']
                    
                    correlations.append((opponent_time_pressure, player_time_usage))
            
            if len(correlations) > 3:
                pressures, usages = zip(*correlations)
                correlation = np.corrcoef(pressures, usages)[0, 1] if len(set(pressures)) > 1 else 0.0
                
                tactical_analysis[f"game_{game_num}_{player}"] = {
                    'time_usage_vs_opponent_pressure_correlation': correlation,
                    'avg_thinking_time': player_moves['thinking_time'].mean(),
                    'moves_analyzed': len(correlations)
                }
    
    return tactical_analysis


def analyze_cross_match_performance(tournament_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze how models perform against different opponents across matches."""
    
    match_results_df = tournament_data['match_results']
    
    performance_matrix = {}
    
    # Get all unique models
    all_models = set()
    for _, match in match_results_df.iterrows():
        all_models.add(match['model1'])
        all_models.add(match['model2'])
    
    all_models = sorted(list(all_models))
    
    # Create performance matrix
    for model in all_models:
        performance_matrix[model] = {
            'matches_played': 0,
            'matches_won': 0,
            'total_games_won': 0,
            'total_games_played': 0,
            'opponents_defeated': [],
            'opponents_lost_to': [],
            'avg_match_duration': 0,
            'performance_by_round': {}
        }
    
    # Populate performance data
    for _, match in match_results_df.iterrows():
        model1, model2 = match['model1'], match['model2']
        winner, loser = match['winner'], match['loser']
        
        # Update for model1
        performance_matrix[model1]['matches_played'] += 1
        performance_matrix[model1]['total_games_played'] += match['games_played']
        
        if winner == model1:
            performance_matrix[model1]['matches_won'] += 1
            performance_matrix[model1]['total_games_won'] += match['score_model1']
            performance_matrix[model1]['opponents_defeated'].append(model2)
        else:
            performance_matrix[model1]['total_games_won'] += match['score_model1']
            performance_matrix[model1]['opponents_lost_to'].append(model2)
        
        # Update for model2
        performance_matrix[model2]['matches_played'] += 1
        performance_matrix[model2]['total_games_played'] += match['games_played']
        
        if winner == model2:
            performance_matrix[model2]['matches_won'] += 1
            performance_matrix[model2]['total_games_won'] += match['score_model2']
            performance_matrix[model2]['opponents_defeated'].append(model1)
        else:
            performance_matrix[model2]['total_games_won'] += match['score_model2']
            performance_matrix[model2]['opponents_lost_to'].append(model1)
        
        # Round-specific performance
        round_name = match['round_name']
        for model in [model1, model2]:
            if round_name not in performance_matrix[model]['performance_by_round']:
                performance_matrix[model]['performance_by_round'][round_name] = {
                    'matches': 0, 'wins': 0
                }
            performance_matrix[model]['performance_by_round'][round_name]['matches'] += 1
            
            if match['winner'] == model:
                performance_matrix[model]['performance_by_round'][round_name]['wins'] += 1
    
    # Calculate derived statistics
    for model in performance_matrix:
        data = performance_matrix[model]
        if data['matches_played'] > 0:
            data['match_win_rate'] = data['matches_won'] / data['matches_played']
            data['game_win_rate'] = data['total_games_won'] / data['total_games_played'] if data['total_games_played'] > 0 else 0
    
    return performance_matrix


def create_tournament_visualizations(tournament_data: Dict[str, Any], output_dir: Path):
    """Create comprehensive visualizations for tournament analysis."""
    
    output_dir.mkdir(exist_ok=True)
    
    match_results_df = tournament_data['match_results']
    metadata = tournament_data['metadata']
    
    # 1. Tournament Bracket Visualization
    plt.figure(figsize=(16, 10))
    
    # Extract bracket information
    bracket = tournament_data['bracket']
    
    # Create bracket diagram
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Quarterfinals
    qf_results = [r for r in match_results_df.iterrows() if r[1]['round_name'] == 'quarterfinals']
    for i, (_, match) in enumerate(qf_results):
        y_pos = 7 - i * 2
        plt.text(1, y_pos, f"{match['model1']}\nvs\n{match['model2']}", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.text(2.5, y_pos, f"Winner:\n{match['winner']}\n({match['score_model1']}-{match['score_model2']})", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Semifinals
    sf_results = [r for r in match_results_df.iterrows() if r[1]['round_name'] == 'semifinals']
    for i, (_, match) in enumerate(sf_results):
        y_pos = 5.5 - i * 3
        plt.text(4, y_pos, f"{match['model1']}\nvs\n{match['model2']}", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.text(5.5, y_pos, f"Winner:\n{match['winner']}\n({match['score_model1']}-{match['score_model2']})", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Finals
    final_result = [r for r in match_results_df.iterrows() if r[1]['round_name'] == 'finals'][0][1]
    plt.text(7, 4, f"FINALS\n{final_result['model1']}\nvs\n{final_result['model2']}", 
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    plt.text(8.5, 4, f"CHAMPION:\n{final_result['winner']}\n({final_result['score_model1']}-{final_result['score_model2']})", 
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="gold"))
    
    plt.title(f"Tournament Bracket: {metadata['tournament_id']}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "tournament_bracket.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Match Duration Analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(match_results_df)), match_results_df['match_duration_seconds'] / 60)
    plt.xlabel('Match Number')
    plt.ylabel('Duration (minutes)')
    plt.title('Match Durations')
    plt.xticks(range(len(match_results_df)), 
               [f"{r['round_name'][:2].upper()}{r['match_number']}" for _, r in match_results_df.iterrows()])
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(match_results_df)), match_results_df['games_played'])
    plt.xlabel('Match Number')
    plt.ylabel('Games Played')
    plt.title('Games per Match')
    plt.xticks(range(len(match_results_df)), 
               [f"{r['round_name'][:2].upper()}{r['match_number']}" for _, r in match_results_df.iterrows()])
    
    plt.tight_layout()
    plt.savefig(output_dir / "match_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_tournament_analysis_notebook(tournament_id: str, data_dir: str = "tournament_data"):
    """Create a comprehensive Jupyter notebook for tournament analysis."""
    
    notebook_content = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# Blitz Chess Tournament Analysis: {tournament_id}\\n",
    "\\n",
    "This notebook provides comprehensive analysis of the tournament data including:\\n",
    "- Tournament bracket and results\\n",
    "- Cross-match performance analysis\\n",
    "- Time pressure behavior analysis\\n",
    "- Tactical time usage patterns\\n",
    "- Token efficiency analysis"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import json\\n",
    "import numpy as np\\n",
    "from pathlib import Path\\n",
    "import sys\\n",
    "\\n",
    "# Add the game_arena path\\n",
    "sys.path.append('../..')\\n",
    "from game_arena.blitz.tournament.analysis import tournament_analysis\\n",
    "from game_arena.blitz.tournament import tournament_manager\\n",
    "\\n",
    "# Load tournament data\\n",
    "tournament_data = tournament_manager.load_tournament_data('{tournament_id}', '{data_dir}')\\n",
    "\\n",
    "metadata = tournament_data['metadata']\\n",
    "bracket = tournament_data['bracket']\\n",
    "match_results_df = tournament_data['match_results']\\n",
    "\\n",
    "print(f\\"Tournament: {{metadata['tournament_id']}}\\")\\n",
    "print(f\\"Winner: {{metadata['winner']}}\\")\\n",
    "print(f\\"Total Matches: {{metadata['total_matches']}}\\")\\n",
    "print(f\\"Total Games: {{metadata['total_games']}}\\")\\n",
    "print(f\\"Duration: {{metadata['tournament_duration_seconds']/60:.1f}} minutes\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Tournament Bracket Results\\n",
    "print(\\"TOURNAMENT RESULTS:\\")\\n",
    "print(\\"=\\" * 50)\\n",
    "\\n",
    "for round_name in ['quarterfinals', 'semifinals', 'finals']:\\n",
    "    round_matches = match_results_df[match_results_df['round_name'] == round_name]\\n",
    "    print(f\\"\\\\n{{round_name.upper()}}:\\")\\n",
    "    \\n",
    "    for _, match in round_matches.iterrows():\\n",
    "        print(f\\"  {{match['model1']}} vs {{match['model2']}} -> {{match['winner']}} wins ({{match['score_model1']}}-{{match['score_model2']}})\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Cross-match performance analysis\\n",
    "performance_matrix = tournament_analysis.analyze_cross_match_performance(tournament_data)\\n",
    "\\n",
    "# Create performance summary\\n",
    "perf_summary = []\\n",
    "for model, stats in performance_matrix.items():\\n",
    "    perf_summary.append({{\\n",
    "        'Model': model,\\n",
    "        'Matches_Played': stats['matches_played'],\\n",
    "        'Matches_Won': stats['matches_won'],\\n",
    "        'Match_Win_Rate': stats.get('match_win_rate', 0),\\n",
    "        'Game_Win_Rate': stats.get('game_win_rate', 0),\\n",
    "        'Games_Won': stats['total_games_won'],\\n",
    "        'Games_Played': stats['total_games_played']\\n",
    "    }})\\n",
    "\\n",
    "perf_df = pd.DataFrame(perf_summary)\\n",
    "print(\\"MODEL PERFORMANCE SUMMARY:\\")\\n",
    "print(perf_df.to_string(index=False))"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Load individual match data for detailed analysis\\n",
    "all_games_data = []\\n",
    "all_moves_data = []\\n",
    "\\n",
    "for _, match in match_results_df.iterrows():\\n",
    "    # Load each match's detailed data\\n",
    "    try:\\n",
    "        match_dir = Path(match['data_directory'])\\n",
    "        if (match_dir / 'games.csv').exists():\\n",
    "            games_df = pd.read_csv(match_dir / 'games.csv')\\n",
    "            games_df['tournament_match_id'] = match['match_id']\\n",
    "            games_df['tournament_round'] = match['round_name']\\n",
    "            all_games_data.append(games_df)\\n",
    "            \\n",
    "        if (match_dir / 'moves.csv').exists():\\n",
    "            moves_df = pd.read_csv(match_dir / 'moves.csv')\\n",
    "            moves_df['tournament_match_id'] = match['match_id']\\n",
    "            moves_df['tournament_round'] = match['round_name']\\n",
    "            all_moves_data.append(moves_df)\\n",
    "    except Exception as e:\\n",
    "        print(f\\"Warning: Could not load data for {{match['match_id']}}: {{e}}\\")\\n",
    "        continue\\n",
    "\\n",
    "if all_games_data:\\n",
    "    combined_games_df = pd.concat(all_games_data, ignore_index=True)\\n",
    "    combined_moves_df = pd.concat(all_moves_data, ignore_index=True)\\n",
    "    print(f\\"Loaded {{len(combined_games_df)}} games and {{len(combined_moves_df)}} moves\\")\\n",
    "else:\\n",
    "    print(\\"Warning: No detailed game data found. Analysis will be limited to match results.\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Time pressure behavior analysis (if detailed data available)\\n",
    "if 'combined_moves_df' in locals():\\n",
    "    time_pressure_analysis = tournament_analysis.analyze_time_pressure_behavior(combined_moves_df)\\n",
    "    \\n",
    "    print(\\"TIME PRESSURE BEHAVIOR ANALYSIS:\\")\\n",
    "    print(\\"=\\" * 50)\\n",
    "    \\n",
    "    for pressure_level, stats in time_pressure_analysis.items():\\n",
    "        print(f\\"\\\\n{{pressure_level}} PRESSURE:\\")\\n",
    "        print(f\\"  Moves: {{stats['move_count']}}\\")\\n",
    "        print(f\\"  Avg Thinking Time: {{stats['avg_thinking_time']:.2f}}s\\")\\n",
    "        print(f\\"  Avg Total Tokens: {{stats['avg_total_tokens']:.0f}}\\")\\n",
    "        print(f\\"  Token Efficiency: {{stats['token_efficiency']:.1f}} tokens/sec\\")\\n",
    "        print(f\\"  Avg Time Remaining: {{stats['avg_time_remaining']:.0f}}s\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Tactical time usage analysis\\n",
    "if 'combined_moves_df' in locals() and 'combined_games_df' in locals():\\n",
    "    tactical_analysis = tournament_analysis.analyze_tactical_time_usage(combined_moves_df, combined_games_df)\\n",
    "    \\n",
    "    print(\\"TACTICAL TIME USAGE ANALYSIS:\\")\\n",
    "    print(\\"=\\" * 50)\\n",
    "    print(\\"Correlation between opponent time pressure and player thinking time:\\")\\n",
    "    \\n",
    "    correlations_by_player = {{}}\\n",
    "    for game_player, stats in tactical_analysis.items():\\n",
    "        player = game_player.split('_')[-1]\\n",
    "        if player not in correlations_by_player:\\n",
    "            correlations_by_player[player] = []\\n",
    "        correlations_by_player[player].append(stats['time_usage_vs_opponent_pressure_correlation'])\\n",
    "    \\n",
    "    for player, correlations in correlations_by_player.items():\\n",
    "        avg_correlation = np.mean(correlations)\\n",
    "        print(f\\"  {{player}}: {{avg_correlation:.3f}} ({{len(correlations)}} games)\\")\\n",
    "        if avg_correlation > 0.3:\\n",
    "            print(f\\"    -> {{player}} tends to use MORE time when opponent is under pressure\\")\\n",
    "        elif avg_correlation < -0.3:\\n",
    "            print(f\\"    -> {{player}} tends to use LESS time when opponent is under pressure\\")\\n",
    "        else:\\n",
    "            print(f\\"    -> {{player}} shows no clear tactical time usage pattern\\")"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Create visualizations\\n",
    "if 'combined_moves_df' in locals():\\n",
    "    # Time pressure vs token usage\\n",
    "    plt.figure(figsize=(15, 5))\\n",
    "    \\n",
    "    plt.subplot(1, 3, 1)\\n",
    "    sns.boxplot(data=combined_moves_df, x='time_pressure_level', y='thinking_time')\\n",
    "    plt.title('Thinking Time by Pressure Level')\\n",
    "    plt.ylabel('Thinking Time (seconds)')\\n",
    "    \\n",
    "    plt.subplot(1, 3, 2)\\n",
    "    # Use output_tokens if available, fallback to total_tokens\\n",
    "    token_col = 'output_tokens' if 'output_tokens' in combined_moves_df.columns else 'total_tokens'\\n",
    "    sns.boxplot(data=combined_moves_df, x='time_pressure_level', y=token_col)\\n",
    "    plt.title('Token Usage by Pressure Level')\\n",
    "    plt.ylabel('Output Tokens' if token_col == 'output_tokens' else 'Total Tokens')\\n",
    "    \\n",
    "    plt.subplot(1, 3, 3)\\n",
    "    # Token efficiency (tokens per second)\\n",
    "    efficiency_token_col = 'output_tokens' if 'output_tokens' in combined_moves_df.columns else 'total_tokens'\\n",
    "    combined_moves_df['token_efficiency'] = combined_moves_df[efficiency_token_col] / combined_moves_df['thinking_time']\\n",
    "    sns.boxplot(data=combined_moves_df, x='time_pressure_level', y='token_efficiency')\\n",
    "    plt.title('Token Efficiency by Pressure Level')\\n",
    "    plt.ylabel('Output Tokens per Second' if efficiency_token_col == 'output_tokens' else 'Tokens per Second')\\n",
    "    \\n",
    "    plt.tight_layout()\\n",
    "    plt.show()"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''
    
    notebook_path = Path(data_dir) / tournament_id / f"{tournament_id}_analysis.ipynb"
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print(f"📓 Tournament analysis notebook created: {notebook_path}")


def generate_tournament_report(tournament_id: str, data_dir: str = "tournament_data") -> str:
    """Generate a comprehensive text report of tournament results."""
    
    tournament_data = tournament_manager.load_tournament_data(tournament_id, data_dir)
    metadata = tournament_data['metadata']
    match_results_df = tournament_data['match_results']
    
    report = f"""
BLITZ CHESS TOURNAMENT REPORT
{metadata['tournament_id']}
{'='*60}

TOURNAMENT OVERVIEW:
- Start Time: {metadata['start_time']}
- End Time: {metadata['end_time']}
- Duration: {metadata['tournament_duration_seconds']/60:.1f} minutes
- Format: {metadata['tournament_type']}
- Time Control: {metadata['time_control']}
- Total Matches: {metadata['total_matches']}
- Total Games: {metadata['total_games']}

CHAMPION: {metadata['winner']}

MATCH RESULTS:
"""
    
    for round_name in ['quarterfinals', 'semifinals', 'finals']:
        round_matches = match_results_df[match_results_df['round_name'] == round_name]
        report += f"\n{round_name.upper()}:\n"
        
        for _, match in round_matches.iterrows():
            duration_min = match['match_duration_seconds'] / 60
            report += f"  {match['model1']} vs {match['model2']}\n"
            report += f"    Winner: {match['winner']} ({match['score_model1']}-{match['score_model2']}) in {match['games_played']} games\n"
            report += f"    Duration: {duration_min:.1f} minutes\n\n"
    
    # Performance analysis
    performance_matrix = analyze_cross_match_performance(tournament_data)
    
    report += "\nMODEL PERFORMANCE SUMMARY:\n"
    report += "-" * 40 + "\n"
    
    for model, stats in performance_matrix.items():
        if stats['matches_played'] > 0:
            report += f"\n{model}:\n"
            report += f"  Matches: {stats['matches_won']}/{stats['matches_played']} ({stats.get('match_win_rate', 0)*100:.1f}%)\n"
            report += f"  Games: {stats['total_games_won']}/{stats['total_games_played']} ({stats.get('game_win_rate', 0)*100:.1f}%)\n"
            
            if stats['opponents_defeated']:
                report += f"  Defeated: {', '.join(stats['opponents_defeated'])}\n"
            if stats['opponents_lost_to']:
                report += f"  Lost to: {', '.join(stats['opponents_lost_to'])}\n"
    
    return report 