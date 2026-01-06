#!/usr/bin/env python3
"""Analysis and visualization for offline evaluation results."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class AggregatedStats:
    """Aggregated statistics for a condition."""
    
    # Identifiers
    model_id: str
    time_level: float
    category: Optional[str] = None
    
    # Sample counts
    n_samples: int = 0
    n_positions: int = 0
    
    # Response time stats
    mean_response_time: float = 0.0
    std_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    
    # Thinking token stats
    mean_thinking_tokens: int = 0
    std_thinking_tokens: float = 0.0
    
    # Quality stats (if available)
    mean_centipawn_loss: Optional[float] = None
    best_move_rate: Optional[float] = None
    blunder_rate: Optional[float] = None


def aggregate_results(results: list, group_by: list[str] = None) -> pd.DataFrame:
    """Aggregate evaluation results into a DataFrame.
    
    Args:
        results: List of EvaluationResult objects or dicts
        group_by: Columns to group by (default: model_id, time_remaining)
        
    Returns:
        DataFrame with aggregated statistics
    """
    if not results:
        return pd.DataFrame()
    
    # Convert to dicts if needed
    if hasattr(results[0], '__dict__'):
        results = [vars(r) if hasattr(r, '__dict__') else r for r in results]
    
    df = pd.DataFrame(results)
    
    if group_by is None:
        group_by = ['model_id', 'time_remaining']
    
    # Aggregate
    agg_funcs = {
        'response_time_seconds': ['mean', 'std', 'min', 'max', 'count'],
        'thinking_tokens': ['mean', 'std'],
        'output_tokens': ['mean'],
    }
    
    # Add quality metrics if present
    if 'centipawn_loss' in df.columns:
        agg_funcs['centipawn_loss'] = ['mean', 'std']
    if 'is_best_move' in df.columns:
        agg_funcs['is_best_move'] = ['mean']
    if 'is_blunder' in df.columns:
        agg_funcs['is_blunder'] = ['mean']
    
    grouped = df.groupby(group_by).agg(agg_funcs)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    return grouped.reset_index()


class OfflineAnalyzer:
    """Analyzer for offline evaluation sessions."""
    
    def __init__(self, sessions: list = None, session_dir: Path = None, include_partial: bool = True):
        """Initialize analyzer.
        
        Args:
            sessions: List of EvaluationSession objects
            session_dir: Directory containing session JSON files
            include_partial: Whether to include in-progress/interrupted sessions
        """
        self.sessions = sessions or []
        self.include_partial = include_partial
        
        if session_dir:
            self._load_sessions(session_dir)
    
    def _load_sessions(self, session_dir: Path):
        """Load all sessions from directory."""
        from game_arena.blitz.offline_eval.evaluator import EvaluationSession
        
        session_dir = Path(session_dir)
        for path in session_dir.glob("*.json"):
            # Skip backup files
            if ".partial." in path.name:
                continue
                
            try:
                session = EvaluationSession.load(path)
                
                # Check if we should include partial sessions
                if not self.include_partial and session.status != "completed":
                    print(f"Skipping partial: {session.session_id} ({session.status})")
                    continue
                
                self.sessions.append(session)
                status_icon = "✅" if session.status == "completed" else "⏳" if session.status == "in_progress" else "⚠️"
                print(f"{status_icon} Loaded: {session.session_id} ({len(session.results)} results, {session.status})")
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    def get_session_status(self) -> pd.DataFrame:
        """Get status summary of all loaded sessions."""
        data = []
        for session in self.sessions:
            stats = session.get_completion_stats()
            data.append({
                "session_id": session.session_id,
                "model_id": session.model_id,
                "status": session.status,
                "completed": stats["completed"],
                "expected": stats["expected"],
                "completion_pct": stats["completion_pct"],
                "start_time": session.start_time,
                "end_time": session.end_time,
            })
        return pd.DataFrame(data)
    
    def watch_live(self, session_path: Path, refresh_interval: float = 5.0):
        """Watch a running session and print live updates.
        
        Args:
            session_path: Path to session JSON file
            refresh_interval: Seconds between refreshes
        """
        import time
        from game_arena.blitz.offline_eval.evaluator import EvaluationSession
        
        print(f"👀 Watching: {session_path}")
        print(f"   Refresh interval: {refresh_interval}s")
        print(f"   Press Ctrl+C to stop\n")
        
        last_count = 0
        try:
            while True:
                try:
                    session = EvaluationSession.load(session_path)
                    current_count = len(session.results)
                    
                    if current_count != last_count:
                        # New results since last check
                        new_results = current_count - last_count
                        last_count = current_count
                        
                        # Get latest result
                        if session.results:
                            latest = session.results[-1]
                            print(f"[{current_count}] {latest.position_id} @ {latest.time_remaining}s: "
                                  f"{latest.move_played} ({latest.thinking_tokens} tokens, "
                                  f"{latest.response_time_seconds:.1f}s)")
                        
                        # Show quick stats
                        if current_count % 10 == 0:
                            df = pd.DataFrame([vars(r) for r in session.results])
                            print(f"\n📊 Quick stats ({current_count} results):")
                            print(f"   Avg response time: {df['response_time_seconds'].mean():.2f}s")
                            print(f"   Avg thinking tokens: {df['thinking_tokens'].mean():.0f}")
                            print()
                    
                    if session.status == "completed":
                        print(f"\n✅ Session completed!")
                        break
                        
                except FileNotFoundError:
                    print("Waiting for session file...")
                except json.JSONDecodeError:
                    pass  # File being written, try again
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopped watching")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a single DataFrame."""
        all_results = []
        for session in self.sessions:
            for result in session.results:
                row = vars(result).copy()
                row['session_id'] = session.session_id
                row['dataset'] = session.dataset_name
                all_results.append(row)
        
        return pd.DataFrame(all_results)
    
    def get_time_pressure_curves(self) -> dict:
        """Get time pressure response curves for each model.
        
        Returns:
            Dict mapping model_id to DataFrame with time_level vs metrics
        """
        df = self.to_dataframe()
        curves = {}
        
        for model_id in df['model_id'].unique():
            model_df = df[df['model_id'] == model_id]
            curve = model_df.groupby('time_remaining').agg({
                'response_time_seconds': ['mean', 'std'],
                'thinking_tokens': ['mean', 'std'],
            }).reset_index()
            curve.columns = ['time_remaining', 'response_mean', 'response_std', 
                           'tokens_mean', 'tokens_std']
            curves[model_id] = curve
        
        return curves
    
    def get_variance_analysis(self) -> pd.DataFrame:
        """Analyze variance across samples for each condition.
        
        Returns:
            DataFrame with coefficient of variation for each condition
        """
        df = self.to_dataframe()
        
        variance_df = df.groupby(['model_id', 'position_id', 'time_remaining']).agg({
            'response_time_seconds': ['mean', 'std', 'count'],
            'thinking_tokens': ['mean', 'std'],
            'move_played': lambda x: len(set(x)),  # Number of unique moves
        }).reset_index()
        
        variance_df.columns = ['model_id', 'position_id', 'time_remaining',
                              'time_mean', 'time_std', 'n_samples',
                              'tokens_mean', 'tokens_std', 'unique_moves']
        
        # Coefficient of variation
        variance_df['time_cv'] = variance_df['time_std'] / variance_df['time_mean']
        variance_df['tokens_cv'] = variance_df['tokens_std'] / variance_df['tokens_mean']
        
        return variance_df
    
    def get_position_difficulty_analysis(self, dataset) -> pd.DataFrame:
        """Analyze how models perform on different position types.
        
        Args:
            dataset: PositionDataset with category information
            
        Returns:
            DataFrame with performance by position category
        """
        df = self.to_dataframe()
        
        # Add position metadata
        pos_meta = {p.position_id: {'category': p.category, 'difficulty': p.difficulty} 
                   for p in dataset.positions}
        
        df['category'] = df['position_id'].map(lambda x: pos_meta.get(x, {}).get('category', 'unknown'))
        df['difficulty'] = df['position_id'].map(lambda x: pos_meta.get(x, {}).get('difficulty', 'unknown'))
        
        # Aggregate by category and time
        category_analysis = df.groupby(['model_id', 'category', 'time_remaining']).agg({
            'response_time_seconds': 'mean',
            'thinking_tokens': 'mean',
            'is_best_move': 'mean' if 'is_best_move' in df.columns else 'count',
        }).reset_index()
        
        return category_analysis
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all models at each time level.
        
        Returns:
            DataFrame comparing models across metrics
        """
        df = self.to_dataframe()
        
        comparison = df.groupby(['model_id', 'time_remaining']).agg({
            'response_time_seconds': 'mean',
            'thinking_tokens': 'mean',
        }).reset_index()
        
        # Pivot for easier comparison
        time_pivot = comparison.pivot(index='time_remaining', 
                                      columns='model_id', 
                                      values='response_time_seconds')
        
        return time_pivot
    
    def generate_report(self, output_path: Path = None, include_partial_warning: bool = True) -> str:
        """Generate a markdown report of the analysis.
        
        Args:
            output_path: Optional path to save report
            include_partial_warning: Whether to warn about partial sessions
            
        Returns:
            Markdown report string
        """
        df = self.to_dataframe()
        
        report = ["# Offline Evaluation Analysis Report\n"]
        
        # Check for empty results
        if df.empty:
            report.append("> ⚠️ **No evaluation results found.** All sessions have 0 results.\n")
            report.append("## Summary\n")
            report.append(f"- **Sessions loaded:** {len(self.sessions)}")
            report.append(f"- **Total evaluations:** 0")
            report.append("")
            report.append("Please check that evaluations completed successfully.")
            
            report_str = "\n".join(report)
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(report_str)
            return report_str
        
        # Check for partial sessions
        partial_sessions = [s for s in self.sessions if s.status != "completed"]
        if partial_sessions and include_partial_warning:
            report.append("> ⚠️ **Warning:** This report includes partial/interrupted sessions. Results may be incomplete.\n")
        
        # Summary
        report.append("## Summary\n")
        report.append(f"- **Sessions analyzed:** {len(self.sessions)}")
        report.append(f"- **Models evaluated:** {df['model_id'].nunique()}")
        report.append(f"- **Positions evaluated:** {df['position_id'].nunique()}")
        report.append(f"- **Total evaluations:** {len(df)}")
        
        # Session status breakdown
        completed = sum(1 for s in self.sessions if s.status == "completed")
        interrupted = sum(1 for s in self.sessions if s.status == "interrupted")
        in_progress = sum(1 for s in self.sessions if s.status == "in_progress")
        if interrupted or in_progress:
            report.append(f"- **Session status:** {completed} completed, {interrupted} interrupted, {in_progress} in progress")
        report.append("")
        
        # Models
        report.append("### Models\n")
        for model in df['model_id'].unique():
            model_df = df[df['model_id'] == model]
            report.append(f"- **{model}**: {len(model_df)} evaluations")
        report.append("")
        
        # Time levels
        report.append("### Time Pressure Levels\n")
        for time_level in sorted(df['time_remaining'].unique(), reverse=True):
            report.append(f"- {time_level}s")
        report.append("")
        
        # Prompt styles (if tracked)
        if 'prompt_style' in df.columns and df['prompt_style'].notna().any():
            report.append("### Prompt Styles\n")
            for style in df['prompt_style'].dropna().unique():
                style_df = df[df['prompt_style'] == style]
                report.append(f"- **{style}**: {len(style_df)} evaluations")
            report.append("")
        
        # Timeout/forfeit analysis
        if 'would_timeout' in df.columns:
            report.append("## ⏰ Timeout Analysis\n")
            
            # Overall timeout rate
            timeout_rate = df['would_timeout'].mean() * 100
            total_timeouts = df['would_timeout'].sum()
            report.append(f"**Overall timeout rate:** {timeout_rate:.1f}% ({total_timeouts}/{len(df)} evaluations)\n")
            
            # Timeout rate by model and time level
            report.append("| Model | Time Level | Timeout Rate | Count |")
            report.append("|-------|------------|--------------|-------|")
            
            timeout_by_condition = df.groupby(['model_id', 'time_remaining']).agg({
                'would_timeout': ['sum', 'count', 'mean'],
            })
            timeout_by_condition.columns = ['timeouts', 'total', 'rate']
            timeout_by_condition = timeout_by_condition.reset_index()
            
            for _, row in timeout_by_condition.iterrows():
                rate_pct = row['rate'] * 100
                icon = "🔴" if rate_pct > 50 else "🟠" if rate_pct > 20 else "🟢"
                report.append(f"| {row['model_id']} | {row['time_remaining']:.0f}s | {icon} {rate_pct:.1f}% | {row['timeouts']:.0f}/{row['total']:.0f} |")
            report.append("")
        
        # Response time by model and time level
        report.append("## Response Time Analysis\n")
        report.append("| Model | Time Level | Avg Response (s) | Std | Avg Tokens |")
        report.append("|-------|------------|------------------|-----|------------|")
        
        summary = df.groupby(['model_id', 'time_remaining']).agg({
            'response_time_seconds': ['mean', 'std'],
            'thinking_tokens': 'mean',
        })
        summary.columns = ['resp_mean', 'resp_std', 'tokens_mean']
        summary = summary.reset_index()
        
        for _, row in summary.iterrows():
            model = row['model_id']
            time_level = row['time_remaining']
            resp_mean = row['resp_mean']
            resp_std = row['resp_std']
            tokens = row['tokens_mean']
            report.append(f"| {model} | {time_level:.0f}s | {resp_mean:.2f} | {resp_std:.2f} | {tokens:.0f} |")
        
        report.append("")
        
        # Variance analysis
        report.append("## Variance Analysis\n")
        variance = self.get_variance_analysis()
        avg_cv = variance.groupby('model_id')['time_cv'].mean()
        
        report.append("Average coefficient of variation for response time:\n")
        for model, cv in avg_cv.items():
            report.append(f"- **{model}**: {cv:.3f}")
        report.append("")
        
        # Unique moves
        report.append("### Move Consistency\n")
        move_consistency = variance.groupby('model_id')['unique_moves'].mean()
        report.append("Average unique moves across samples (lower = more consistent):\n")
        for model, unique in move_consistency.items():
            report.append(f"- **{model}**: {unique:.2f}")
        report.append("")
        
        # Move quality analysis (if centipawn data available)
        if 'centipawn_loss' in df.columns and df['centipawn_loss'].notna().any():
            report.append("## ♟️ Move Quality Analysis\n")
            
            # Overall stats
            quality_df = df[df['centipawn_loss'].notna()]
            report.append(f"**Moves analyzed:** {len(quality_df)}\n")
            
            # By model and time level
            report.append("| Model | Time Level | Avg CP Loss | Blunder Rate | Best Move % |")
            report.append("|-------|------------|-------------|--------------|-------------|")
            
            quality_summary = quality_df.groupby(['model_id', 'time_remaining']).agg({
                'centipawn_loss': 'mean',
                'is_blunder': 'mean',
                'is_best_move': 'mean',
            }).reset_index()
            
            for _, row in quality_summary.iterrows():
                model = row['model_id']
                time_level = row['time_remaining']
                cp_loss = row['centipawn_loss']
                blunder_rate = row['is_blunder'] * 100
                best_move_rate = row['is_best_move'] * 100
                
                # Color code quality
                cp_icon = "🟢" if cp_loss < 25 else "🟡" if cp_loss < 50 else "🟠" if cp_loss < 100 else "🔴"
                report.append(f"| {model} | {time_level:.0f}s | {cp_icon} {cp_loss:.1f} | {blunder_rate:.1f}% | {best_move_rate:.1f}% |")
            
            report.append("")
            
            # Quality by prompt style
            if 'prompt_style' in df.columns and df['prompt_style'].notna().any():
                report.append("### Quality by Prompt Style\n")
                style_quality = quality_df.groupby(['model_id', 'prompt_style']).agg({
                    'centipawn_loss': 'mean',
                    'is_blunder': 'mean',
                }).reset_index()
                
                report.append("| Model | Prompt Style | Avg CP Loss | Blunder Rate |")
                report.append("|-------|--------------|-------------|--------------|")
                
                for _, row in style_quality.iterrows():
                    report.append(f"| {row['model_id']} | {row['prompt_style']} | {row['centipawn_loss']:.1f} | {row['is_blunder']*100:.1f}% |")
                
                report.append("")
        
        report_str = "\n".join(report)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_str)
            print(f"Report saved: {output_path}")
        
        return report_str


def create_visualization_notebook(session_dir: Path, output_path: Path) -> None:
    """Create a Jupyter notebook for visualizing offline eval results.
    
    Args:
        session_dir: Directory containing session JSON files
        output_path: Path for the output notebook
    """
    cells = [
        {
            "cell_type": "markdown",
            "source": "# Offline Evaluation Analysis\n\nThis notebook visualizes the results of controlled time pressure experiments."
        },
        {
            "cell_type": "code",
            "source": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# Load sessions
from game_arena.blitz.offline_eval import OfflineAnalyzer

analyzer = OfflineAnalyzer(session_dir=Path("{session_dir}"))
df = analyzer.to_dataframe()

print(f"Loaded {len(df)} evaluations")
print(f"Models: {df['model_id'].unique()}")
print(f"Time levels: {sorted(df['time_remaining'].unique())}")""".format(session_dir=session_dir)
        },
        {
            "cell_type": "markdown",
            "source": "## Response Time vs Time Remaining"
        },
        {
            "cell_type": "code",
            "source": """fig, ax = plt.subplots(figsize=(12, 6))

for model in df['model_id'].unique():
    model_df = df[df['model_id'] == model]
    grouped = model_df.groupby('time_remaining')['response_time_seconds'].agg(['mean', 'std'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                label=model, marker='o', capsize=5)

ax.set_xlabel('Time Remaining (seconds)')
ax.set_ylabel('Response Time (seconds)')
ax.set_title('Response Time vs Time Pressure')
ax.legend()
ax.invert_xaxis()  # Lower time on right
plt.tight_layout()
plt.show()"""
        },
        {
            "cell_type": "markdown",
            "source": "## Thinking Tokens vs Time Remaining"
        },
        {
            "cell_type": "code",
            "source": """fig, ax = plt.subplots(figsize=(12, 6))

for model in df['model_id'].unique():
    model_df = df[df['model_id'] == model]
    grouped = model_df.groupby('time_remaining')['thinking_tokens'].agg(['mean', 'std'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                label=model, marker='s', capsize=5)

ax.set_xlabel('Time Remaining (seconds)')
ax.set_ylabel('Thinking Tokens')
ax.set_title('Thinking Depth vs Time Pressure')
ax.legend()
ax.invert_xaxis()
plt.tight_layout()
plt.show()"""
        },
        {
            "cell_type": "markdown",
            "source": "## Variance Across Samples"
        },
        {
            "cell_type": "code",
            "source": """variance_df = analyzer.get_variance_analysis()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time CV
sns.boxplot(data=variance_df, x='time_remaining', y='time_cv', 
            hue='model_id', ax=axes[0])
axes[0].set_title('Response Time Coefficient of Variation')
axes[0].set_xlabel('Time Remaining (s)')
axes[0].set_ylabel('CV (std/mean)')

# Unique moves
sns.boxplot(data=variance_df, x='time_remaining', y='unique_moves', 
            hue='model_id', ax=axes[1])
axes[1].set_title('Move Consistency (unique moves across samples)')
axes[1].set_xlabel('Time Remaining (s)')
axes[1].set_ylabel('Unique Moves')

plt.tight_layout()
plt.show()"""
        },
        {
            "cell_type": "markdown",
            "source": "## Heatmap: Model x Time Level"
        },
        {
            "cell_type": "code",
            "source": """pivot = df.pivot_table(
    values='response_time_seconds', 
    index='model_id', 
    columns='time_remaining',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
ax.set_title('Average Response Time (seconds)')
ax.set_xlabel('Time Remaining (seconds)')
plt.tight_layout()
plt.show()"""
        },
    ]
    
    # Convert to notebook format
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "cells": [
            {
                "cell_type": cell.get("cell_type", "code"),
                "metadata": {},
                "source": cell["source"].split("\n"),
                "outputs": [],
                **({"execution_count": None} if cell.get("cell_type") != "markdown" else {}),
            }
            for cell in cells
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Notebook created: {output_path}")

