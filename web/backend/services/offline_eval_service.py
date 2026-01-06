#!/usr/bin/env python3
"""Service for loading and analyzing offline evaluation results."""

import json
import math
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


@dataclass
class OfflineEvalSummary:
    """Summary of offline evaluation results."""
    total_sessions: int
    total_evaluations: int
    models: list[str]
    prompt_styles: list[str]
    time_levels: list[float]
    overall_timeout_rate: float
    has_move_quality: bool


class OfflineEvalService:
    """Service for accessing offline evaluation data."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path("_results/offline_eval")
    
    def get_sessions(self) -> list[dict]:
        """Get list of all evaluation sessions."""
        sessions = []
        
        if not self.results_dir.exists():
            return sessions
        
        for session_file in self.results_dir.glob("*.json"):
            if "partial" in session_file.name:
                continue
            
            try:
                with open(session_file) as f:
                    data = json.load(f)
                
                sessions.append({
                    "session_id": data.get("session_id", session_file.stem),
                    "model_id": data.get("model_id", "unknown"),
                    "dataset_name": data.get("dataset_name", "unknown"),
                    "status": data.get("status", "unknown"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "result_count": len(data.get("results", [])),
                    "prompt_style": data.get("config", {}).get("prompt_style", "standard"),
                })
            except Exception:
                continue
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda s: s.get("start_time", ""), reverse=True)
        return sessions
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a specific session by ID."""
        session_file = self.results_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file) as f:
            return json.load(f)
    
    def get_summary(self) -> dict:
        """Get summary statistics across all sessions."""
        df = self.to_dataframe()
        
        if df.empty:
            return {
                "total_sessions": 0,
                "total_evaluations": 0,
                "models": [],
                "prompt_styles": [],
                "time_levels": [],
                "overall_timeout_rate": 0.0,
                "has_move_quality": False,
            }
        
        has_move_quality = 'centipawn_loss' in df.columns and bool(df['centipawn_loss'].notna().any())
        overall_timeout_rate = float(df['would_timeout'].mean()) if 'would_timeout' in df.columns else 0.0
        
        return sanitize_for_json({
            "total_sessions": len(self.get_sessions()),
            "total_evaluations": len(df),
            "models": sorted(df['model_id'].unique().tolist()),
            "prompt_styles": sorted(df['prompt_style'].dropna().unique().tolist()) if 'prompt_style' in df.columns else [],
            "time_levels": sorted([float(t) for t in df['time_remaining'].unique().tolist()], reverse=True),
            "overall_timeout_rate": overall_timeout_rate,
            "has_move_quality": has_move_quality,
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a DataFrame."""
        all_results = []
        
        for session_file in self.results_dir.glob("*.json"):
            if "partial" in session_file.name:
                continue
            
            try:
                with open(session_file) as f:
                    data = json.load(f)
                
                session_id = data.get("session_id", session_file.stem)
                config = data.get("config", {})
                
                for result in data.get("results", []):
                    result["session_id"] = session_id
                    result["prompt_style"] = config.get("prompt_style", "standard")
                    all_results.append(result)
            except Exception:
                continue
        
        return pd.DataFrame(all_results)
    
    def get_timeout_analysis(self) -> dict:
        """Get timeout analysis by model and time level."""
        df = self.to_dataframe()
        
        if df.empty or 'would_timeout' not in df.columns:
            return {"by_model_time": [], "by_prompt_style": []}
        
        # By model and time level
        by_model_time = df.groupby(['model_id', 'time_remaining']).agg({
            'would_timeout': ['sum', 'count', 'mean'],
        }).reset_index()
        by_model_time.columns = ['model_id', 'time_remaining', 'timeouts', 'total', 'rate']
        
        model_time_data = []
        for _, row in by_model_time.iterrows():
            model_time_data.append(sanitize_for_json({
                "model_id": row['model_id'],
                "time_remaining": row['time_remaining'],
                "timeouts": row['timeouts'],
                "total": row['total'],
                "rate": row['rate'],
            }))
        
        # By prompt style
        style_data = []
        if 'prompt_style' in df.columns:
            by_style = df.groupby(['model_id', 'prompt_style']).agg({
                'would_timeout': ['sum', 'count', 'mean'],
            }).reset_index()
            by_style.columns = ['model_id', 'prompt_style', 'timeouts', 'total', 'rate']
            
            for _, row in by_style.iterrows():
                style_data.append(sanitize_for_json({
                    "model_id": row['model_id'],
                    "prompt_style": row['prompt_style'],
                    "timeouts": row['timeouts'],
                    "total": row['total'],
                    "rate": row['rate'],
                }))
        
        return {
            "by_model_time": model_time_data,
            "by_prompt_style": style_data,
        }
    
    def get_response_time_analysis(self) -> list[dict]:
        """Get response time analysis by model and time level."""
        df = self.to_dataframe()
        
        if df.empty:
            return []
        
        summary = df.groupby(['model_id', 'time_remaining']).agg({
            'response_time_seconds': ['mean', 'std'],
            'thinking_tokens': 'mean',
        }).reset_index()
        summary.columns = ['model_id', 'time_remaining', 'resp_mean', 'resp_std', 'tokens_mean']
        
        result = []
        for _, row in summary.iterrows():
            result.append(sanitize_for_json({
                "model_id": row['model_id'],
                "time_remaining": row['time_remaining'],
                "avg_response_time": row['resp_mean'],
                "std_response_time": row['resp_std'],
                "avg_thinking_tokens": row['tokens_mean'],
            }))
        
        return result
    
    def get_move_quality_analysis(self) -> dict:
        """Get move quality analysis (centipawn loss, blunders, etc.)."""
        df = self.to_dataframe()
        
        if df.empty or 'centipawn_loss' not in df.columns:
            return {"available": False, "by_model_time": [], "by_prompt_style": []}
        
        quality_df = df[df['centipawn_loss'].notna()].copy()
        
        if quality_df.empty:
            return {"available": False, "by_model_time": [], "by_prompt_style": []}
        
        # Filter out extreme outliers (likely mate scores or parsing errors)
        # Normal centipawn loss should be between -500 and 1000
        quality_df = quality_df[
            (quality_df['centipawn_loss'] >= -500) & 
            (quality_df['centipawn_loss'] <= 1000)
        ]
        
        if quality_df.empty:
            return {"available": False, "by_model_time": [], "by_prompt_style": []}
        
        # By model and time level
        by_model_time = quality_df.groupby(['model_id', 'time_remaining']).agg({
            'centipawn_loss': 'mean',
            'is_blunder': 'mean',
            'is_best_move': 'mean',
        }).reset_index()
        
        model_time_data = []
        for _, row in by_model_time.iterrows():
            model_time_data.append(sanitize_for_json({
                "model_id": row['model_id'],
                "time_remaining": row['time_remaining'],
                "avg_centipawn_loss": row['centipawn_loss'],
                "blunder_rate": row['is_blunder'],
                "best_move_rate": row['is_best_move'],
            }))
        
        # By prompt style
        style_data = []
        if 'prompt_style' in df.columns:
            by_style = quality_df.groupby(['model_id', 'prompt_style']).agg({
                'centipawn_loss': 'mean',
                'is_blunder': 'mean',
            }).reset_index()
            
            for _, row in by_style.iterrows():
                style_data.append(sanitize_for_json({
                    "model_id": row['model_id'],
                    "prompt_style": row['prompt_style'],
                    "avg_centipawn_loss": row['centipawn_loss'],
                    "blunder_rate": row['is_blunder'],
                }))
        
        return {
            "available": True,
            "total_analyzed": len(quality_df),
            "by_model_time": model_time_data,
            "by_prompt_style": style_data,
        }
    
    def get_ablation_comparison(self) -> dict:
        """Get side-by-side comparison for prompt style ablation."""
        df = self.to_dataframe()
        
        if df.empty or 'prompt_style' not in df.columns:
            return {"available": False}
        
        styles = df['prompt_style'].dropna().unique().tolist()
        if len(styles) < 2:
            return {"available": False}
        
        comparison = {}
        for model_id in df['model_id'].unique():
            model_df = df[df['model_id'] == model_id]
            
            model_comparison = {}
            for style in styles:
                style_df = model_df[model_df['prompt_style'] == style]
                if style_df.empty:
                    continue
                
                style_data = {
                    "evaluations": int(len(style_df)),
                    "timeout_rate": style_df['would_timeout'].mean() if 'would_timeout' in style_df.columns else None,
                    "avg_response_time": style_df['response_time_seconds'].mean(),
                    "avg_thinking_tokens": style_df['thinking_tokens'].mean(),
                }
                
                if 'centipawn_loss' in style_df.columns and style_df['centipawn_loss'].notna().any():
                    quality_df = style_df[style_df['centipawn_loss'].notna()]
                    style_data["avg_centipawn_loss"] = quality_df['centipawn_loss'].mean()
                    style_data["blunder_rate"] = quality_df['is_blunder'].mean()
                
                model_comparison[style] = sanitize_for_json(style_data)
            
            if model_comparison:
                comparison[model_id] = model_comparison
        
        return {
            "available": True,
            "styles": styles,
            "models": comparison,
        }

