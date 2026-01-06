#!/usr/bin/env python3
"""Model profile API endpoints."""

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from services.model_stats_service import ModelStatsService

router = APIRouter()


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types and handle NaN/Inf for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float) or (hasattr(obj, 'item') and callable(obj.item)):
        val = float(obj) if hasattr(obj, 'item') else obj
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


def get_model_stats_service(request: Request) -> ModelStatsService:
    """Get or create model stats service."""
    if not hasattr(request.app.state, "model_stats_service"):
        results_dir = Path(__file__).parent.parent.parent.parent / "_results"
        request.app.state.model_stats_service = ModelStatsService(results_dir)
    return request.app.state.model_stats_service


@router.get("")
async def list_models(request: Request):
    """Get list of all models with basic stats."""
    service = get_model_stats_service(request)
    return sanitize_for_json(service.get_all_models())


@router.get("/{model_id}")
async def get_model_profile(request: Request, model_id: str):
    """Get complete profile for a specific model."""
    service = get_model_stats_service(request)
    profile = service.get_model_profile(model_id)
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    # Convert to dict, handling nested dataclasses
    result = {
        "model_id": profile.model_id,
        "display_name": profile.display_name,
        "total_matches": profile.total_matches,
        "total_games": profile.total_games,
        "total_moves": profile.total_moves,
        "wins": profile.wins,
        "losses": profile.losses,
        "draws": profile.draws,
        "elo": profile.elo,
        "win_rate": profile.win_rate,
        "avg_move_time": profile.avg_move_time,
        "avg_thinking_tokens": profile.avg_thinking_tokens,
        "speed_adaptation_ratio": profile.speed_adaptation_ratio,
        "quality_degradation_ratio": profile.quality_degradation_ratio,
        "thinking_reduction_ratio": profile.thinking_reduction_ratio,
        "pressure_stats": [asdict(ps) for ps in profile.pressure_stats],
        "recent_matches": [
            {
                "match_id": m.match_id,
                "opponent": m.opponent,
                "wins": m.wins,
                "losses": m.losses,
                "draws": m.draws,
                "result": m.result,
                "date": m.date.isoformat(),
            }
            for m in profile.recent_matches
        ],
    }
    return sanitize_for_json(result)


@router.get("/compare")
async def compare_models(request: Request, models: str):
    """Compare multiple models' pressure behavior.
    
    Args:
        models: Comma-separated list of model IDs
    """
    service = get_model_stats_service(request)
    model_ids = [m.strip() for m in models.split(",")]
    
    if len(model_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compare")
    
    return sanitize_for_json(service.get_model_comparison(model_ids))

