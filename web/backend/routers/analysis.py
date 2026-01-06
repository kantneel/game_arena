#!/usr/bin/env python3
"""Analysis API endpoints."""

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from services.analysis_service import AnalysisService

router = APIRouter()


def sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types and handle NaN/Inf for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float) or (hasattr(obj, 'item') and callable(obj.item)):
        # Handle numpy floats and regular floats
        val = float(obj) if hasattr(obj, 'item') else obj
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif hasattr(obj, 'item'):
        # Other numpy scalars
        return obj.item()
    return obj


def get_analysis_service(request: Request) -> AnalysisService:
    """Get or create analysis service."""
    if not hasattr(request.app.state, "analysis_service"):
        results_dir = Path(__file__).parent.parent.parent.parent / "_results"
        request.app.state.analysis_service = AnalysisService(results_dir)
    return request.app.state.analysis_service


@router.get("/matches/{match_id}")
async def get_match_analysis(request: Request, match_id: str):
    """Get complete time pressure analysis for a match."""
    service = get_analysis_service(request)
    analysis = service.analyze_match(match_id)
    
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Match not found or no move data: {match_id}")
    
    # Convert to dict for JSON response, sanitizing numpy types and NaN values
    result = {
        "match_id": analysis.match_id,
        "model_a": {
            "name": analysis.model_a_profile.model_name,
            "total_moves": analysis.model_a_profile.total_moves,
            "speed_adaptation_ratio": analysis.model_a_profile.speed_adaptation_ratio,
            "quality_degradation_ratio": analysis.model_a_profile.quality_degradation_ratio,
            "thinking_reduction_ratio": analysis.model_a_profile.thinking_reduction_ratio,
            "pressure_stats": [asdict(ps) for ps in analysis.model_a_profile.pressure_stats],
        },
        "model_b": {
            "name": analysis.model_b_profile.model_name,
            "total_moves": analysis.model_b_profile.total_moves,
            "speed_adaptation_ratio": analysis.model_b_profile.speed_adaptation_ratio,
            "quality_degradation_ratio": analysis.model_b_profile.quality_degradation_ratio,
            "thinking_reduction_ratio": analysis.model_b_profile.thinking_reduction_ratio,
            "pressure_stats": [asdict(ps) for ps in analysis.model_b_profile.pressure_stats],
        },
        "insights": analysis.insights,
    }
    return sanitize_for_json(result)


@router.get("/matches/{match_id}/scatter")
async def get_pressure_scatter(request: Request, match_id: str):
    """Get scatter plot data: time remaining vs move time."""
    service = get_analysis_service(request)
    data = service.get_pressure_scatter_data(match_id)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    
    return sanitize_for_json(data)


@router.get("/matches/{match_id}/thinking")
async def get_thinking_by_pressure(request: Request, match_id: str):
    """Get thinking tokens grouped by pressure level."""
    service = get_analysis_service(request)
    data = service.get_thinking_by_pressure(match_id)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    
    return sanitize_for_json(data)

