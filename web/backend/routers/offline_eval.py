#!/usr/bin/env python3
"""API routes for offline evaluation results."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from web.backend.services.offline_eval_service import OfflineEvalService

router = APIRouter(prefix="/api/offline-eval", tags=["offline-eval"])

# Initialize service with default path
_service: Optional[OfflineEvalService] = None


def get_service() -> OfflineEvalService:
    global _service
    if _service is None:
        # Path relative to project root (web/backend/../../_results)
        project_root = Path(__file__).parent.parent.parent.parent
        _service = OfflineEvalService(project_root / "_results" / "offline_eval")
    return _service


@router.get("/sessions")
async def get_sessions():
    """Get list of all offline evaluation sessions."""
    service = get_service()
    return {"sessions": service.get_sessions()}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific evaluation session."""
    service = get_service()
    session = service.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.get("/summary")
async def get_summary():
    """Get summary statistics across all evaluations."""
    service = get_service()
    return service.get_summary()


@router.get("/analysis/timeouts")
async def get_timeout_analysis():
    """Get timeout analysis by model and time level."""
    service = get_service()
    return service.get_timeout_analysis()


@router.get("/analysis/response-times")
async def get_response_time_analysis():
    """Get response time analysis."""
    service = get_service()
    return {"data": service.get_response_time_analysis()}


@router.get("/analysis/move-quality")
async def get_move_quality_analysis():
    """Get move quality analysis (centipawn loss, blunders)."""
    service = get_service()
    return service.get_move_quality_analysis()


@router.get("/analysis/ablation")
async def get_ablation_comparison():
    """Get prompt style ablation comparison."""
    service = get_service()
    return service.get_ablation_comparison()

