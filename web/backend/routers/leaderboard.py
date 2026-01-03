#!/usr/bin/env python3
"""Leaderboard API endpoints."""

from fastapi import APIRouter, Request

from models.schemas import LeaderboardResponse
from services.elo_service import EloService

router = APIRouter()

# Global ELO service instance
_elo_service: EloService | None = None


def get_elo_service(request: Request) -> EloService:
    """Get or create the ELO service, rebuilding from matches if needed."""
    global _elo_service
    
    if _elo_service is None:
        _elo_service = EloService()
        match_service = request.app.state.match_service
        _elo_service.rebuild_from_matches(list(match_service.matches.values()))
    
    return _elo_service


@router.get("", response_model=LeaderboardResponse)
async def get_leaderboard(request: Request):
    """Get the current model leaderboard sorted by ELO."""
    elo_service = get_elo_service(request)
    return elo_service.get_leaderboard()


@router.post("/rebuild")
async def rebuild_leaderboard(request: Request):
    """Rebuild the leaderboard from all match data."""
    global _elo_service
    
    _elo_service = EloService()
    match_service = request.app.state.match_service
    _elo_service.rebuild_from_matches(list(match_service.matches.values()))
    
    leaderboard = _elo_service.get_leaderboard()
    
    return {
        "status": "ok",
        "models_ranked": len(leaderboard.models),
    }

