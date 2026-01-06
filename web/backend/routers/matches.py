#!/usr/bin/env python3
"""Match-related API endpoints."""

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from models.schemas import MatchSummary, MatchDetail, GameDetail
from services.match_runner import MatchConfig, get_match_runner

router = APIRouter()


@router.get("", response_model=list[MatchSummary])
async def list_matches(
    request: Request,
    limit: int = 50,
    offset: int = 0,
):
    """Get all matches, sorted by most recent first."""
    match_service = request.app.state.match_service
    return match_service.get_all_matches(limit=limit, offset=offset)


@router.post("/refresh")
async def refresh_matches(request: Request):
    """Refresh the match cache by rescanning the results directory."""
    match_service = request.app.state.match_service
    match_service.scan_results()
    
    return {
        "status": "ok",
        "matches_loaded": len(match_service.matches),
    }


@router.post("/start")
async def start_match(config: MatchConfig):
    """Start a new match with the given configuration.
    
    This spawns a background process to run the match.
    The match will appear in the matches list once it starts recording data.
    """
    runner = get_match_runner()
    result = runner.start_match(config)
    return result


@router.get("/processes")
async def get_all_processes():
    """Get list of all tracked match processes (running and recent)."""
    runner = get_match_runner()
    return runner.get_all_processes()


@router.get("/processes/{pid}")
async def get_process_status(pid: int):
    """Get detailed status of a specific process including logs."""
    runner = get_match_runner()
    status = runner.get_process_status(pid)
    if not status:
        raise HTTPException(status_code=404, detail=f"Process {pid} not found")
    return status


@router.post("/processes/{pid}/stop")
async def stop_match(pid: int):
    """Stop a running match by process ID."""
    runner = get_match_runner()
    if runner.stop_match(pid):
        return {"status": "stopped", "pid": pid}
    raise HTTPException(status_code=404, detail=f"Process {pid} not found")


@router.post("/processes/clear")
async def clear_finished_processes():
    """Remove finished processes from tracking."""
    runner = get_match_runner()
    removed = runner.clear_finished()
    return {"status": "ok", "removed": removed}


@router.get("/processes/test")
async def test_subprocess():
    """Test subprocess output capture with a simple command."""
    import subprocess
    import sys
    
    # Run a simple Python command that prints to stdout
    result = subprocess.run(
        [sys.executable, "-c", "print('Hello from subprocess!'); import sys; print('stderr test', file=sys.stderr)"],
        capture_output=True,
        text=True,
    )
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@router.delete("/{match_id}")
async def delete_match(request: Request, match_id: str):
    """Delete a match and all its data.
    
    This permanently removes the match directory from the filesystem.
    """
    match_service = request.app.state.match_service
    
    if match_id not in match_service.matches:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    
    metadata = match_service.matches[match_id]
    match_dir = Path(metadata.get("_dir", ""))
    
    if not match_dir.exists():
        raise HTTPException(status_code=404, detail=f"Match directory not found: {match_id}")
    
    try:
        # Remove the entire match directory
        shutil.rmtree(match_dir)
        
        # Remove from cache
        del match_service.matches[match_id]
        
        return {"status": "deleted", "match_id": match_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete match: {e}")


@router.get("/{match_id}", response_model=MatchDetail)
async def get_match(request: Request, match_id: str):
    """Get full details for a specific match."""
    match_service = request.app.state.match_service
    match = match_service.get_match(match_id)
    
    if not match:
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    
    return match


@router.get("/{match_id}/games/{game_number}", response_model=GameDetail)
async def get_game(request: Request, match_id: str, game_number: int):
    """Get full details for a specific game including all moves."""
    match_service = request.app.state.match_service
    game = match_service.get_game(match_id, game_number)
    
    if not game:
        raise HTTPException(
            status_code=404,
            detail=f"Game {game_number} not found in match {match_id}"
        )
    
    return game


@router.get("/{match_id}/live/{game_number}", response_model=GameDetail)
async def get_live_game(request: Request, match_id: str, game_number: int):
    """Get live game data including moves from the current in-progress game.
    
    This endpoint works for games that are still in progress and haven't
    been recorded in games_summary.csv yet.
    """
    match_service = request.app.state.match_service
    
    # Refresh to get latest data
    match_service.scan_results()
    
    game = match_service.get_live_game(match_id, game_number)
    
    if not game:
        raise HTTPException(
            status_code=404,
            detail=f"Game {game_number} not found in match {match_id}"
        )
    
    return game

