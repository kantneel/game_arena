#!/usr/bin/env python3
"""Live match WebSocket endpoints."""

import asyncio
import json
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from models.schemas import LiveMatchState

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for live updates."""
    
    def __init__(self):
        self.active_connections: dict[str, Set[WebSocket]] = {}  # match_id -> connections
        self.global_connections: Set[WebSocket] = set()  # Connections watching all matches
    
    async def connect(self, websocket: WebSocket, match_id: str | None = None):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        
        if match_id:
            if match_id not in self.active_connections:
                self.active_connections[match_id] = set()
            self.active_connections[match_id].add(websocket)
        else:
            self.global_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket, match_id: str | None = None):
        """Remove a WebSocket connection."""
        if match_id and match_id in self.active_connections:
            self.active_connections[match_id].discard(websocket)
        self.global_connections.discard(websocket)
    
    async def broadcast_to_match(self, match_id: str, message: dict):
        """Send a message to all connections watching a specific match."""
        connections = self.active_connections.get(match_id, set())
        dead_connections = set()
        
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)
        
        # Clean up dead connections
        for conn in dead_connections:
            self.active_connections[match_id].discard(conn)
    
    async def broadcast_global(self, message: dict):
        """Send a message to all global connections."""
        dead_connections = set()
        
        for connection in self.global_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)
        
        # Clean up dead connections
        for conn in dead_connections:
            self.global_connections.discard(conn)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_global(websocket: WebSocket):
    """WebSocket endpoint for watching all live matches."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive, wait for client messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/{match_id}")
async def websocket_match(websocket: WebSocket, match_id: str):
    """WebSocket endpoint for watching a specific live match."""
    await manager.connect(websocket, match_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, match_id)


@router.get("/matches")
async def get_live_matches(request: Request):
    """Get list of currently live matches."""
    from datetime import datetime, timedelta
    
    match_service = request.app.state.match_service
    
    # Rescan results directory to pick up new/updated matches
    match_service.scan_results()
    
    # Filter for matches without end_time (still in progress)
    # Also check last_updated to filter out stale/abandoned matches
    stale_threshold = timedelta(minutes=5)
    now = datetime.now()
    
    live_matches = []
    for m in match_service.matches.values():
        if m.get("end_time"):
            continue  # Match completed
        
        # Check if match is stale (no heartbeat in 5 minutes)
        last_updated_str = m.get("last_updated")
        if last_updated_str:
            try:
                last_updated = datetime.fromisoformat(last_updated_str)
                if now - last_updated > stale_threshold:
                    continue  # Stale match, skip
            except (ValueError, TypeError):
                pass
        
        live_matches.append(match_service._to_match_summary(m))
    
    return live_matches


@router.get("/stale")
async def get_stale_matches(request: Request):
    """Get list of stale/abandoned matches (no end_time but no recent heartbeat)."""
    from datetime import datetime, timedelta
    
    match_service = request.app.state.match_service
    match_service.scan_results()
    
    stale_threshold = timedelta(minutes=5)
    now = datetime.now()
    
    stale_matches = []
    for m in match_service.matches.values():
        if m.get("end_time"):
            continue  # Match completed, not stale
        
        # Check if match is stale
        last_updated_str = m.get("last_updated")
        if last_updated_str:
            try:
                last_updated = datetime.fromisoformat(last_updated_str)
                if now - last_updated > stale_threshold:
                    summary = match_service._to_match_summary(m)
                    summary.status = "stale"
                    stale_matches.append(summary)
            except (ValueError, TypeError):
                pass
        else:
            # No heartbeat at all - could be old match format or truly abandoned
            start_time_str = m.get("start_time")
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str)
                    if now - start_time > stale_threshold:
                        summary = match_service._to_match_summary(m)
                        summary.status = "stale"
                        stale_matches.append(summary)
                except (ValueError, TypeError):
                    pass
    
    return stale_matches


@router.post("/abandon/{match_id}")
async def abandon_match(request: Request, match_id: str):
    """Mark a stale match as abandoned by setting its end_time."""
    from datetime import datetime
    import json
    from pathlib import Path
    
    match_service = request.app.state.match_service
    
    if match_id not in match_service.matches:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Match not found: {match_id}")
    
    match_data = match_service.matches[match_id]
    match_dir = Path(match_data.get("_dir", ""))
    metadata_file = match_dir / "metadata.json"
    
    if not metadata_file.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Metadata file not found for match: {match_id}")
    
    # Update the metadata file
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    metadata["end_time"] = datetime.now().isoformat()
    metadata["final_winner"] = "abandoned"
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Refresh the match service cache
    match_service.scan_results()
    
    return {"status": "ok", "match_id": match_id, "message": "Match marked as abandoned"}


# Function to be called by the game engine to broadcast updates
async def broadcast_move(
    match_id: str,
    game_number: int,
    player: str,
    move: str,
    fen: str,
    model_a_time: float,
    model_b_time: float,
    thinking_preview: str = "",
):
    """Broadcast a move to all connected clients.
    
    This should be called by the game engine when a move is made.
    """
    message = {
        "type": "move",
        "match_id": match_id,
        "game_number": game_number,
        "player": player,
        "move": move,
        "fen": fen,
        "model_a_time": model_a_time,
        "model_b_time": model_b_time,
        "thinking_preview": thinking_preview[:200],  # Limit preview size
    }
    
    await manager.broadcast_to_match(match_id, message)
    await manager.broadcast_global(message)


async def broadcast_game_end(
    match_id: str,
    game_number: int,
    winner: str,
    result: str,
    termination: str,
):
    """Broadcast game end to all connected clients."""
    message = {
        "type": "game_end",
        "match_id": match_id,
        "game_number": game_number,
        "winner": winner,
        "result": result,
        "termination": termination,
    }
    
    await manager.broadcast_to_match(match_id, message)
    await manager.broadcast_global(message)


async def broadcast_match_end(
    match_id: str,
    winner: str,
    model_a_score: int,
    model_b_score: int,
):
    """Broadcast match end to all connected clients."""
    message = {
        "type": "match_end",
        "match_id": match_id,
        "winner": winner,
        "model_a_score": model_a_score,
        "model_b_score": model_b_score,
    }
    
    await manager.broadcast_to_match(match_id, message)
    await manager.broadcast_global(message)

