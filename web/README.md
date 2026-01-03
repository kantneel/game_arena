# Game Arena Web Dashboard

A real-time web interface for viewing LLM chess battles.

## Quick Start

### Option 1: Single Command (Recommended)

From the `game_arena` root directory:

```bash
# Install with web dependencies
uv pip install -e ".[web]"

# Run both backend and frontend
./scripts/run_web.sh
```

This starts:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Option 2: Run Separately

#### Backend (FastAPI)

```bash
cd web/backend

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend (Next.js)

```bash
cd web/frontend

# Install dependencies (first time only)
bun install

# Run development server
bun dev
```

## Features

### Pages

- **Home** (`/`) - Live matches, recent results, top rankings
- **Live** (`/live`) - Currently running matches with real-time updates
- **Matches** (`/matches`) - All completed matches
- **Match Detail** (`/matches/[id]`) - Individual match with all games
- **Game Replay** (`/matches/[id]/games/[num]`) - Interactive game replay with move-by-move playback
- **Leaderboard** (`/leaderboard`) - ELO rankings for all models

### Real-time Updates

The dashboard connects via WebSocket to receive live updates during matches:

- Move updates with board position
- Time remaining for each player
- Game and match completion events
- Thinking preview (first 200 chars of model reasoning)

## API Endpoints

### Matches

- `GET /api/matches` - List all matches
- `GET /api/matches/{match_id}` - Get match details
- `GET /api/matches/{match_id}/games/{game_number}` - Get game with moves
- `POST /api/matches/refresh` - Refresh match cache

### Leaderboard

- `GET /api/leaderboard` - Get current rankings
- `POST /api/leaderboard/rebuild` - Rebuild ELO from all matches

### Live

- `GET /api/live/matches` - Get currently live matches
- `WS /api/live/ws` - Global WebSocket for all matches
- `WS /api/live/ws/{match_id}` - WebSocket for specific match

## Configuration

### Environment Variables

Frontend (`.env.local`):
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
NEXT_PUBLIC_WS_URL=ws://localhost:8000/api/live
```

## Integration with Game Engine

To broadcast live updates from the game engine, import and call the broadcast functions:

```python
from web.backend.routers.live import broadcast_move, broadcast_game_end, broadcast_match_end

# After each move
await broadcast_move(
    match_id="...",
    game_number=1,
    player="claude-sonnet-4.5",
    move="e2e4",
    fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    model_a_time=295.5,
    model_b_time=300.0,
    thinking_preview="I'll open with e4 to control the center...",
)

# After game ends
await broadcast_game_end(
    match_id="...",
    game_number=1,
    winner="model_a",
    result="1-0",
    termination="checkmate",
)

# After match ends
await broadcast_match_end(
    match_id="...",
    winner="model_a",
    model_a_score=3,
    model_b_score=2,
)
```

## Tech Stack

- **Backend**: FastAPI, Pydantic, WebSockets
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Chess**: react-chessboard, chess.js
- **Charts**: Recharts

