#!/usr/bin/env python3
"""Game Arena Web Dashboard - FastAPI Backend."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import matches, leaderboard, live, config
from services.match_service import MatchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    # Initialize match service with results directory
    results_dir = Path(__file__).parent.parent.parent / "_results"
    app.state.match_service = MatchService(results_dir)
    app.state.match_service.scan_results()
    print(f"📊 Loaded {len(app.state.match_service.matches)} matches from {results_dir}")
    yield
    # Cleanup on shutdown
    print("👋 Shutting down Game Arena backend")


app = FastAPI(
    title="Game Arena",
    description="LLM Chess Battle Arena - Live matches, results, and leaderboards",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(matches.router, prefix="/api/matches", tags=["matches"])
app.include_router(leaderboard.router, prefix="/api/leaderboard", tags=["leaderboard"])
app.include_router(live.router, prefix="/api/live", tags=["live"])
app.include_router(config.router, prefix="/api/config", tags=["config"])


@app.get("/")
async def root():
    return {
        "name": "Game Arena API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}


def run_server():
    """Entry point for running the web server."""
    import uvicorn
    uvicorn.run(
        "web.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["web/backend"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

