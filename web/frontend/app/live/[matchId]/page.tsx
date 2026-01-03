"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ChessBoardComponent } from "@/components/chess/ChessBoard";
import { api, MatchDetail, GameDetail } from "@/lib/api";

interface LiveState {
  fen: string;
  lastMove: string | null;
  modelATime: number;
  modelBTime: number;
  toMove: "model_a" | "model_b";
  gameNumber: number;
  moveCount: number;
  modelAScore: number;
  modelBScore: number;
  thinkingPreview: string;
}

export default function LiveMatchPage() {
  const params = useParams();
  const matchId = params.matchId as string;

  const [match, setMatch] = useState<MatchDetail | null>(null);
  const [currentGame, setCurrentGame] = useState<GameDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [state, setState] = useState<LiveState>({
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    lastMove: null,
    modelATime: 300,
    modelBTime: 300,
    toMove: "model_a",
    gameNumber: 1,
    moveCount: 0,
    modelAScore: 0,
    modelBScore: 0,
    thinkingPreview: "",
  });

  const [connected, setConnected] = useState(true);
  const modelA = match?.model_a || "Model A";
  const modelB = match?.model_b || "Model B";

  // Fetch match details and poll for updates
  useEffect(() => {
    let isMounted = true;

    const fetchMatchData = async () => {
      try {
        // Refresh the match cache first
        await api.refreshMatches();
        
        const matchData = await api.getMatch(matchId);
        if (!isMounted) return;
        
        setMatch(matchData);
        setLoading(false);
        
        // Update scores
        setState(prev => ({
          ...prev,
          modelAScore: matchData.model_a_score,
          modelBScore: matchData.model_b_score,
          gameNumber: matchData.games.length > 0 ? matchData.games.length : 1,
        }));

        // Get current game number from metadata or completed games + 1
        // The current_game from metadata is more accurate for live matches
        const completedGames = matchData.games.length;
        const currentGameNum = matchData.current_game || completedGames + 1;

        try {
          // Use getLiveGame endpoint which can read in-progress moves
          const gameData = await api.getLiveGame(matchId, currentGameNum);
          if (!isMounted) return;
          
          setCurrentGame(gameData);
          
          if (gameData.moves && gameData.moves.length > 0) {
            const lastMove = gameData.moves[gameData.moves.length - 1];
            // Get FEN from last move (the position AFTER the move)
            // We'd need to compute this, but for now use the move info
            setState(prev => ({
              ...prev,
              lastMove: lastMove.move,
              moveCount: gameData.moves.length,
              // Determine who's to move based on move count
              toMove: gameData.moves.length % 2 === 0 ? "model_a" : "model_b",
              modelATime: calculateRemainingTime(gameData.moves, "white"),
              modelBTime: calculateRemainingTime(gameData.moves, "black"),
            }));
            
            // Build FEN from moves
            const fen = buildFenFromMoves(gameData.moves);
            if (fen) {
              setState(prev => ({ ...prev, fen }));
            }
          }
        } catch (gameErr) {
          // Game might not exist yet if match just started
          console.log("Game not available yet:", gameErr);
        }

        setConnected(true);
      } catch (err) {
        if (!isMounted) return;
        console.error("Failed to fetch match:", err);
        setError("Failed to load match");
        setConnected(false);
      }
    };

    // Initial fetch
    fetchMatchData();
    
    // Poll every 3 seconds for updates
    const interval = setInterval(fetchMatchData, 3000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [matchId]);

  // Calculate remaining time from move history
  function calculateRemainingTime(moves: GameDetail["moves"], color: string): number {
    const colorMoves = moves.filter(m => m.color === color);
    if (colorMoves.length === 0) return state.modelATime; // Initial time
    
    const lastColorMove = colorMoves[colorMoves.length - 1];
    return lastColorMove.time_remaining;
  }

  // Build FEN from move history (simplified - just track position)
  function buildFenFromMoves(moves: GameDetail["moves"]): string | null {
    if (moves.length === 0) return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    // Get the FEN from the last move's fen_before, then we'd need to apply the move
    // For now, use a simple approach - just return the fen_before of the last move
    // and note that the actual position is one move ahead
    const lastMove = moves[moves.length - 1];
    if (lastMove.fen_before) {
      // This is the position BEFORE the last move
      // Ideally we'd compute the position after, but this gives us a close approximation
      return lastMove.fen_before;
    }
    
    return null;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading match...</div>
      </div>
    );
  }

  if (error || !match) {
    return (
      <div className="text-center py-16">
        <h1 className="text-2xl font-bold text-gray-300">Match not found</h1>
        <p className="text-gray-500 mt-2">{error}</p>
        <Link href="/live" className="text-arena-accent hover:underline mt-4 block">
          ← Back to live matches
        </Link>
      </div>
    );
  }

  // Parse time control
  const timeControl = match?.time_control || "300+3";
  const [initialTime] = timeControl.split("+").map(Number);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Link
            href="/live"
            className="text-sm text-gray-400 hover:text-white transition-colors mb-2 block"
          >
            ← All Live Matches
          </Link>
          <div className="flex items-center gap-3">
            <span
              className={`w-3 h-3 rounded-full ${
                connected ? "bg-green-500" : "bg-red-500 animate-pulse"
              }`}
            />
            <h1 className="text-2xl font-bold">
              Live: {modelA} vs {modelB}
            </h1>
          </div>
          <div className="text-sm text-gray-500 mt-1">
            {timeControl} • {match?.rethinking_enabled ? "Rethinking ON" : "Rethinking OFF"}
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-400">Game {state.gameNumber}</div>
          <div className="font-mono text-lg">
            {state.modelAScore} - {state.modelBScore}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Board */}
        <div className="lg:col-span-2">
          <div className="card p-6">
            {/* Player Bar - Model A (top if black) */}
            <div className="flex items-center justify-between mb-4 p-3 rounded-lg bg-arena-border/50">
              <div className="flex items-center gap-3">
                <span className="w-4 h-4 bg-gray-800 rounded border border-gray-600" />
                <span className="font-medium">{modelA}</span>
              </div>
              <div className="font-mono text-xl font-bold">
                {formatTime(state.modelATime)}
              </div>
            </div>

            {/* Chess Board */}
            <div className="flex justify-center">
              <ChessBoardComponent
                fen={state.fen}
                lastMove={state.lastMove || undefined}
                size={480}
              />
            </div>

            {/* Player Bar - Model B (bottom if white) */}
            <div className="flex items-center justify-between mt-4 p-3 rounded-lg bg-arena-border/50">
              <div className="flex items-center gap-3">
                <span className="w-4 h-4 bg-white rounded" />
                <span className="font-medium">{modelB}</span>
              </div>
              <div className="font-mono text-xl font-bold">
                {formatTime(state.modelBTime)}
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Match Score */}
          <div className="card p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              Match Score
            </h3>
            <div className="flex items-center justify-between">
              <div className="text-center">
                <div className="font-bold">{modelA}</div>
                <div className="text-3xl font-mono font-bold text-arena-accent">
                  {state.modelAScore}
                </div>
              </div>
              <div className="text-2xl text-gray-500">-</div>
              <div className="text-center">
                <div className="font-bold">{modelB}</div>
                <div className="text-3xl font-mono font-bold text-arena-accent">
                  {state.modelBScore}
                </div>
              </div>
            </div>
          </div>

          {/* Current Game Info */}
          <div className="card p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              Current Game
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Game:</span>
                <span>#{state.gameNumber}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Moves:</span>
                <span>{state.moveCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">To move:</span>
                <span className="capitalize">
                  {state.toMove === "model_a" ? modelA : modelB}
                </span>
              </div>
            </div>
          </div>

          {/* Recent Moves */}
          {currentGame && currentGame.moves && currentGame.moves.length > 0 && (
            <div className="card p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-3">
                Recent Moves
              </h3>
              <div className="space-y-1 text-sm font-mono max-h-40 overflow-y-auto">
                {currentGame.moves.slice(-10).map((move, i) => (
                  <div 
                    key={i} 
                    className={`flex justify-between py-1 ${
                      i === currentGame.moves.length - 1 ? "text-arena-accent font-bold" : "text-gray-400"
                    }`}
                  >
                    <span>{move.move_number}. {move.move}</span>
                    <span className="text-gray-600">{move.time_taken.toFixed(1)}s</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Connection Status */}
          <div className="card p-4">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  connected ? "bg-green-500" : "bg-red-500"
                }`}
              />
              <span className="text-sm text-gray-400">
                {connected ? "Connected • Polling every 3s" : "Reconnecting..."}
              </span>
            </div>
            {state.moveCount > 0 && (
              <div className="text-xs text-gray-600 mt-2">
                Last update: {state.moveCount} moves
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const tenths = Math.floor((seconds % 1) * 10);

  if (seconds < 10) {
    return `${mins}:${secs.toString().padStart(2, "0")}.${tenths}`;
  }
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

