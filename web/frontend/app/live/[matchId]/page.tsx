"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Chess } from "chess.js";
import { ChessBoardComponent } from "@/components/chess/ChessBoard";
import { api, MatchDetail, GameDetail, ProcessDetail } from "@/lib/api";

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
  const [showLogs, setShowLogs] = useState(false);
  const [processDetail, setProcessDetail] = useState<ProcessDetail | null>(null);
  const [processPid, setProcessPid] = useState<number | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const hasLoadedMatch = useRef(false);
  
  const modelA = match?.model_a || "Model A";
  const modelB = match?.model_b || "Model B";

  // Fetch match details and poll for updates
  useEffect(() => {
    let isMounted = true;

    // Build FEN from move history using chess.js
    const buildFenFromMoves = (moves: GameDetail["moves"]): string => {
      const startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
      if (!moves || moves.length === 0) return startFen;
      
      try {
        const chess = new Chess();
        
        for (const moveRecord of moves) {
          try {
            chess.move(moveRecord.move);
          } catch (e) {
            console.warn("Invalid move in sequence:", moveRecord.move);
            break;
          }
        }
        
        return chess.fen();
      } catch (e) {
        console.error("Error building FEN:", e);
        return startFen;
      }
    };

    // Calculate remaining time from move history
    const calculateRemainingTime = (moves: GameDetail["moves"], color: string, initialTime: number): number => {
      const colorMoves = moves.filter(m => m.color === color);
      if (colorMoves.length === 0) return initialTime;
      
      const lastColorMove = colorMoves[colorMoves.length - 1];
      return lastColorMove.time_remaining;
    };
    
    // Parse initial time from time control string (e.g., "300+3" -> 300)
    const parseInitialTime = (timeControl: string): number => {
      const match = timeControl.match(/^(\d+)/);
      return match ? parseInt(match[1], 10) : 300;
    };

    const fetchMatchData = async () => {
      try {
        // Refresh the match cache first
        await api.refreshMatches();
        
        const matchData = await api.getMatch(matchId);
        if (!isMounted) return;
        
        setMatch(matchData);
        setLoading(false);
        hasLoadedMatch.current = true;

        // Determine which game to show:
        // 1. Try the next game after completed ones (in case it's in progress)
        // 2. Fall back to the last completed game if next game hasn't started
        const completedGames = matchData.games?.length || 0;
        const nextGameNum = completedGames + 1;
        
        let gameData: GameDetail | null = null;
        let gameNum = nextGameNum;
        
        try {
          // Try fetching the next game (in progress)
          gameData = await api.getLiveGame(matchId, nextGameNum);
        } catch {
          // Next game doesn't exist yet - fall back to last completed game
          if (completedGames > 0) {
            gameNum = completedGames;
            try {
              gameData = await api.getGame(matchId, completedGames);
            } catch {
              // No game data available
            }
          }
        }
        if (!isMounted) return;
        
        if (!gameData) {
          console.log("[Live] No game data available for game", nextGameNum);
          return;
        }
        
        console.log(`[Live] Game ${gameData.game_number}: ${gameData.moves?.length || 0} moves`);
        
        setCurrentGame(gameData);
        
        // Update state with game data
        const moves = gameData.moves || [];
        
        // Build FEN from all moves using chess.js
        const fen = buildFenFromMoves(moves);
        const lastMove = moves.length > 0 ? moves[moves.length - 1] : null;
        
        // For odd games (1, 3, 5), model_a is white. For even games (2, 4, 6), model_b is white.
        const actualGameNum = gameData.game_number || gameNum;
        const modelAIsWhite = actualGameNum % 2 === 1;
        const initialTime = parseInitialTime(matchData.time_control || "300+3");
        
        setState({
          fen,
          lastMove: lastMove?.move || null,
          moveCount: moves.length,
          // Who's to move: white moves on even move counts (0, 2, 4...), black on odd (1, 3, 5...)
          toMove: moves.length % 2 === 0 
            ? (modelAIsWhite ? "model_a" : "model_b")
            : (modelAIsWhite ? "model_b" : "model_a"),
          modelATime: modelAIsWhite 
            ? calculateRemainingTime(moves, "white", initialTime)
            : calculateRemainingTime(moves, "black", initialTime),
          modelBTime: modelAIsWhite 
            ? calculateRemainingTime(moves, "black", initialTime)
            : calculateRemainingTime(moves, "white", initialTime),
          gameNumber: actualGameNum,
          modelAScore: matchData.model_a_score,
          modelBScore: matchData.model_b_score,
          thinkingPreview: "",
        });

        setConnected(true);
      } catch (err) {
        if (!isMounted) return;
        console.error("Failed to fetch match:", err);
        if (!hasLoadedMatch.current) {
          setError("Failed to load match");
          setLoading(false);
        }
        setConnected(false);
      }
    };

    // Initial fetch
    fetchMatchData();
    
    // Poll every 2 seconds for faster updates
    const interval = setInterval(fetchMatchData, 1000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchId]);

  // Find and poll the process for this match
  useEffect(() => {
    if (!match) return;
    
    let isMounted = true;

    const findAndPollProcess = async () => {
      try {
        // Find the process matching this match
        if (!processPid) {
          const processes = await api.getRunningProcesses();
          const matchingProcess = processes.find(
            p => p.model_a === match.model_a && p.model_b === match.model_b && p.status === "running"
          );
          if (matchingProcess && isMounted) {
            setProcessPid(matchingProcess.pid);
          }
        }
        
        // Fetch logs if we have a PID
        if (processPid) {
          try {
            const detail = await api.getProcessDetail(processPid);
            if (isMounted) {
              setProcessDetail(detail);
              // Auto-scroll logs if panel is open
              if (showLogs) {
                logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
              }
            }
          } catch {
            // Process may have ended, clear PID to search again
            if (isMounted) {
              setProcessPid(null);
              setProcessDetail(null);
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch process:", err);
      }
    };

    findAndPollProcess();
    const interval = setInterval(findAndPollProcess, 1000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [match, processPid, showLogs]);

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
  
  // Determine which model is white for this game
  const modelAIsWhite = state.gameNumber % 2 === 1;
  const whiteModel = modelAIsWhite ? modelA : modelB;
  const blackModel = modelAIsWhite ? modelB : modelA;
  const whiteTime = modelAIsWhite ? state.modelATime : state.modelBTime;
  const blackTime = modelAIsWhite ? state.modelBTime : state.modelATime;

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
          <div className="text-sm text-gray-500 mt-1 flex items-center gap-2 flex-wrap">
            <span>{timeControl} • {match?.rethinking_enabled ? "Rethinking ON" : "Rethinking OFF"}</span>
            {match?.notes && (
              <span className="relative group cursor-help">
                <span className="text-xs text-gray-400 bg-arena-border/50 px-2 py-0.5 rounded">
                  📝 {match.notes.length > 30 ? match.notes.slice(0, 30) + "..." : match.notes}
                </span>
                <span className="absolute bottom-full left-0 mb-2 px-3 py-2 text-xs text-white bg-gray-800 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-normal pointer-events-none z-20 max-w-[300px]">
                  {match.notes}
                </span>
              </span>
            )}
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
            {/* Player Bar - Black (top) */}
            <div className={`flex items-center justify-between mb-4 p-3 rounded-lg ${
              state.toMove === (modelAIsWhite ? "model_b" : "model_a") 
                ? "bg-arena-accent/20 ring-1 ring-arena-accent" 
                : "bg-arena-border/50"
            }`}>
              <div className="flex items-center gap-3">
                <span className="w-4 h-4 bg-gray-800 rounded border border-gray-600" />
                <span className="font-medium">{blackModel}</span>
                {state.toMove === (modelAIsWhite ? "model_b" : "model_a") && (
                  <span className="text-xs text-arena-accent animate-pulse">thinking...</span>
                )}
              </div>
              <div className="font-mono text-xl font-bold">
                {formatTime(blackTime)}
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

            {/* Player Bar - White (bottom) */}
            <div className={`flex items-center justify-between mt-4 p-3 rounded-lg ${
              state.toMove === (modelAIsWhite ? "model_a" : "model_b") 
                ? "bg-arena-accent/20 ring-1 ring-arena-accent" 
                : "bg-arena-border/50"
            }`}>
              <div className="flex items-center gap-3">
                <span className="w-4 h-4 bg-white rounded" />
                <span className="font-medium">{whiteModel}</span>
                {state.toMove === (modelAIsWhite ? "model_a" : "model_b") && (
                  <span className="text-xs text-arena-accent animate-pulse">thinking...</span>
                )}
              </div>
              <div className="font-mono text-xl font-bold">
                {formatTime(whiteTime)}
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
                {connected ? "Connected • Polling every 2s" : "Reconnecting..."}
              </span>
            </div>
            {state.moveCount > 0 && (
              <div className="text-xs text-gray-600 mt-2">
                Last update: {state.moveCount} moves
              </div>
            )}
          </div>

          {/* Show Logs Button */}
          <button
            onClick={() => setShowLogs(!showLogs)}
            className={`w-full py-2 rounded-lg text-sm font-medium transition-colors ${
              showLogs 
                ? "bg-arena-accent text-white" 
                : "bg-arena-border text-gray-400 hover:bg-gray-700 hover:text-white"
            }`}
          >
            {showLogs ? "▼ Hide Logs" : "▶ Show Live Logs"}
            {processDetail && (
              <span className="ml-2 text-xs opacity-70">
                ({processDetail.logs.length} lines)
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Live Logs Panel (Collapsible) */}
      {showLogs && (
        <div className="card p-4 space-y-3 animate-fade-in">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
              <span>📋</span> Process Logs
              {processDetail && (
                <span className={`px-2 py-0.5 rounded text-xs ${
                  processDetail.status === "running" 
                    ? "bg-blue-500/20 text-blue-400" 
                    : processDetail.status === "completed"
                    ? "bg-green-500/20 text-green-400"
                    : "bg-red-500/20 text-red-400"
                }`}>
                  {processDetail.status}
                </span>
              )}
            </h3>
            {processPid && (
              <span className="text-xs text-gray-600">PID: {processPid}</span>
            )}
          </div>
          
          <div className="bg-black rounded-lg p-4 max-h-80 overflow-y-auto font-mono text-xs">
            {!processDetail ? (
              <div className="text-gray-500 italic">
                {processPid ? "Loading logs..." : "No active process found for this match"}
              </div>
            ) : processDetail.logs.length === 0 ? (
              <div className="text-gray-500 italic">Waiting for output...</div>
            ) : (
              processDetail.logs.map((log, i) => {
                // Strip ANSI color codes
                const cleanLine = log.line.replace(/\u001b\[[0-9;]*m/g, "");
                
                // Determine line color based on content
                const lineColor = 
                  cleanLine.includes("Error") || cleanLine.includes("❌") || cleanLine.includes("failed")
                    ? "text-red-400"
                    : cleanLine.includes("✅") || cleanLine.includes("🎉") || cleanLine.includes("WINNER")
                    ? "text-green-400"
                    : cleanLine.includes("⏰") || cleanLine.includes("Thinking time")
                    ? "text-blue-400"
                    : cleanLine.includes("===") || cleanLine.includes("BLITZ")
                    ? "text-cyan-400 font-bold"
                    : cleanLine.includes("Move ") && cleanLine.includes("turn")
                    ? "text-yellow-400"
                    : cleanLine.includes("Final move:")
                    ? "text-green-300"
                    : cleanLine.startsWith("I") && cleanLine.includes("HTTP Request")
                    ? "text-gray-700"  // Dim the HTTP logs
                    : "text-gray-300";
                
                return (
                  <div key={i} className="py-0.5 leading-relaxed">
                    <span className="text-gray-700 mr-2 select-none">
                      {new Date(log.time).toLocaleTimeString()}
                    </span>
                    <span className={lineColor}>
                      {cleanLine}
                    </span>
                  </div>
                );
              })
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
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

