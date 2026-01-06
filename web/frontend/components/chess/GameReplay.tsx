"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { Chess } from "chess.js";
import { ChessBoardComponent } from "./ChessBoard";
import { MoveRecord } from "@/lib/api";

interface GameReplayProps {
  moves: MoveRecord[];
  whiteModel: string;
  blackModel: string;
  autoPlay?: boolean;
}

// Classify move quality based on centipawn loss
function getMoveQuality(cpLoss: number | null): {
  label: string;
  color: string;
  bgColor: string;
  emoji: string;
} {
  if (cpLoss === null) {
    return { label: "Unknown", color: "text-gray-400", bgColor: "bg-gray-700", emoji: "❓" };
  }
  if (cpLoss < 0) {
    return { label: "Excellent", color: "text-emerald-400", bgColor: "bg-emerald-500/20", emoji: "✨" };
  }
  if (cpLoss <= 10) {
    return { label: "Best", color: "text-green-400", bgColor: "bg-green-500/20", emoji: "✓" };
  }
  if (cpLoss <= 25) {
    return { label: "Good", color: "text-lime-400", bgColor: "bg-lime-500/20", emoji: "👍" };
  }
  if (cpLoss <= 50) {
    return { label: "Inaccuracy", color: "text-yellow-400", bgColor: "bg-yellow-500/20", emoji: "⚠️" };
  }
  if (cpLoss <= 100) {
    return { label: "Mistake", color: "text-orange-400", bgColor: "bg-orange-500/20", emoji: "❌" };
  }
  return { label: "Blunder", color: "text-red-400", bgColor: "bg-red-500/20", emoji: "💥" };
}

// Get color for CP loss bar
function getCpLossBarColor(cpLoss: number | null): string {
  if (cpLoss === null) return "bg-gray-600";
  if (cpLoss < 0) return "bg-emerald-500";
  if (cpLoss <= 10) return "bg-green-500";
  if (cpLoss <= 25) return "bg-lime-500";
  if (cpLoss <= 50) return "bg-yellow-500";
  if (cpLoss <= 100) return "bg-orange-500";
  return "bg-red-500";
}

export function GameReplay({
  moves,
  whiteModel,
  blackModel,
  autoPlay = false,
}: GameReplayProps) {
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [playbackSpeed, setPlaybackSpeed] = useState(1500);

  // Check if analysis is available
  const hasAnalysis = useMemo(() => {
    return moves.some((m) => m.centipawn_loss !== null);
  }, [moves]);

  // Build game state from moves
  const gameStates = useCallback(() => {
    const chess = new Chess();
    const states: { fen: string; move: string | null }[] = [
      { fen: chess.fen(), move: null },
    ];

    for (const moveRecord of moves) {
      try {
        chess.move(moveRecord.move);
        states.push({ fen: chess.fen(), move: moveRecord.move });
      } catch (e) {
        console.error("Invalid move:", moveRecord.move);
        break;
      }
    }

    return states;
  }, [moves]);

  const states = gameStates();
  const currentState = states[currentMoveIndex + 1] || states[0];
  const currentMoveData = currentMoveIndex >= 0 ? moves[currentMoveIndex] : null;

  // Compute stats for the quality distribution
  const qualityStats = useMemo(() => {
    if (!hasAnalysis) return null;
    
    const analyzed = moves.filter((m) => m.centipawn_loss !== null);
    const blunders = analyzed.filter((m) => (m.centipawn_loss ?? 0) >= 100);
    const mistakes = analyzed.filter((m) => (m.centipawn_loss ?? 0) >= 50 && (m.centipawn_loss ?? 0) < 100);
    const inaccuracies = analyzed.filter((m) => (m.centipawn_loss ?? 0) >= 25 && (m.centipawn_loss ?? 0) < 50);
    const good = analyzed.filter((m) => (m.centipawn_loss ?? 0) < 25);
    
    const avgCpLoss = analyzed.length > 0 
      ? analyzed.reduce((sum, m) => sum + (m.centipawn_loss ?? 0), 0) / analyzed.length 
      : 0;
    
    return { 
      total: analyzed.length,
      blunders: blunders.length,
      mistakes: mistakes.length,
      inaccuracies: inaccuracies.length,
      good: good.length,
      avgCpLoss,
    };
  }, [moves, hasAnalysis]);

  // Auto-play
  useEffect(() => {
    if (!isPlaying) return;

    const timer = setInterval(() => {
      setCurrentMoveIndex((prev) => {
        if (prev >= moves.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, playbackSpeed);

    return () => clearInterval(timer);
  }, [isPlaying, moves.length, playbackSpeed]);

  const goToStart = () => setCurrentMoveIndex(-1);
  const goBack = () => setCurrentMoveIndex((prev) => Math.max(-1, prev - 1));
  const goForward = () =>
    setCurrentMoveIndex((prev) => Math.min(moves.length - 1, prev + 1));
  const goToEnd = () => setCurrentMoveIndex(moves.length - 1);
  const togglePlay = () => setIsPlaying((prev) => !prev);

  const currentQuality = currentMoveData ? getMoveQuality(currentMoveData.centipawn_loss) : null;

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      {/* Board */}
      <div className="flex-shrink-0">
        <ChessBoardComponent
          fen={currentState.fen}
          lastMove={currentState.move || undefined}
          size={440}
        />

        {/* Quality Indicator Bar - shows during playback */}
        {currentMoveData && hasAnalysis && (
          <div className="mt-3 px-2">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-gray-400">Move Quality:</span>
              {currentQuality && (
                <span className={`text-xs font-medium ${currentQuality.color}`}>
                  {currentQuality.emoji} {currentQuality.label}
                </span>
              )}
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all duration-300 ${getCpLossBarColor(currentMoveData.centipawn_loss)}`}
                style={{ 
                  width: `${Math.max(5, 100 - Math.min(100, Math.abs(currentMoveData.centipawn_loss ?? 0)))}%` 
                }}
              />
            </div>
            {currentMoveData.centipawn_loss !== null && (
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>CP Loss: {currentMoveData.centipawn_loss.toFixed(0)}</span>
                {currentMoveData.best_move && currentMoveData.move !== currentMoveData.best_move && (
                  <span>Best: {currentMoveData.best_move}</span>
                )}
              </div>
            )}
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-2 mt-4">
          <button
            onClick={goToStart}
            className="p-2 rounded-lg bg-arena-border hover:bg-arena-accent/20 transition-colors"
            title="Go to start"
          >
            ⏮
          </button>
          <button
            onClick={goBack}
            className="p-2 rounded-lg bg-arena-border hover:bg-arena-accent/20 transition-colors"
            title="Previous move"
          >
            ◀
          </button>
          <button
            onClick={togglePlay}
            className="px-4 py-2 rounded-lg bg-arena-accent hover:bg-arena-accent-dim transition-colors"
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
          <button
            onClick={goForward}
            className="p-2 rounded-lg bg-arena-border hover:bg-arena-accent/20 transition-colors"
            title="Next move"
          >
            ▶
          </button>
          <button
            onClick={goToEnd}
            className="p-2 rounded-lg bg-arena-border hover:bg-arena-accent/20 transition-colors"
            title="Go to end"
          >
            ⏭
          </button>
        </div>

        {/* Playback speed */}
        <div className="flex items-center justify-center gap-2 mt-2">
          <span className="text-xs text-gray-400">Speed:</span>
          {[2000, 1500, 1000, 500].map((speed) => (
            <button
              key={speed}
              onClick={() => setPlaybackSpeed(speed)}
              className={`px-2 py-1 text-xs rounded ${
                playbackSpeed === speed 
                  ? "bg-arena-accent text-white" 
                  : "bg-gray-700 text-gray-400 hover:bg-gray-600"
              }`}
            >
              {speed === 2000 ? "0.5x" : speed === 1500 ? "1x" : speed === 1000 ? "1.5x" : "2x"}
            </button>
          ))}
        </div>

        <div className="text-center text-sm text-gray-400 mt-2">
          Move {currentMoveIndex + 1} of {moves.length}
        </div>
      </div>

      {/* Move List & Info */}
      <div className="flex-1 space-y-4">
        {/* Player Info */}
        <div className="card p-4">
          <div className="flex justify-between items-center mb-2">
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 bg-white rounded-sm" />
              <span className="font-medium">{whiteModel}</span>
            </div>
          </div>
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 bg-gray-800 rounded-sm border border-gray-600" />
              <span className="font-medium">{blackModel}</span>
            </div>
          </div>
        </div>

        {/* Quality Summary */}
        {hasAnalysis && qualityStats && (
          <div className="card p-4">
            <h3 className="font-medium mb-3 flex items-center gap-2">
              <span>♟️</span> Move Quality Summary
            </h3>
            <div className="grid grid-cols-4 gap-2 text-center text-xs mb-3">
              <div className="bg-green-500/20 rounded p-2">
                <div className="text-green-400 font-bold">{qualityStats.good}</div>
                <div className="text-gray-400">Good</div>
              </div>
              <div className="bg-yellow-500/20 rounded p-2">
                <div className="text-yellow-400 font-bold">{qualityStats.inaccuracies}</div>
                <div className="text-gray-400">Inaccuracies</div>
              </div>
              <div className="bg-orange-500/20 rounded p-2">
                <div className="text-orange-400 font-bold">{qualityStats.mistakes}</div>
                <div className="text-gray-400">Mistakes</div>
              </div>
              <div className="bg-red-500/20 rounded p-2">
                <div className="text-red-400 font-bold">{qualityStats.blunders}</div>
                <div className="text-gray-400">Blunders</div>
              </div>
            </div>
            <div className="text-sm text-gray-400">
              Average CP Loss: <span className="text-white font-medium">{qualityStats.avgCpLoss.toFixed(1)}</span>
            </div>
          </div>
        )}

        {/* Current Move Info */}
        {currentMoveData && (
          <div className="card p-4">
            <h3 className="font-medium mb-2">Move Details</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-400">Player:</div>
              <div>{currentMoveData.player}</div>
              <div className="text-gray-400">Move:</div>
              <div className="font-mono">{currentMoveData.move}</div>
              <div className="text-gray-400">Time taken:</div>
              <div>{currentMoveData.time_taken.toFixed(1)}s</div>
              <div className="text-gray-400">Time remaining:</div>
              <div>{formatTime(currentMoveData.time_remaining)}</div>
              {currentMoveData.thinking_tokens && (
                <>
                  <div className="text-gray-400">Thinking tokens:</div>
                  <div>{currentMoveData.thinking_tokens.toLocaleString()}</div>
                </>
              )}
              
              {/* Stockfish Analysis */}
              {hasAnalysis && (
                <>
                  <div className="col-span-2 border-t border-gray-700 my-2" />
                  <div className="text-gray-400">CP Loss:</div>
                  <div className={currentQuality?.color}>
                    {currentMoveData.centipawn_loss !== null 
                      ? `${currentMoveData.centipawn_loss.toFixed(0)} ${currentQuality?.emoji}`
                      : "—"}
                  </div>
                  <div className="text-gray-400">Quality:</div>
                  <div className={currentQuality?.color}>
                    {currentQuality?.label ?? "—"}
                  </div>
                  {currentMoveData.best_move && currentMoveData.move !== currentMoveData.best_move && (
                    <>
                      <div className="text-gray-400">Best move:</div>
                      <div className="font-mono text-cyan-400">{currentMoveData.best_move}</div>
                    </>
                  )}
                  {currentMoveData.win_probability_loss !== null && (
                    <>
                      <div className="text-gray-400">WP Loss:</div>
                      <div className={currentMoveData.win_probability_loss > 0.05 ? "text-red-400" : "text-gray-300"}>
                        {(currentMoveData.win_probability_loss * 100).toFixed(1)}%
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Move List with Quality Indicators */}
        <div className="card p-4 max-h-80 overflow-y-auto">
          <h3 className="font-medium mb-2">Moves</h3>
          <div className="font-mono text-sm space-y-1">
            {groupMovesIntoPairs(moves).map((pair, i) => (
              <div
                key={i}
                className="flex gap-2 hover:bg-arena-border/30 px-1 rounded items-center"
              >
                <span className="text-gray-500 w-8">{i + 1}.</span>
                <MoveCell 
                  move={pair.white?.move ?? null}
                  cpLoss={pair.white?.centipawn_loss ?? null}
                  isSelected={currentMoveIndex === i * 2}
                  onClick={() => setCurrentMoveIndex(i * 2)}
                  hasAnalysis={hasAnalysis}
                />
                {pair.black && (
                  <MoveCell 
                    move={pair.black.move}
                    cpLoss={pair.black.centipawn_loss}
                    isSelected={currentMoveIndex === i * 2 + 1}
                    onClick={() => setCurrentMoveIndex(i * 2 + 1)}
                    hasAnalysis={hasAnalysis}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Correlation hint */}
        {hasAnalysis && (
          <div className="text-xs text-gray-500 italic">
            💡 Watch for patterns: Do moves with less thinking time correlate with lower quality?
          </div>
        )}
      </div>
    </div>
  );
}

// Individual move cell with quality indicator
function MoveCell({
  move,
  cpLoss,
  isSelected,
  onClick,
  hasAnalysis,
}: {
  move: string | null;
  cpLoss: number | null;
  isSelected: boolean;
  onClick: () => void;
  hasAnalysis: boolean;
}) {
  if (!move) return <span className="w-20">...</span>;
  
  const quality = getMoveQuality(cpLoss);
  
  return (
    <span
      className={`w-20 cursor-pointer flex items-center gap-1 ${
        isSelected ? "bg-arena-accent/30 rounded px-1" : ""
      }`}
      onClick={onClick}
    >
      {hasAnalysis && cpLoss !== null && (
        <span 
          className={`w-1.5 h-1.5 rounded-full ${getCpLossBarColor(cpLoss)}`}
          title={`${quality.label}: ${cpLoss.toFixed(0)} CP`}
        />
      )}
      <span className={hasAnalysis && cpLoss !== null && cpLoss >= 100 ? "text-red-400" : ""}>
        {move}
      </span>
    </span>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function groupMovesIntoPairs(
  moves: MoveRecord[]
): { white: MoveRecord | null; black: MoveRecord | null }[] {
  const pairs: { white: MoveRecord | null; black: MoveRecord | null }[] = [];

  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({
      white: moves[i] || null,
      black: moves[i + 1] || null,
    });
  }

  return pairs;
}
