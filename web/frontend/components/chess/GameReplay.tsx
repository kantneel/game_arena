"use client";

import { useState, useEffect, useCallback } from "react";
import { Chess } from "chess.js";
import { ChessBoardComponent } from "./ChessBoard";
import { MoveRecord } from "@/lib/api";

interface GameReplayProps {
  moves: MoveRecord[];
  whiteModel: string;
  blackModel: string;
  autoPlay?: boolean;
}

export function GameReplay({
  moves,
  whiteModel,
  blackModel,
  autoPlay = false,
}: GameReplayProps) {
  const [game] = useState(new Chess());
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(autoPlay);

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
    }, 1500);

    return () => clearInterval(timer);
  }, [isPlaying, moves.length]);

  const goToStart = () => setCurrentMoveIndex(-1);
  const goBack = () => setCurrentMoveIndex((prev) => Math.max(-1, prev - 1));
  const goForward = () =>
    setCurrentMoveIndex((prev) => Math.min(moves.length - 1, prev + 1));
  const goToEnd = () => setCurrentMoveIndex(moves.length - 1);
  const togglePlay = () => setIsPlaying((prev) => !prev);

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      {/* Board */}
      <div className="flex-shrink-0">
        <ChessBoardComponent
          fen={currentState.fen}
          lastMove={currentState.move || undefined}
          size={440}
        />

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
            </div>
          </div>
        )}

        {/* Move List */}
        <div className="card p-4 max-h-64 overflow-y-auto">
          <h3 className="font-medium mb-2">Moves</h3>
          <div className="font-mono text-sm space-y-1">
            {groupMovesIntoPairs(moves).map((pair, i) => (
              <div
                key={i}
                className="flex gap-2 hover:bg-arena-border/30 px-1 rounded"
              >
                <span className="text-gray-500 w-8">{i + 1}.</span>
                <span
                  className={`w-16 cursor-pointer ${
                    currentMoveIndex === i * 2
                      ? "bg-arena-accent/30 rounded px-1"
                      : ""
                  }`}
                  onClick={() => setCurrentMoveIndex(i * 2)}
                >
                  {pair.white || "..."}
                </span>
                {pair.black && (
                  <span
                    className={`w-16 cursor-pointer ${
                      currentMoveIndex === i * 2 + 1
                        ? "bg-arena-accent/30 rounded px-1"
                        : ""
                    }`}
                    onClick={() => setCurrentMoveIndex(i * 2 + 1)}
                  >
                    {pair.black}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function groupMovesIntoPairs(
  moves: MoveRecord[]
): { white: string | null; black: string | null }[] {
  const pairs: { white: string | null; black: string | null }[] = [];

  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({
      white: moves[i]?.move || null,
      black: moves[i + 1]?.move || null,
    });
  }

  return pairs;
}

